import functools
from typing import Any, List

import numpy as np
import os
import os.path as osp
import torch
import torch.nn.functional as F
import torch.optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch3d.renderer.cameras import PerspectiveCameras

from config.args_config import default_argument_parser, setup_cfg
from nnutils.logger import MyLogger
from datasets import build_dataloader
from models import dec, enc
from nnutils.hand_utils import ManopthWrapper
from nnutils import geom_utils, mesh_utils, slurm_utils


def get_hTx(frame, batch):
    hTn = geom_utils.inverse_rt(batch['nTh'])
    hTx = hTn
    return hTx


class IHoi(pl.LightningModule):
    def __init__(self, cfg, **kwargs) -> None:
        super().__init__()
        # self.hparams = cfg
        # self.hparams.update(cfg)
        # self.cfg = cfg

        cfg = {'CAMERA': {'F': 100.0}, 'DB': {'CACHE': True, 'CLS': 'sdf_img', 'DIR': '/glusterfs/yufeiy2/fair/data/', 'IMAGE': False, 'INPUT': 'rgba', 'JIT_ART': 0.2, 'JIT_P': 0, 'JIT_SCALE': 0.5, 'JIT_TRANS': 0.2, 'NAME': 'rhoi', 'NUM_POINTS': 4096, 'RADIUS': 0.2, 'TESTNAME': 'rhoi', 'DET_TH': 0.0, 'DT': 1, 'GT': 'none', 'IOU_TH': 0.4, 'T0': 0, 'TIME': 10, 'VOX_RESO': 32}, 'DUM': '', 'EXP': 'aug', 'GPU': 0, 'HAND': {'MANO_PATH': '../data/smplx/mano', 'WRAP': 'mano'}, 'LOSS': {'ENFORCE_MINMAX': True, 'KL': 0.0001, 'OCC': 'strict', 'OFFSCREEN': 'gt', 'RECON': 1.0, 'SDF_MINMAX': 0.1, 'SCALE': 1.0, 'SO3': 1.0, 'TRANS': 1.0, 'VIEW': 10.0, 'SMOOTH': 0.0, 'CONTACT': 0.0, 'REPULSE': 0.0}, 'MODEL': {'BATCH_SIZE': 1, 'DEC': 'PixCoord', 'ENC': 'ImageSpEnc', 'ENC_RESO': -3, 'FRAME': 'norm', 'FREQ': 10, 'GRAD': 'none', 'IS_PCA': 0, 'LATENT_DIM': 128, 'NAME': 'IHoi', 'PC_DIM': 128, 'SDF': {'DIMS': [512, 512, 512, 512, 512, 512, 512, 512], 'GEOMETRIC_INIT': False, 'SKIP_IN': [4], 'th': False}, 'THETA_DIM': 45, 'THETA_EMB': 'pca', 'Z_DIM': 256, 'CLS': 'ft_3d_pifu', 'OCC': 'sdf', 'VOX': {'BLOCKS': [2, 2], 'UP': [1, 2]}, 'VOX_INP': 16}, 'MODEL_PATH': '../output/aug/pifu_MODEL.DECPixCoord/large/rhoi_3dDB.INPUT_rgba_MODEL.SDF.th_False_DB.JIT_ART_0.2', 'MODEL_SIG': 'aug/pifu_MODEL.DECPixCoord', 'OPT': {'BATCH_SIZE': 16, 'INIT': 'zero', 'LR': 0.001, 'NAME': 'opt', 'NET': False, 'OPT': 'adam', 'STEP': 1000}, 'OUTPUT_DIR': '../output/', 'RENDER': {'METRIC': 1}, 'SEED': 123, 'SOLVER': {'BASE_LR': 1e-05}, 'TEST': {'DIR': '', 'NAME': 'default', 'NUM': 2, 'SET': 'test'}, 'TRAIN': {'EPOCH': 20000, 'EVAL_EVERY': 250, 'ITERS': 50000, 'PRINT_EVERY': 100}, 'SLURM': {'NGPU': 1, 'PART': 'devlab', 'RUN': False, 'TIME': 720, 'WORK': 10}, 'VOX_TH': 0.1, 'FT': {'ENC': True, 'DEC': True}}
        self.cfg = cfg

        self.dec = dec.build_net(cfg['MODEL'])  # sdf
        self.enc = enc.build_net(cfg['MODEL']['ENC'], cfg)  # ImageSpEnc(cfg, out_dim=cfg.MODEL.Z_DIM, layer=cfg.MODEL.ENC_RESO, modality=cfg.DB.INPUT)
        self.hand_wrapper = ManopthWrapper()

        self.minT = -0.1
        self.maxT = 0.1
        self.sdf_key = '%sSdf' % cfg['MODEL']['FRAME'][0]
        self.obj_key = '%sObj' % cfg['MODEL']['FRAME'][0]
        self.metric = 'val'
        self._train_loader = None

    def val_dataloader(self):
        test = self.cfg['DB']['NAME'] if self.cfg['DB']['TESTNAME'] == '' else self.cfg['DB']['TESTNAME']
        val_dataloader = build_dataloader(self.cfg, 'test', is_train=False, name=test)
        trainval_dataloader = build_dataloader(self.cfg, 'train', is_train=True, 
            shuffle=False, bs=min(8, self.cfg.MODEL.BATCH_SIZE), name=self.cfg['DB'].NAME)
        return [val_dataloader, trainval_dataloader]

    def get_jsTx(self, hA, hTx):
        hTjs = self.hand_wrapper.pose_to_transform(hA, False)  # (N, J, 4, 4)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx_exp = geom_utils.se3_to_matrix(hTx
                  ).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx_exp        
        return jsTx

    def sdf(self, hA, sdf_hA_jsTx, hTx):
        sdf = functools.partial(sdf_hA_jsTx, hA=hA, jsTx=self.get_jsTx(hA, hTx))
        return sdf


    def forward(self, batch):
        image_feat = self.enc(batch['image'], mask=batch['obj_mask'])  # (N, D, H, W)
        
        hTx = get_hTx(self.cfg['MODEL']['FRAME'], batch)
        xTh = geom_utils.inverse_rt(hTx)
        cTx = geom_utils.compose_se3(batch['cTh'], hTx)
        cameras = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=batch['image'].device)

        with torch.enable_grad():
            sdf_hA_jsTx = functools.partial(self.dec, 
                z=image_feat, cTx=cTx, cam=cameras)
            sdf_hA = functools.partial(self.sdf, sdf_hA_jsTx=sdf_hA_jsTx, hTx=hTx)
            sdf = sdf_hA(batch['hA'])

        out = {
            'sdf': sdf,
            'sdf_hA': sdf_hA,
            'hTx': hTx,
            'xTh': xTh,
        }
        return out

    def step(self, batch, batch_idx):
        image_feat = self.enc(batch['image'], mask=batch['obj_mask'])  # (N, D, H, W)
        
        xXyz = batch[self.sdf_key][..., :3]
        hTx = get_hTx('norm', batch)
        cTx = geom_utils.compose_se3(batch['cTh'], hTx)
        cameras = PerspectiveCameras(batch['cam_f'], batch['cam_p'], device=xXyz.device)

        hTjs = self.hand_wrapper.pose_to_transform(batch['hA'], False)  # (N, J, 4, 4)
        N, num_j, _, _ = hTjs.size()
        jsTh = geom_utils.inverse_rt(mat=hTjs, return_mat=True)
        hTx = geom_utils.se3_to_matrix(hTx
                ).unsqueeze(1).repeat(1, num_j, 1, 1)
        jsTx = jsTh @ hTx

        pred_sdf = self.dec(xXyz, image_feat, batch['hA'], cTx, cameras, jsTx=jsTx, )
        ndcPoints = mesh_utils.transform_points(mesh_utils.apply_transform(xXyz, cTx), cameras)
        
        out = {self.sdf_key: pred_sdf, 'z': image_feat, 'jsTx': jsTx}

        loss, losses = 0., {}
        cfg = self.cfg.LOSS

        recon_loss = self.sdf_loss(pred_sdf, batch[self.sdf_key][..., -1:], ndcPoints,
            cfg.RECON, cfg.ENFORCE_MINMAX, )
        loss = loss + recon_loss
        losses['recon'] = recon_loss

        losses['loss'] = loss
        return losses, out

    def sdf_loss(self, sdf_pred, sdf_gt, ndcPoints, wgt=1, minmax=False, ):
        # recon loss
        mode = 'gt'  # [gt, out, idc]
        if mode == 'gt':
            pass
        elif mode == 'out':
            mask = torch.all(ndcPoints <= 1, dim=-1, keepdim=True) &\
                 torch.all(ndcPoints >= -1, dim=-1, keepdim=True)
            value = self.maxT if 'sdf' == 'sdf' else 1
            sdf_gt = mask * sdf_gt + (~mask) * value
        elif mode == 'idc':
            mask = torch.any(ndcPoints <= 1, dim=-1, keepdim=True) & \
                torch.any(ndcPoints >= -1, dim=-1, keepdim=True)
            sdf_pred = sdf_pred * mask  # the idc region to zero
            sdf_gt = sdf_gt * mask
        else:
            raise NotImplementedError

        if minmax or self.current_epoch >= 20000 // 2:
            sdf_pred = torch.clamp(sdf_pred, self.minT, self.maxT)
            sdf_gt = torch.clamp(sdf_gt, self.minT, self.maxT)
        recon_loss = wgt * F.l1_loss(sdf_pred, sdf_gt)
        return recon_loss


def main(cfg, args):
    pl.seed_everything(123)
    
    model = IHoi(cfg)
    if args.ckpt is not None:
        print('load from', args.ckpt)
        model = model.load_from_checkpoint(args.ckpt, cfg=cfg, strict=False)

    # instantiate model
    if args.eval:
        logger = MyLogger(save_dir='../output/',
                        name=os.path.dirname('aug/pifu_MODEL.DECPixCoord'),
                        version=os.path.basename('aug/pifu_MODEL.DECPixCoord'),
                        subfolder='',
                        resume=True,
                        )
        trainer = pl.Trainer(gpus='0,',
                             default_root_dir='../output/aug/pifu_MODEL.DECPixCoord/large/rhoi_3dDB.INPUT_rgba_MODEL.SDF.th_False_DB.JIT_ART_0.2',
                             logger=logger,
                            #  resume_from_checkpoint=args.ckpt,
                             )
        print('../output/aug/pifu_MODEL.DECPixCoord/large/rhoi_3dDB.INPUT_rgba_MODEL.SDF.th_False_DB.JIT_ART_0.2', trainer.weights_save_path, args.ckpt)

        model.freeze()
        trainer.test(model=model, verbose=False)
    else:
        logger = MyLogger(save_dir='../output/',
                        name=os.path.dirname('aug/pifu_MODEL.DECPixCoord'),
                        version=os.path.basename('aug/pifu_MODEL.DECPixCoord'),
                        subfolder='',
                        resume=args.slurm or args.ckpt is not None,
                        )
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor='val_loss/dataloader_idx_0',
            mode='min',
            save_last=True,
        )
        lr_monitor = LearningRateMonitor()

        max_epoch = 20000 # max(cfg.TRAIN.EPOCH, cfg.TRAIN.ITERS // every_iter)
        trainer = pl.Trainer(
                            # gpus=1,
                             gpus=-1,
                             accelerator='dp',
                             num_sanity_val_steps=1,
                             limit_val_batches=2,
                             check_val_every_n_epoch=250,
                             default_root_dir='../output/aug/pifu_MODEL.DECPixCoord/large/rhoi_3dDB.INPUT_rgba_MODEL.SDF.th_False_DB.JIT_ART_0.2',
                             logger=logger,
                             max_epochs=max_epoch,
                             callbacks=[checkpoint_callback, lr_monitor],
                             progress_bar_refresh_rate=0 if args.slurm else None,            
                             )
        trainer.fit(model)





if __name__ == '__main__':
    arg_parser = default_argument_parser()
    arg_parser = slurm_utils.add_slurm_args(arg_parser)
    args = arg_parser.parse_args()
    
    cfg = setup_cfg(args)
    save_dir = os.path.dirname(cfg.MODEL_PATH)
    slurm_utils.slurm_wrapper(args, save_dir, main, {'args': args, 'cfg': cfg}, resubmit=False)
