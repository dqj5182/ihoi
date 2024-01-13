import functools
from typing import Any, List

import numpy as np
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from pytorch3d.renderer.cameras import PerspectiveCameras

from models.dec import PixCoord
from models.enc import ImageSpEnc
from nnutils.hand_utils import ManopthWrapper
from nnutils import geom_utils


def get_hTx(batch):
    hTn = geom_utils.inverse_rt(batch['nTh'])
    hTx = hTn
    return hTx


class IHoi(nn.Module):
    def __init__(self, cfg, **kwargs) -> None:
        super().__init__()
        cfg = {'CAMERA': {'F': 100.0}, 'DB': {'CACHE': True, 'CLS': 'sdf_img', 'DIR': '/glusterfs/yufeiy2/fair/data/', 'IMAGE': False, 'INPUT': 'rgba', 'JIT_ART': 0.2, 'JIT_P': 0, 'JIT_SCALE': 0.5, 'JIT_TRANS': 0.2, 'NAME': 'rhoi', 'NUM_POINTS': 4096, 'RADIUS': 0.2, 'TESTNAME': 'rhoi', 'DET_TH': 0.0, 'DT': 1, 'GT': 'none', 'IOU_TH': 0.4, 'T0': 0, 'TIME': 10, 'VOX_RESO': 32}, 'DUM': '', 'EXP': 'aug', 'GPU': 0, 'HAND': {'MANO_PATH': '../data/smplx/mano', 'WRAP': 'mano'}, 'LOSS': {'ENFORCE_MINMAX': True, 'KL': 0.0001, 'OCC': 'strict', 'OFFSCREEN': 'gt', 'RECON': 1.0, 'SDF_MINMAX': 0.1, 'SCALE': 1.0, 'SO3': 1.0, 'TRANS': 1.0, 'VIEW': 10.0, 'SMOOTH': 0.0, 'CONTACT': 0.0, 'REPULSE': 0.0}, 'MODEL': {'BATCH_SIZE': 1, 'DEC': 'PixCoord', 'ENC': 'ImageSpEnc', 'ENC_RESO': -3, 'FRAME': 'norm', 'FREQ': 10, 'GRAD': 'none', 'IS_PCA': 0, 'LATENT_DIM': 128, 'NAME': 'IHoi', 'PC_DIM': 128, 'SDF': {'DIMS': [512, 512, 512, 512, 512, 512, 512, 512], 'GEOMETRIC_INIT': False, 'SKIP_IN': [4], 'th': False}, 'THETA_DIM': 45, 'THETA_EMB': 'pca', 'Z_DIM': 256, 'CLS': 'ft_3d_pifu', 'OCC': 'sdf', 'VOX': {'BLOCKS': [2, 2], 'UP': [1, 2]}, 'VOX_INP': 16}, 'MODEL_PATH': '../output/aug/pifu_MODEL.DECPixCoord/large/rhoi_3dDB.INPUT_rgba_MODEL.SDF.th_False_DB.JIT_ART_0.2', 'MODEL_SIG': 'aug/pifu_MODEL.DECPixCoord', 'OPT': {'BATCH_SIZE': 16, 'INIT': 'zero', 'LR': 0.001, 'NAME': 'opt', 'NET': False, 'OPT': 'adam', 'STEP': 1000}, 'OUTPUT_DIR': '../output/', 'RENDER': {'METRIC': 1}, 'SEED': 123, 'SOLVER': {'BASE_LR': 1e-05}, 'TEST': {'DIR': '', 'NAME': 'default', 'NUM': 2, 'SET': 'test'}, 'TRAIN': {'EPOCH': 20000, 'EVAL_EVERY': 250, 'ITERS': 50000, 'PRINT_EVERY': 100}, 'SLURM': {'NGPU': 1, 'PART': 'devlab', 'RUN': False, 'TIME': 720, 'WORK': 10}, 'VOX_TH': 0.1, 'FT': {'ENC': True, 'DEC': True}}
        self.cfg = cfg
        self.enc = ImageSpEnc(out_dim=256, layer=-2, modality='rgba') # ImageSpEnc(cfg, out_dim=cfg.MODEL.Z_DIM, layer=cfg.MODEL.ENC_RESO, modality=cfg.DB.INPUT)
        self.dec = PixCoord(cfg['MODEL'], 256, 45, 10)  # sdf
        self.hand_wrapper = ManopthWrapper()

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
        
        hTx = get_hTx(batch)
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