import importlib
import logging
import numpy as np
import os
import os.path as osp
import glob

import torch
import torch.nn as nn


def latest_ckpt(prop, include_last=False, till=-1):
    if not prop.endswith('.ckpt'):
        if include_last:
            ckpt = os.path.join(prop, 'checkpoints', 'last.ckpt')
            if os.path.exists(ckpt):
                return ckpt
        ckpt_list = glob.glob(os.path.join(prop, 'checkpoints', 'epoch*.ckpt'))
        print(ckpt_list)
        epoch_list = [int(os.path.basename(e)[len('epoch='):].split('-step')[0]) for e in ckpt_list]
        last_ckpt = os.path.join(prop, 'checkpoints/last.ckpt')
        if len(epoch_list) == 0 and os.path.exists(last_ckpt):
            return last_ckpt
        if len(epoch_list) == 0:
            return None
        inds = np.argmax(epoch_list)
        return ckpt_list[inds]


def get_model_name(cfg,  cli_args, eval, config_file):
    if eval:
        name = os.path.basename(cfg.MODEL_SIG)
    else:
        # name = '%s' % (cfg.DB.NAME)
        # name += '_%s' % (cfg.MODEL.NAME)
        name = '%s' % (osp.basename(config_file).split('.')[0])
    
        skip_list = ['EXP', 'GPU',
                     'TEST.NAME']
        for full_key, v in zip(cli_args[0::2], cli_args[1::2]):
            if full_key in skip_list:
                continue
            name += '_%s%s' % (full_key, str(v))

    return name
    

def load_model(cfg, ckpt_dir, ckpt_epoch) -> nn.Module:
    #cfg = {'CAMERA': {'F': 100.0}, 'DB': {'CACHE': True, 'CLS': 'sdf_img', 'DIR': '/glusterfs/yufeiy2/fair/data/', 'IMAGE': False, 'INPUT': 'rgba', 'JIT_ART': 0.2, 'JIT_P': 0, 'JIT_SCALE': 0.5, 'JIT_TRANS': 0.2, 'NAME': 'rhoi', 'NUM_POINTS': 4096, 'RADIUS': 0.2, 'TESTNAME': 'rhoi', 'DET_TH': 0.0, 'DT': 1, 'GT': 'none', 'IOU_TH': 0.4, 'T0': 0, 'TIME': 10, 'VOX_RESO': 32}, 'DUM': '', 'EXP': 'aug', 'GPU': 0, 'HAND': {'MANO_PATH': '../data/smplx/mano', 'WRAP': 'mano'}, 'LOSS': {'ENFORCE_MINMAX': True, 'KL': 0.0001, 'OCC': 'strict', 'OFFSCREEN': 'gt', 'RECON': 1.0, 'SDF_MINMAX': 0.1, 'SCALE': 1.0, 'SO3': 1.0, 'TRANS': 1.0, 'VIEW': 10.0, 'SMOOTH': 0.0, 'CONTACT': 0.0, 'REPULSE': 0.0}, 'MODEL': {'BATCH_SIZE': 1, 'DEC': 'PixCoord', 'ENC': 'ImageSpEnc', 'ENC_RESO': -3, 'FRAME': 'norm', 'FREQ': 10, 'GRAD': 'none', 'IS_PCA': 0, 'LATENT_DIM': 128, 'NAME': 'IHoi', 'PC_DIM': 128, 'SDF': {'DIMS': [512, 512, 512, 512, 512, 512, 512, 512], 'GEOMETRIC_INIT': False, 'SKIP_IN': [4], 'th': False}, 'THETA_DIM': 45, 'THETA_EMB': 'pca', 'Z_DIM': 256, 'CLS': 'ft_3d_pifu', 'OCC': 'sdf', 'VOX': {'BLOCKS': [2, 2], 'UP': [1, 2]}, 'VOX_INP': 16}, 'MODEL_PATH': '../output/aug/pifu_MODEL.DECPixCoord/large/rhoi_3dDB.INPUT_rgba_MODEL.SDF.th_False_DB.JIT_ART_0.2', 'MODEL_SIG': 'aug/pifu_MODEL.DECPixCoord', 'OPT': {'BATCH_SIZE': 16, 'INIT': 'zero', 'LR': 0.001, 'NAME': 'opt', 'NET': False, 'OPT': 'adam', 'STEP': 1000}, 'OUTPUT_DIR': '../output/', 'RENDER': {'METRIC': 1}, 'SEED': 123, 'SOLVER': {'BASE_LR': 1e-05}, 'TEST': {'DIR': '', 'NAME': 'default', 'NUM': 2, 'SET': 'test'}, 'TRAIN': {'EPOCH': 20000, 'EVAL_EVERY': 250, 'ITERS': 50000, 'PRINT_EVERY': 100}, 'SLURM': {'NGPU': 1, 'PART': 'devlab', 'RUN': False, 'TIME': 720, 'WORK': 10}, 'VOX_TH': 0.1, 'FT': {'ENC': True, 'DEC': True}}
    Model = getattr(importlib.import_module(".ihoi", "models"), 'IHoi')

    model = Model(cfg)
    ckpt = osp.join(ckpt_dir, 'checkpoints', '%s.ckpt' % ckpt_epoch)
    logging.info('load from %s' % ckpt)
    ckpt = torch.load(ckpt)['state_dict']
    load_my_state_dict(model, ckpt)
    
    model.eval()
    model.cuda()
    return model
            

def load_my_state_dict(model: torch.nn.Module, state_dict, lambda_own=lambda x: x):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        own_name = lambda_own(name)
        # own_name = '.'.join(name.split('.')[1:])
        if own_name not in own_state:
            logging.warn('Not found in checkpoint %s %s' % (name, own_name))
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        if param.size() != own_state[own_name].size():
            logging.warn('size not match %s %s %s' % (
                name, str(param.size()), str(own_state[own_name].size())))
            continue
        own_state[own_name].copy_(param)



def to_cuda(data, device='cuda'):
    new_data = {}
    for key in data:
        if hasattr(data[key], 'cuda'):
            new_data[key] = data[key].to(device)
        else:
            new_data[key] = data[key]
    return new_data