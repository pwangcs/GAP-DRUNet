# -*- coding: utf-8 -*-

'''
# ====================

Created on Jul 14 11:46:46 2022

@author: Ping Wang (wangping@westlake.edu.cn)

# ====================
'''

import numpy as np
import torch
import torch.nn as nn
from .models.network_unet import UNetRes
from .utils import utils_model
# from .utils.utils_image import single2tensor4


def video2tensor4(img):
    if img.ndim == 3:
        img = np.expand_dims(img, axis=3)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 3, 0, 1).float()


def drunet_denoiser(vnoisy, sigma, model=None, gray=True):
    x8 = False        # default: False, x8 to boost performance
    n_channels = 1 if gray else 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    if model is None:
		# Create model
		# print('Loading model ...\n')
        model = UNetRes(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv",upsample_mode="convtranspose")
		# Load saved weights
        if gray: 
            model_path = 'packages\drunet\drunet_gray.pth'
        else:
            model_path = 'packages\drunet\drunet_color.pth'
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k,v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    with torch.no_grad():
        # print(vnoisy.shape, sigma.shape)
        vnoisy = video2tensor4(vnoisy)
        vnoisy = torch.cat((vnoisy, torch.FloatTensor([sigma]).repeat(vnoisy.shape[0], 1, vnoisy.shape[2], vnoisy.shape[3])), dim=1)
        vnoisy = vnoisy.to(device)

        if not x8 and vnoisy.size(2)//8==0 and vnoisy.size(3)//8==0:
            outv = model(vnoisy)
        elif not x8 and (vnoisy.size(2)//8!=0 or vnoisy.size(3)//8!=0):
            outv = utils_model.test_mode(model, vnoisy, refield=64, mode=5)
        elif x8:
            outv = utils_model.test_mode(model, vnoisy, mode=3)
        
        outv = outv.data.squeeze().float().clamp_(0, 1).cpu().numpy()
        if outv.ndim == 3:
            outv = np.transpose(outv, (1, 2, 0))

    return outv