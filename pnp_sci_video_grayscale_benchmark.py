#!/usr/bin/env python
# coding: utf-8

# ## GAP-TV for Video Compressive Sensing
# ### GAP-TV
# > X. Yuan, "Generalized alternating projection based total variation minimization for compressive sensing," in *IEEE International Conference on Image Processing (ICIP)*, 2016, pp. 2539-2543.
# ### Code credit
# [Xin Yuan](https://www.bell-labs.com/usr/x.yuan "Dr. Xin Yuan, Bell Labs"), [Bell Labs](https://www.bell-labs.com/), xyuan@bell-labs.com, created Aug 7, 2018.  
# [Yang Liu](https://liuyang12.github.io "Yang Liu, Tsinghua University"), [Tsinghua University](http://www.tsinghua.edu.cn/publish/thu2018en/index.html), y-liu16@mails.tsinghua.edu.cn, updated Jan 20, 2019.

# In[1]:


import os
import time
import math
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from statistics import mean

from pnp_sci import admmdenoise_cacti

from utils import (A_, At_)


# In[2]:


# [0] environment configuration
datasetdir = './dataset/cacti/grayscale_benchmark' # dataset
resultsdir = './results' # results

alldatname = ['kobe32','traffic48','runner40','drop40','crash32','aerial32']
allnframes = [      -1,         -1,         1,       1,       -1,        -1]

for datname, nframe in zip(alldatname, allnframes):
    # datname = 'kobe32'        # name of the dataset
    # datname = 'traffic48'     # name of the dataset
    # datname = 'runner40'      # name of the dataset
    # datname = 'drop40'        # name of the dataset
    # datname = 'crash32'       # name of the dataset
    # datname = 'aerial32'      # name of the dataset
    # datname = 'bicycle24'     # name of the dataset
    # datname = 'starfish48'    # name of the dataset

    # datname = 'starfish_c16_48'    # name of the dataset

    matfile = datasetdir + '/' + datname + '_cacti.mat' # path of the .mat data file


    # In[3]:


    from scipy.io.matlab.mio import _open_file
    from scipy.io.matlab.miobase import get_matfile_version

    # [1] load data
    if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
        file = sio.loadmat(matfile) # for '-v7.2' and lower version of .mat file (MATLAB)
        meas = np.float32(file['meas'])
        mask = np.float32(file['mask'])
        orig = np.float32(file['orig'])
    else: # MATLAB .mat v7.3
        file =  h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
        meas = np.float32(file['meas']).transpose()
        mask = np.float32(file['mask']).transpose()
        orig = np.float32(file['orig']).transpose()

    print(meas.shape, mask.shape, orig.shape)

    ############### Permutate Mask Partly#########
    # mask[100:125,100:125,:] = np.random.permutation(mask[100:125,100:125,:].reshape(-1)).reshape(25,25,-1)
    # print(meas.shape, mask.shape, orig.shape)



    iframe = 0
    # nframe = 1
    # nframe = meas.shape[2]
    if nframe < 0:
        nframe = meas.shape[2]
    MAXB = 255.

    # common parameters and pre-calculation for PnP
    # define forward model and its transpose
    A  = lambda x :  A_(x, mask) # forward model function handle
    At = lambda y : At_(y, mask) # transpose of forward model

    mask_sum = np.sum(mask, axis=2)
    mask_sum[mask_sum==0] = 1


    # In[4]:

    import torch
    from packages.drunet.models.network_unet import UNetRes
    ## [2.2] GAP-DRUNet
    projmeth = 'gap' # projection method
    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of GAP
    denoiser = 'drunet' # video non-local network 
    noise_estimate = False # disable noise estimation for GAP
    # sigma = [50/255, 25/255, 12/255, 6/255] 
    sigma = [100/255, 50/255, 25/255, 12/255]# pre-set noise standard deviation
    iter_max = [20, 20, 20, 20] # maximum number of iterations
    n_channels = 1
    # sigma    = [12/255] # pre-set noise standard deviation
    # iter_max = [20] # maximum number of iterations
    useGPU = True # use GPU

    # pre-load the model for drunet image denoising
    # NUM_IN_FR_EXT = 5 # temporal size of patch
    model = UNetRes(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv",upsample_mode="convtranspose")
    # Load saved weights
    state_temp_dict = torch.load('./packages/drunet/drunet_gray.pth')
    if useGPU:
        device_ids = [0]
        # model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
        model = model.cuda()
    # else:
        # # CPU mode: remove the DataParallel wrapper
        # state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)

    model.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model.eval()

    vgapdrunet,tgapdrunet,psnr_gapdrunet,ssim_gapdrunet,psnrall_gapdrunet = admmdenoise_cacti(meas, mask, A, At,
                                              projmeth=projmeth, v0=None, orig=orig,
                                              iframe=iframe, nframe=nframe,
                                              MAXB=MAXB, maskdirection='plain',
                                              _lambda=_lambda, accelerate=accelerate, 
                                              denoiser=denoiser, model=model, 
                                              iter_max=iter_max, sigma=sigma)

    print('{}-{} PSNR {:2.2f} dB, SSIM {:.4f}, running time {:.1f} seconds.'.format(
        projmeth.upper(), denoiser.upper(), mean(psnr_gapdrunet), mean(ssim_gapdrunet), tgapdrunet))


    # In[8]:


    # [3.3] result demonstration of GAP-Denoise
    nmask = mask.shape[2]
    
    savedmatdir = resultsdir + '/savedmat/cacti/'
    if not os.path.exists(savedmatdir):
        os.makedirs(savedmatdir)
    # sio.savemat('{}gap{}_{}{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask),
    #             {'vgapdenoise':vgapdenoise},{'psnr_gapdenoise':psnr_gapdenoise})
    sio.savemat('{}gap{}_{}_{:d}_sigma{:d}.mat'.format(savedmatdir,denoiser.lower(),datname,nmask,int(sigma[-1]*MAXB)),
                {
                #  'vgaptv':vgaptv, 
                #  'psnr_gaptv':psnr_gaptv,
                #  'ssim_gaptv':ssim_gaptv,
                #  'psnrall_tv':psnrall_gaptv,
                #  'tgaptv':tgaptv,
                #  'vgapffdnet':vgapffdnet, 
                #  'psnr_gapffdnet':psnr_gapffdnet,
                #  'ssim_gapffdnet':ssim_gapffdnet,
                #  'psnrall_ffdnet':psnrall_gapffdnet,
                #  'tgapffdnet':tgapffdnet,
                 'vgapdrunet':vgapdrunet, 
                 'psnr_gapdrunet':psnr_gapdrunet,
                 'ssim_gapdrunet':ssim_gapdrunet,
                 'psnrall_drunet':psnrall_gapdrunet,
                 'tgapdrunet':tgapdrunet
                 })

