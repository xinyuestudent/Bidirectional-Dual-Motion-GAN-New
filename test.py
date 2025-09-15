'''
@xinyuewang
The implementation code of the paper "Bidirectional Dual Motion GAN for Video Frame Prediction in Intelligent Driving" with Pytorch
data: Sep,12,2025
'''
import sys
sys.path.append('./Mono/')
import argparse
import PIL
PIL.PILLOW_VERSION = PIL.__version__
import os
import numpy as np
import math
import itertools
import datetime
import time
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import metric
import KL_flow as KL
import cv2
import argparse
import lpips.lpips as lpips
from pytorch_msssim0 import ssim,ms_ssim,SSIM,MS_SSIM
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from models import *
from datasets import *
from utils import *
import Mono.utilsmono as monout
from Mono.main_monodepth_pytorch import Model as MonoDepthModel
from laplacian_of_guassian import *
from flownet2.models import FlowNet2C  # the path is depended on where you create this module
from flownet2.utils.frame_utils import read_gen  # the path is depended on where you create this module
import time
from tqdm import tqdm
import scipy
import LPIPDISTS as LD
import doiprocess as doi

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=15, help="epoch to start training from")
parser.add_argument("--img_height", type=int, default=128, help="size of image height") #当图片的宽和高设置为128的时候会导致内存溢出
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--sequence_len", type=float, default=5, help="the original length of frame sequence")

opt = parser.parse_args()##
print(opt)

cuda = torch.cuda.is_available()

# Image transformations
transforms_ = [
    transforms.Resize((int(opt.img_height),int(opt.img_width)), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    ImageTestDataset("./dataset/pickle_data/real_test_cal_data.pkl", transforms_=transforms_,nt=opt.sequence_len),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=0,
)
input_shape = (opt.batch_size, opt.sequence_len, opt.channels, opt.img_height, opt.img_width)
# print("=====output of the Generator=====")
G = GeneratorResNet(input_shape, opt.n_residual_blocks)
G = torch.nn.DataParallel(G, device_ids=range(torch.cuda.device_count()))

if cuda:
    G = G.cuda()

if cuda:
    G.load_state_dict(torch.load("../RetrospectiveCycleGAN/saved_models/real_cal_0/G_future_%d.pth" %  opt.epoch))
#else:  /home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/saved_models/no_single_net_/G_snigle_net__future_12.pth
    #G.load_state_dict(torch.load("/home/ubd/EMANet-master/Video_Prediction_ZOO-master/RetrospectiveCycleGAN/saved_models/no_single_net_/G_snigle_net__future_%d.pth" %  opt.epoch, map_location='cpu'))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

count = 0
psnr_data = 0
ssim_data = 0
mse_data = 1
lpips_data = 1
psnr_data = 0
mae_data = 0
msssim_data = 0
tatal_PSNR = 0
total_SSIM = 0
total_MSE = 0
total_MSSSIM = 0
total_LPIPS = 0 
total_MAE = 0
total_DSSIM = 0
ms_ssim_module=MS_SSIM(win_size=7,win_sigma=1.5,data_range=1,size_average=True,channel=3)
loss_fn = lpips.LPIPS(net='alex')

#################loading Depth
parser1 = argparse.ArgumentParser()
parser1.add_argument('--model', default='resnet18_md',help='encoder architecture: ' +'resnet18_md or resnet50_md ' + '(default: resnet18)'+ 'or torchvision version of any resnet model')
parser1.add_argument('--input_channels', default=2,help='Number of channels in input tensor')
parser1.add_argument('--pretrained', default=False,help='Use weights of pretrained model')
args1 = parser1.parse_args([])
depthmodel= monout.get_model(args1.model, input_channels=args1.input_channels, pretrained=args1.pretrained)
depthmodel = depthmodel.cuda()
depthmodel_1= monout.get_model(args1.model, input_channels=3, pretrained=args1.pretrained)
depthmodel_1 = depthmodel_1.cuda()
#################loading Depth
#calculate the mse,psnr ans ssim over the testing data.
for i, frame_seq in enumerate(test_dataloader):

    frame_seq = frame_seq.type(Tensor)
    real_A = Variable(frame_seq[:,-1,...]) #[bs,1,c,h,w]
    input_A = Variable(frame_seq[:,:-1,...].view((frame_seq.size(0),-1)+frame_seq.size()[3:]))
    
    result_fake_depth = depthmodel_1(real_A)
    gen_dep= result_fake_depth[0]
    gen_dep_ex_A = torch.cat([gen_dep, gen_dep[:, :1, :, :]], dim=1)
    
    
    A1 = G(input_A,gen_dep_ex_A)
       
    count += 1
    psnr = metric.PSNR(real_A.squeeze(0).detach().cpu().clone().numpy(),A1.squeeze(0).detach().cpu().clone().numpy())
    import skimage
    ssim = metric.SSIM(real_A.squeeze(0).detach().cpu().clone().numpy(),A1.squeeze(0).detach().cpu().clone().numpy())
    mse = metric.MSE(real_A.squeeze(0).detach().cpu().clone().numpy(),A1.squeeze(0).detach().cpu().clone().numpy())
    ms_ssim_loss=ms_ssim_module(real_A.detach().cpu().clone().numpy(),A1.detach().cpu().clone().numpy())
    mae = metric.MAE(real_A.squeeze(0).detach().cpu().clone().numpy(), A1.squeeze(0).detach().cpu().clone().numpy())                            
    lpips =  loss_fn(real_A.detach().cpu().clone(), A1.detach().cpu().clone())  
    dssim = metric.DSSIM(real_A.squeeze(0).detach().cpu().clone().numpy(),A1.squeeze(0).detach().cpu().clone().numpy())
    
    if psnr > psnr_data:
        psnr_data = psnr
    if ssim > ssim_data:
        ssim_data = ssim
    if ms_ssim_loss > msssim_data:
        msssim_data = ms_ssim_loss
    if mse < mse_data:
        mse_data = mse 
    if lpips < lpips_data:
        lpips_data = lpips 
    tatal_PSNR += psnr
    total_SSIM += ssim
    total_MSE += mse
    total_MSSSIM += ms_ssim_loss
    total_MAE += mae
    total_DSSIM += dssim
    lpips_total = lpips
    total_LPIPS += lpips_total
    print("[{}/{}][currunt psnr: {:.4f} average psnr: {:.4f}] [currunt ssim: {:.4f} average ssim: {:.4f}] \
#[currunt mse: {:.4f} average mse: {:.4f}] [currunt mae: {:.4f} average mae: {:.4f}] [currunt dssim: {:.4f} average dssim: {:.4f}]\
#[currunt LPIPS: {:.4f} average LPIPS: {:.4f}] [currunt MSSSIM: {:.4f} average MSSSIM: {:.4f}]".format(count,len(test_dataloader),psnr,tatal_PSNR/count,ssim,total_SSIM/count,mse,total_MSE/count,mae,total_MAE/count,dssim,total_DSSIM/count,lpips_total.item(),total_LPIPS.item()/count,ms_ssim_loss.item(),total_MSSSIM/count))

   prev = A1.cpu().clone()
   curr = A1[j+1].cpu().clone()
   image_grid_prev = prev
   
   save_image(image_grid_prev, "../RetrospectiveCycleGAN/saved_images/real_cal_1/fake_jpg/fake_%s_prev.png" % (i), normalize=False)
   image_grid_curr = curr
   save_image(image_grid_curr, "../RetrospectiveCycleGAN/saved_images/new_cal/fake_jpg/fake_%s_curr.png" % (j), normalize=False) 
   prev_true = real_A.cpu().clone()
   curr_true = real_A[j+1].cpu().clone()
   image_grid_prev_true = prev_true
   save_image(image_grid_prev_true, "../RetrospectiveCycleGAN/saved_images/real_cal_1/real_jpg/true_%s_prev.png" % (i), normalize=False)
   image_grid_curr_true = curr_true
   save_image(image_grid_curr_true, "../RetrospectiveCycleGAN/saved_images/new_cal/real_jpg/true_%s_curr.png" % (j), normalize=False)  
print("Final SSIM={}, LPIPS={},PSNR={},MSE={},MSSSIM={}".format(ssim_data,lpips_data,psnr_data,mse_data,msssim_data))
