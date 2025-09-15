'''
@xinyuewang
The implementation code of the paper "Bidirectional Dual Motion GAN for Video Frame Prediction in Intelligent Driving" with Pytorch
data: Sep,12,2025
'''
import sys
sys.path.append('./Mono/')
import argparse
import os
import PIL
PIL.PILLOW_VERSION = PIL.__version__
import numpy as np
import math
import itertools
import datetime
import time
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import Mono.utilsmono as monout
from Mono.main_monodepth_pytorch import Model as MonoDepthModel
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
from laplacian_of_guassian import *
from flownet2.models import FlowNet2C  # the path is depended on where you create this module
from flownet2.utils.frame_utils import read_gen  # the path is depended on where you create this module
import time
from tqdm import tqdm
import scipy
import LPIPDISTS as LD
import matplotlib.pyplot as plt
from  computeloss import compute_depth_optical_losses as cal_loss
from  computeloss import compute_final_loss
from  computeloss import train_discriminator_B

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--train_data", type=str, default="../RetrospectiveCycleGAN/dataset/pickle_data/real_train_cal_data.pkl", help="the path of pickle file about train data")
parser.add_argument("--val_data", type=str, default="../RetrospectiveCycleGAN/dataset/pickle_data/real_val_cal_data.pkl", help="the path of pickle file about validation data")
parser.add_argument("--test_data", type=str, default="../RetrospectiveCycleGAN/dataset/pickle_data/real_test_cal_data.pkl", help="the path of pickle file about testing data")
parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=0, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height") #当图片的宽和高设置为256的时候会导致内存溢出
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=2000, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_LoG", type=float, default=0.005, help="cycle loss weight")
parser.add_argument("--lambda_frame_GAN", type=float, default=0.003, help="identity loss weight")
parser.add_argument("--lambda_seq_GAN", type=float, default=0.003, help="identity loss weight")
parser.add_argument("--sequence_len", type=float, default=5, help="the original length of frame sequence(n+1)")
parser.add_argument("--save_model_path", type=str, default="saved_models/real_cal_0/", help="the path of saving the models")
parser.add_argument("--save_image_path", type=str, default="saved_images/real_cal_0/", help="the path of saving the images")
parser.add_argument("--log_file", type=str, default="log_real_cal_0.txt", help="the logging info of training")


opt = parser.parse_args()##返回一个命名空间,如果想要使用变量,可用args.attr
print(opt)

# Create sample and checkpoint directories
os.makedirs(opt.save_image_path, exist_ok=True)
os.makedirs(opt.save_model_path, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_Limage = torch.nn.L1Loss()
cuda = torch.cuda.is_available()

# Image transformations
transforms_ = [
    transforms.Resize((int(opt.img_height),int(opt.img_width)), Image.BICUBIC),
    transforms.ToTensor(),#Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
                          #Converts a PIL Image or numpy.ndarray (H x W x C) in the range
                          #[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),#[0,1] -> [-1,1]
#     transforms.Normalize([0.485, ], [0.229, ])
]

# Training data loader
dataloader = DataLoader(
    ImageTrainDataset(opt.train_data, transforms_=transforms_,nt=opt.sequence_len),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
    drop_last=True
)

# val data loader
val_dataloader = DataLoader(
      ImageValDataset(opt.val_data, transforms_=transforms_,nt=opt.sequence_len),
      batch_size=1,
      shuffle=False,
      num_workers=opt.n_cpu,
      drop_last=True
  )

# test data loader
test_dataloader = DataLoader(
    ImageTestDataset(opt.test_data, transforms_=transforms_,nt=opt.sequence_len),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
    drop_last=True
)


input_shape = (opt.batch_size, opt.sequence_len, opt.channels, opt.img_height, opt.img_width)
G_future = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_past = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = TriDiscriminatorA(input_shape)
D_B = TriDiscriminatorB(input_shape)
D_C = DiscriminatorB(input_shape)
#loss_fn = lpips.LPIPS(net='alex')
Laplacian = Laplacian()
dists = LD.DISTS()
lpips_vgg = LD.LPIPSvgg()

sequence_len = opt.sequence_len

if cuda:
    G_future = torch.nn.DataParallel(G_future, device_ids=range(torch.cuda.device_count()))
    G_future = G_future.cuda()
    G_past = torch.nn.DataParallel(G_past, device_ids=range(torch.cuda.device_count()))
    G_past = G_past.cuda()
    D_A = torch.nn.DataParallel(D_A, device_ids=range(torch.cuda.device_count()))
    D_A = D_A.cuda()
    D_B = torch.nn.DataParallel(D_B, device_ids=range(torch.cuda.device_count()))
    D_B = D_B.cuda()
    D_C = torch.nn.DataParallel(D_C, device_ids=range(torch.cuda.device_count()))
    D_C = D_C.cuda()
    criterion_GAN.cuda()
    criterion_Limage.cuda()
    Laplacian = Laplacian.cuda()
    #loss_fn = loss_fn.cuda()
if opt.epoch != 0:
    # Load pretrained models
    G_future.load_state_dict(torch.load("../RetrospectiveCycleGAN/saved_models/cal_dual_Ora_tri/G_future_%d.pth" %  (opt.epoch)))
    G_past.load_state_dict(torch.load("../RetrospectiveCycleGAN/saved_models/cal_dual_Ora_tri/G_past_%d.pth" %  (opt.epoch)))
    D_A.load_state_dict(torch.load("../RetrospectiveCycleGAN/saved_models/cal_dual_Ora_tri/D_A_%d.pth" %  (opt.epoch)))
    D_B.load_state_dict(torch.load("../RetrospectiveCycleGAN/saved_models/cal_dual_Ora_tri/D_B_%d.pth" %  (opt.epoch)))
    D_C.load_state_dict(torch.load("../RetrospectiveCycleGAN/saved_models/cal_dual_Ora_tri/D_F_%d.pth" %  (opt.epoch)))
else:
    # Initialize weights
    G_future.apply(weights_init_normal)
    G_past.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)
    D_C.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_future.parameters(),G_past.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(itertools.chain(D_B.parameters(),D_C.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step# lr_lambda为操作学习率的函数
)

lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

#保存中间的训练结果
def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    imgs = imgs.type(Tensor)
    input_A = imgs[:,:-1,...]
    input_A = input_A.view((imgs.size(0),-1,)+imgs.size()[3:])
    G_future.eval()
    real_A = Variable(imgs[:,-1:,...])
    A1 = G_future(input_A)
    frames = torch.cat((imgs[0,:-1,],A1[0].unsqueeze(0),imgs[0]), 0)
    image_grid = make_grid(frames,nrow=opt.sequence_len,normalize=False)
    save_image(image_grid, opt.save_image_path+"fake_%s.png" % (batches_done), normalize=False)

def ReverseSeq(Seq):
    length = Seq.size(1)
    return torch.cat([Seq[:,i-2:i+1,...] for i in range(length-1,-1,-3)],1)

count = 0
#################loading Flownet2.0
parser0 = argparse.ArgumentParser()
parser0.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser0.add_argument("--rgb_max", type=float, default=255.)
args0 = parser0.parse_args([])
flownet = FlowNet2C(args0).cuda()
    # load the state_dict
dict = torch.load("./flownet2/FlowNet2-C_checkpoint.pth.tar")
flownet.load_state_dict(dict["state_dict"])
loss_fn = lpips.LPIPS(net='alex')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#################loading Flownet2.0
#################loadingDepth
parser1 = argparse.ArgumentParser()
parser1.add_argument('--model', default='resnet18_md',help='encoder architecture: ' +'resnet18_md or resnet50_md ' + '(default: resnet18)'+ 'or torchvision version of any resnet model')
parser1.add_argument('--input_channels', default=2,help='Number of channels in input tensor')
parser1.add_argument('--pretrained', default=False,help='Use weights of pretrained model')
args1 = parser1.parse_args([])
depthmodel= monout.get_model(args1.model, input_channels=args1.input_channels, pretrained=args1.pretrained)
depthmodel = depthmodel.cuda()
depthmodel_1= monout.get_model(args1.model, input_channels=3, pretrained=args1.pretrained)
depthmodel_1 = depthmodel_1.cuda()
#################loadingDepth
prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, frame_seq in tqdm(enumerate(dataloader)):

        count = count + 1
        frame_seq = frame_seq.type(Tensor)

        real_A = Variable(frame_seq[:,-1,...]) #[bs,1,c,h,w]
        input_A = Variable(frame_seq[:,:-1,...].view((frame_seq.size(0),-1)+frame_seq.size()[3:]))
        real_B = Variable(frame_seq[:,0,...]) #[bs,1,c,h,w]
        input_B_ = Variable(frame_seq[:,1:,...].view((frame_seq.size(0),-1)+frame_seq.size()[3:]))
        input_B = ReverseSeq(input_B_)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((frame_seq.size(0), *D_B.module.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((frame_seq.size(0), *D_B.module.output_shape))), requires_grad=False)
        valid_tri = Variable(Tensor(np.ones((2, *D_B.module.output_shape))), requires_grad=False)
        fake_tri = Variable(Tensor(np.zeros((2, *D_B.module.output_shape))), requires_grad=False)

        #------------------------
        #  Train Generator
        #------------------------
        G_future.train()
        G_past.train()        
        optimizer_G.zero_grad()
        
        result_fake_depth = depthmodel_1(real_A)
        gen_dep= result_fake_depth[0]
        gen_dep_ex_A = torch.cat([gen_dep, gen_dep[:, :1, :, :]], dim=1)
        
        result_fake_depth = depthmodel_1(real_B)
        gen_dep= result_fake_depth[0]
        gen_dep_ex_B = torch.cat([gen_dep, gen_dep[:, :1, :, :]], dim=1)

        #L_Image loss which minimize the L1 Distance between the image pair
        A1 = G_future(input_A,gen_dep_ex_A) # x^'_{n}  generated future frame
        B1 = G_past(input_B,gen_dep_ex_B) # x^'_{m} generated past frame
                       
                
        depth_loss_A, opt_loss_A, depth_loss_B, opt_loss_B = cal_loss(A1,real_A,B1,real_B,flownet, depthmodel,criterion_Limage, criterion_Limage,read_gen, sequence_len )

        input_A_A1_ = torch.cat((input_A[:,3:,...],A1),1)
        input_A_A1 = ReverseSeq(input_A_A1_)
        input_B_B1 = torch.cat((B1,input_A[:,3:,...]),1)
        A11 = G_future(input_B_B1,gen_dep_ex_A)# x^''_{n}
        B11 = G_past(input_A_A1,gen_dep_ex_B)

        loss_A_A1 = criterion_Limage(real_A,A1)
        loss_A_A11 = criterion_Limage(real_A,A11)
        loss_A1_A11 = criterion_Limage(A1,A11)
        loss_B_B1 = criterion_Limage(real_B,B1)
        loss_B_B11 = criterion_Limage(real_B,B11)
        loss_B1_B11 = criterion_Limage(B1,B11)
        
        loss_Image = (loss_A_A1 + loss_A_A11 + loss_A1_A11 + loss_B_B1 + loss_B_B11 + loss_B1_B11 ) / 6
        
        
        #L_LoG loss 
        L_LoG_A_A1 = criterion_Limage(Laplacian(real_A),Laplacian(A1))
        L_LoG_A_A11 = criterion_Limage(Laplacian(real_A),Laplacian(A11))
        L_LoG_A1_A11 = criterion_Limage(Laplacian(A1),Laplacian(A11))
        L_LoG_B_B1 = criterion_Limage(Laplacian(real_B),Laplacian(B1))
        L_LoG_B_B11 = criterion_Limage(Laplacian(real_B),Laplacian(B11))
        L_LoG_B1_B11 = criterion_Limage(Laplacian(B1),Laplacian(B11))

        loss_LoG = (L_LoG_A_A1 + L_LoG_A_A11 + L_LoG_A1_A11 + L_LoG_B_B1 + L_LoG_B_B11 + L_LoG_B1_B11) / 6
#         loss_LoG.backward()
        #GAN frame Loss(Least Square Loss)
        
        loss_frame_GAN_A1  = criterion_GAN(D_A(A1)[0],valid)# lead the synthetic frame become similiar to the real frame
        loss_frame_GAN_B1  = criterion_GAN(D_A(B1)[0],valid)
        loss_frame_GAN_A11 = criterion_GAN(D_A(A11)[0],valid)
        loss_frame_GAN_B11 = criterion_GAN(D_A(B11)[0],valid)
        #Total frame loss
        loss_frame_GAN = (loss_frame_GAN_A1 + loss_frame_GAN_B1 + loss_frame_GAN_A11 + loss_frame_GAN_B11) / 4
        #GAN seq Loss 
        #four kinds of the synthetic frame sequence
        input_B1_A  = torch.cat((B1,input_A[:,3:,...],real_A),1)
        input_B11_A1 = torch.cat((B11,input_A[:,3:,...],A1),1)
        input_B_A1 = torch.cat((real_B,input_A[:,3:,...],A1),1)
        input_B1_A11 = torch.cat((B1,input_A[:,3:,...],A11),1)
        loss_seq_GAN_B1_A = criterion_GAN(D_C(input_B1_A)[0],valid)
        loss_seq_GAN_B11_A1 = criterion_GAN(D_C(input_B11_A1)[0],valid)
        loss_seq_GAN_B_A1 = criterion_GAN(D_C(input_B_A1)[0],valid)
        loss_seq_GAN_B1_A11 = criterion_GAN(D_C(input_B1_A11)[0],valid)
        # Total seq loss
        loss_seq_GAN = (loss_seq_GAN_B1_A + loss_seq_GAN_B11_A1 + loss_seq_GAN_B_A1 + loss_seq_GAN_B1_A11) / 4
        # Total GAN loss
        total_loss_GAN = loss_Image + opt.lambda_frame_GAN * depth_loss_A + opt_loss_A + opt.lambda_frame_GAN *depth_loss_B + opt_loss_B + opt.lambda_LoG*loss_LoG + opt.lambda_frame_GAN *loss_frame_GAN + opt.lambda_seq_GAN*loss_seq_GAN
        total_loss_GAN.backward() #
        optimizer_G.step() # 

        #------------------------
        #  Train Discriminator A
        #------------------------
        optimizer_D_A.zero_grad()
        # Real loss
        final_loss = compute_final_loss(flownet, depthmodel,D_A, criterion_Limage, criterion_Limage, criterion_GAN,read_gen,valid_tri, fake_tri,sequence_len)
        
        loss_fake_A11 = criterion_GAN(D_A(A11.detach())[0], fake)
        loss_fake_B11 = criterion_GAN(D_A(B11.detach())[0], fake)
        # Total loss
        loss_D_A = (final_loss + loss_fake_A11  + loss_fake_B11 ) / 3
        loss_D_A.backward()#retain_graph=True 
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()
        #real loss
        loss_D_B = train_discriminator_B(flownet, depthmodel,D_B,D_C, criterion_GAN, frame_seq, input_B1_A, input_B11_A1, input_B_A1, input_B1_A11, valid, fake,valid_tri, fake_tri, read_gen)
        loss_D_B.backward()
        optimizer_D_B.step()
        total_loss_D = (loss_D_A + loss_D_B)/2
        # --------------
        #  Log Progress
        # --------------
        batches_done = epoch*len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if count % 100 == 0:
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, img: %f, LoG: %f, adv_frame: %f, adv_seq: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    total_loss_D.item(),
                    total_loss_GAN.item(),
                    loss_Image.item(),
                    loss_LoG.item(), 
                    loss_frame_GAN.item(),
                    loss_seq_GAN.item(),
                    time_left,
                )
            )
        if count > opt.sample_interval/opt.batch_size:
            #sample_images(batches_done)
            count = 0
        time.sleep(0.5)
    # Update learning rates linear decay each 100 epochs
    if epoch % 100 == 0:
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    if epoch % 1 == 0:
        num = 0
        tatal_PSNR = 0
        total_SSIM = 0
        total_MSE = 0
        total_MSSSIM = 0
        total_LPIPS = 0 
        total_DISTS = 0 
        total_LPIPSvgg = 0 
        ms_ssim_module=MS_SSIM(win_size=7,win_sigma=1.5,data_range=1,size_average=True,channel=3)
        print('Valing')
        for i, frame_seq_test in tqdm(enumerate(val_dataloader)):

            frame_seq_test = frame_seq_test.type(Tensor)
            real_A_test = Variable(frame_seq_test[:,-1,...]) #[bs,1,c,h,w]         
            input_A = Variable(frame_seq_test[:,:-1,...].view((frame_seq_test.size(0),-1)+frame_seq_test.size()[3:])) 
            
            result_fake_depth = depthmodel_1(real_A_test)
            gen_dep= result_fake_depth[0]
            gen_dep_ex_A = torch.cat([gen_dep, gen_dep[:, :1, :, :]], dim=1)
            
            A1 = G_future(input_A,gen_dep_ex_A)
            
            num += 1
            psnr = metric.PSNR(real_A_test.squeeze(0).detach().cpu().clone().numpy(), A1.squeeze(0).detach().cpu().clone().numpy())
            ssim = metric.SSIM(real_A_test.squeeze(0).detach().cpu().clone().numpy(), A1.squeeze(0).detach().cpu().clone().numpy())
            mse = metric.MSE(real_A_test.squeeze(0).detach().cpu().clone().numpy(), A1.squeeze(0).detach().cpu().clone().numpy())*1000     
            ms_ssim_loss=ms_ssim_module(real_A_test.detach().cpu().clone().numpy(),A1.detach().detach().cpu().clone().numpy())
                               
            lpips =  loss_fn(real_A_test.detach().cpu().clone(), A1.detach().cpu().clone())    
            lv_score = lpips_vgg(real_A_test.detach().cpu().clone(), A1.detach().cpu().clone())    
            
            ds_score = dists(real_A_test.detach().cpu().clone(), A1.detach().cpu().clone())   
            
            tatal_PSNR += psnr
            total_SSIM += ssim
            total_MSE += mse
            total_MSSSIM += ms_ssim_loss
            total_LPIPS += lpips
            total_LPIPSvgg +=lv_score
            total_DISTS += ds_score
            
            time.sleep(0.5)    
        testinfo = "Epoch: {} PSNR={}, SSIM={}, MSE={}, MSSSIM={}, LPIPS={}, DISTS={}, LPIPSvgg={} \n".format(epoch,tatal_PSNR/num,total_SSIM/num,total_MSE/num,total_MSSSIM/num,total_LPIPS/num,total_LPIPSvgg/num,total_DISTS/num)
        with open(opt.log_file, 'a+') as f:
              f.write(testinfo)

    #save the model
    if opt.checkpoint_interval != -1 and epoch % 1 == 0:
        # Save model checkpoints
        torch.save(G_future.state_dict(),   opt.save_model_path+"G_future_%d.pth" % (epoch+1))
        torch.save(G_past.state_dict(),   opt.save_model_path+"G_past_%d.pth" % (epoch+1))
        torch.save(D_A.state_dict(), opt.save_model_path+"D_A_%d.pth" % (epoch+1))
        torch.save(D_B.state_dict(), opt.save_model_path+"D_B_%d.pth" % (epoch+1))
        torch.save(D_C.state_dict(), opt.save_model_path+"D_C_%d.pth" % (epoch+1))
