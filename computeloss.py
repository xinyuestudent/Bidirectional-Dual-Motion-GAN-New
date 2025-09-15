import PIL
PIL.PILLOW_VERSION = PIL.__version__
import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid

def train_discriminator_B(flownet, depthnet,D_B,D_C, criterion_GAN, 
                          frame_seq, input_B1_A, input_B11_A1, 
                          input_B_A1, input_B1_A11, valid, fake, valid_s, fake_s, read_gen, device="cuda"):
    total_loss = 0.0
    loss_real = criterion_GAN(D_C(Variable(frame_seq.view((frame_seq.size(0),-1)+frame_seq.size()[3:])))[0], valid)
    loss_fake_B1_A = criterion_GAN(D_C(input_B1_A.detach())[0],fake)
    loss_fake_B11_A1 = criterion_GAN(D_C(input_B11_A1.detach())[0],fake)
    loss_fake_B_A1 = criterion_GAN(D_C(input_B_A1.detach())[0],fake)
    loss_fake_B1_A11 = criterion_GAN(D_C(input_B1_A11.detach())[0],fake)
    for channel in ["A", "B"]:

        for j in range(len(frame_seq)):
            # 读取 fake
            pim1_fake = read_gen(f"../RetrospectiveCycleGAN/fake_jpg_{channel}1/fake_{j}_prev.png")
            pim2_fake = read_gen(f"../RetrospectiveCycleGAN/fake_jpg_{channel}1/fake_{j}_curr.png")
            images_fake = np.array([pim1_fake, pim2_fake]).transpose(3,0,1,2)
            im_fake = torch.from_numpy(images_fake.astype(np.float32)).unsqueeze(0).to(device)
            im_fake_dep = torch.from_numpy(images_fake.astype(np.float32)).to(device)
            # 生成光流和 depth
            result_fake_flow = flownet(im_fake).squeeze()
            result_fake_flow_ex = result_fake_flow.unsqueeze(1).repeat(1, 3, 1, 1)  
            result_fake_depth = depthnet(im_fake_dep)
            if isinstance(result_fake_depth, tuple):
                result_fake_depth = result_fake_depth[0]
            result_fake_depth = result_fake_depth.squeeze()

            # 读取 real
            pim1_real = read_gen(f"../RetrospectiveCycleGAN/true_jpg_{channel}/true_{j}_prev.png")
            pim2_real = read_gen(f"../RetrospectiveCycleGAN/true_jpg_{channel}/true_{j}_curr.png")
            images_real = np.array([pim1_real, pim2_real]).transpose(3,0,1,2)
            im_real = torch.from_numpy(images_real.astype(np.float32)).unsqueeze(0).to(device) #([ 3, 2, 128, 128])
            im_real_dep = torch.from_numpy(images_real.astype(np.float32)).to(device)

            result_real_flow = flownet(im_real).squeeze()# ([2, 128, 128])
            result_real_flow_ex = result_real_flow.unsqueeze(1).repeat(1, 3, 1, 1) 
            result_real_depth = depthnet(im_real_dep)
            if isinstance(result_real_depth, tuple):
                result_real_depth = result_real_depth[0]
            result_real_depth = result_real_depth.squeeze()#([3, 2, 128, 128])
            im_real_se = im_real.squeeze()
            im_fake_se = im_fake.squeeze()

            # ---- GAN Loss ----
            pred_image, pred_flow, pred_depth = D_B(im_real_se.permute(1,0,2,3), result_real_flow_ex, result_real_depth.permute(1,0,2,3))
            pred_image_fake, pred_flow_fake, pred_depth_fake = D_B(im_fake_se.permute(1,0,2,3), result_fake_flow_ex, result_fake_depth.permute(1,0,2,3))

            gan_loss = (criterion_GAN(pred_image, valid_s) + criterion_GAN(pred_image_fake.detach(), fake_s) +
                        criterion_GAN(pred_flow, valid_s)  + criterion_GAN(pred_flow_fake.detach(), fake_s) +
                        criterion_GAN(pred_depth, valid_s) + criterion_GAN(pred_depth_fake.detach(), fake_s)) / 6.0

            # 累加总 loss

            total_loss += gan_loss

    # 对两通道求平均
    final_loss = (total_loss / 2.0 + loss_real + loss_fake_B1_A + loss_fake_B11_A1 + loss_fake_B_A1 + loss_fake_B1_A11)/6
    return final_loss

def compute_depth_optical_losses(A1, real_A,B1, real_B,
    flownet, depthnet,
    criterion_Limage, criterion_Ldepth,
    read_gen, sequence_len , device="cuda"
):
    """
    计算 A 和 B 通道的 Optical Flow Loss + Depth Loss

    Args:
        flownet: 光流网络
        depthnet: 深度估计网络
        criterion_Limage: 图像损失函数 (比如 L1)
        criterion_Ldepth: 深度损失函数 (比如 L1/MSE)
        read_gen: 图像读取函数，输入 path -> numpy(H, W, C)
        device: "cuda" or "cpu"
    
    Returns:
        depth_loss_total_A, optical_loss_total_A,
        depth_loss_total_B, optical_loss_total_B
    """
 
    depth_loss   = {}
    optical_loss_0_A, optical_loss_1_A = [], []
    depth_loss_A = []

    optical_loss_0_B, optical_loss_1_B = [], []
    depth_loss_B = []
    length = sequence_len
    # A 通道
    for j in range(length - 1):
        
        prev = A1[j].cpu().clone()
        curr = A1[j+1].cpu().clone()
        image_grid_prev = prev
        save_image(image_grid_prev, "../RetrospectiveCycleGAN/fake_jpg_A1/fake_%s_prev.png" % (j), normalize=False)
        image_grid_curr = curr
        save_image(image_grid_curr, "../RetrospectiveCycleGAN/fake_jpg_A1//fake_%s_curr.png" % (j), normalize=False)  
        prev_true = real_A[j].cpu().clone()
        curr_true = real_A[j+1].cpu().clone()
        image_grid_prev_true = prev_true
        save_image(image_grid_prev_true, "../RetrospectiveCycleGAN/true_jpg_A/true_%s_prev.png" % (j), normalize=False)
        image_grid_curr_true = curr_true
        save_image(image_grid_curr_true, "../RetrospectiveCycleGAN/true_jpg_A/true_%s_curr.png" % (j), normalize=False) 
        # --- fake A ---
        pim1_fake = read_gen(f"../RetrospectiveCycleGAN/fake_jpg_A1/fake_{j}_prev.png")
        pim2_fake = read_gen(f"../RetrospectiveCycleGAN/fake_jpg_A1/fake_{j}_curr.png")
        images_fake = np.array([pim1_fake, pim2_fake]).transpose(3, 0, 1, 2)
        im_fake = torch.from_numpy(images_fake.astype(np.float32)).unsqueeze(0).to(device)
        im_fake_dep = torch.from_numpy(images_fake.astype(np.float32)).to(device)
        result_fake_flow = flownet(im_fake).squeeze()
        result_fake_depth = depthnet(im_fake_dep)

        # --- real A ---
        pim1_real = read_gen(f"../RetrospectiveCycleGAN/true_jpg_A/true_{j}_prev.png")
        pim2_real = read_gen(f"../RetrospectiveCycleGAN/true_jpg_A/true_{j}_curr.png")
        images_real = np.array([pim1_real, pim2_real]).transpose(3, 0, 1, 2)
        im_real = torch.from_numpy(images_real.astype(np.float32)).unsqueeze(0).to(device)
        im_real_dep = torch.from_numpy(images_real.astype(np.float32)).to(device)
        result_real_flow = flownet(im_real).squeeze()
        result_real_depth = depthnet(im_real_dep)

        # 光流损失
        loss_opt_0 = criterion_Limage(result_fake_flow[0], result_real_flow[0])
        loss_opt_1 = criterion_Limage(result_fake_flow[1], result_real_flow[1])
        optical_loss_0_A.append(loss_opt_0)
        optical_loss_1_A.append(loss_opt_1)

        # 深度损失
        loss_depth = criterion_Ldepth(result_fake_depth[0], result_real_depth[0])
        depth_loss_A.append(loss_depth)

    optical_loss_total_A = (sum(optical_loss_0_A) / len(optical_loss_0_A) +
                            sum(optical_loss_1_A) / len(optical_loss_1_A)) / 2
    depth_loss_total_A = sum(depth_loss_A) / len(depth_loss_A)

    # B 通道
    for j in range(length - 1):
        prev = B1[j].cpu().clone()
        curr = B1[j+1].cpu().clone()
        image_grid_prev = prev
        save_image(image_grid_prev, "../RetrospectiveCycleGAN/fake_jpg_B1/fake_%s_prev.png" % (j), normalize=False)
        image_grid_curr = curr
        save_image(image_grid_curr, "../RetrospectiveCycleGAN/fake_jpg_B1/fake_%s_curr.png" % (j), normalize=False)  
        prev_true = real_B[j].cpu().clone()
        curr_true = real_B[j+1].cpu().clone()
        image_grid_prev_true = prev_true
        save_image(image_grid_prev_true, "../RetrospectiveCycleGAN/true_jpg_B/true_%s_prev.png" % (j), normalize=False)
        image_grid_curr_true = curr_true
        save_image(image_grid_curr_true, "../RetrospectiveCycleGAN/true_jpg_B/true_%s_curr.png" % (j), normalize=False) 
        # --- fake B ---
        pim1_fake = read_gen(f"../RetrospectiveCycleGAN/fake_jpg_B1/fake_{j}_prev.png")
        pim2_fake = read_gen(f"../RetrospectiveCycleGAN/fake_jpg_B1/fake_{j}_curr.png")
        images_fake = np.array([pim1_fake, pim2_fake]).transpose(3, 0, 1, 2)
        im_fake = torch.from_numpy(images_fake.astype(np.float32)).unsqueeze(0).to(device)
        im_fake_dep = torch.from_numpy(images_fake.astype(np.float32)).to(device)
        result_fake_flow = flownet(im_fake).squeeze()
        result_fake_depth = depthnet(im_fake_dep)

        # --- real B ---
        pim1_real = read_gen(f"../RetrospectiveCycleGAN/true_jpg_B/true_{j}_prev.png")
        pim2_real = read_gen(f"../RetrospectiveCycleGAN/true_jpg_B/true_{j}_curr.png")
        images_real = np.array([pim1_real, pim2_real]).transpose(3, 0, 1, 2)
        im_real = torch.from_numpy(images_real.astype(np.float32)).unsqueeze(0).to(device)
        im_real_dep = torch.from_numpy(images_real.astype(np.float32)).to(device)
        result_real_flow = flownet(im_real).squeeze()
        result_real_depth = depthnet(im_real_dep)

        # 光流损失
        loss_opt_0 = criterion_Limage(result_fake_flow[0], result_real_flow[0])
        loss_opt_1 = criterion_Limage(result_fake_flow[1], result_real_flow[1])
        optical_loss_0_B.append(loss_opt_0)
        optical_loss_1_B.append(loss_opt_1)

        # 深度损失
        loss_depth = criterion_Ldepth(result_fake_depth[0], result_real_depth[0])
        depth_loss_B.append(loss_depth)

    optical_loss_total_B = (sum(optical_loss_0_B) / len(optical_loss_0_B) +
                            sum(optical_loss_1_B) / len(optical_loss_1_B)) / 2
    depth_loss_total_B = sum(depth_loss_B) / len(depth_loss_B)

    return depth_loss_total_A, optical_loss_total_A, depth_loss_total_B, optical_loss_total_B

def compute_final_loss(
    flownet, depthnet,
    D_A, criterion_Limage, criterion_Ldepth, criterion_GAN,
    read_gen,
    valid, fake,
    sequence_len,
    device="cuda",
    weight_l1=1.0, weight_depth=1.0, weight_gan=1.0
):
    """
    返回最终综合 loss (只一个标量)
    """

    total_loss = 0.0

    for channel in ["A", "B"]:

        for j in range(sequence_len):
            # 读取 fake
            pim1_fake = read_gen(f"../RetrospectiveCycleGAN/fake_jpg_{channel}1/fake_{j}_prev.png")
            pim2_fake = read_gen(f"../RetrospectiveCycleGAN/fake_jpg_{channel}1/fake_{j}_curr.png")
            images_fake = np.array([pim1_fake, pim2_fake]).transpose(3,0,1,2)
            im_fake = torch.from_numpy(images_fake.astype(np.float32)).unsqueeze(0).to(device)
            im_fake_dep = torch.from_numpy(images_fake.astype(np.float32)).to(device)
            # 生成光流和 depth
            result_fake_flow = flownet(im_fake).squeeze()
            result_fake_flow_ex = result_fake_flow.unsqueeze(1).repeat(1, 3, 1, 1)  
            result_fake_depth = depthnet(im_fake_dep)
            if isinstance(result_fake_depth, tuple):
                result_fake_depth = result_fake_depth[0]
            result_fake_depth = result_fake_depth.squeeze()

            # 读取 real
            pim1_real = read_gen(f"../RetrospectiveCycleGAN/true_jpg_{channel}/true_{j}_prev.png")
            pim2_real = read_gen(f"../RetrospectiveCycleGAN/true_jpg_{channel}/true_{j}_curr.png")
            images_real = np.array([pim1_real, pim2_real]).transpose(3,0,1,2)
            im_real = torch.from_numpy(images_real.astype(np.float32)).unsqueeze(0).to(device) #([ 3, 2, 128, 128])
            im_real_dep = torch.from_numpy(images_real.astype(np.float32)).to(device)

            result_real_flow = flownet(im_real).squeeze()# ([2, 128, 128])
            result_real_flow_ex = result_real_flow.unsqueeze(1).repeat(1, 3, 1, 1) 
            result_real_depth = depthnet(im_real_dep)
            if isinstance(result_real_depth, tuple):
                result_real_depth = result_real_depth[0]
            result_real_depth = result_real_depth.squeeze()#([3, 2, 128, 128])
            im_real_se = im_real.squeeze()
            im_fake_se = im_fake.squeeze()

            # ---- GAN Loss ----
            pred_image, pred_flow, pred_depth = D_A(im_real_se.permute(1,0,2,3), result_real_flow_ex, result_real_depth.permute(1,0,2,3))
            pred_image_fake, pred_flow_fake, pred_depth_fake = D_A(im_fake_se.permute(1,0,2,3), result_fake_flow_ex, result_fake_depth.permute(1,0,2,3))

            gan_loss = (criterion_GAN(pred_image, valid) + criterion_GAN(pred_image_fake.detach(), fake) +
                        criterion_GAN(pred_flow, valid)  + criterion_GAN(pred_flow_fake.detach(), fake) +
                        criterion_GAN(pred_depth, valid) + criterion_GAN(pred_depth_fake.detach(), fake)) / 6.0

            # 累加总 loss

            total_loss += gan_loss

    # 对两通道求平均
    final_loss = total_loss / 2.0
    return final_loss