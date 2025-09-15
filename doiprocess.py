import torch
import numpy as np
from torchvision.utils import save_image

def batch_opt_tensor(A_fake, A_real, flow_model, use_cuda=True):
    """
    Compute optical flow between generated frames and corresponding real frames.
    
    A_fake, A_real: [num_frames, C, H, W] Tensor
    flow_model: optical flow network (expects (img1, img2))
    
    return: optical_fake_list, optical_real_list
            - optical_fake_list[i] is flow from A_fake[i] to A_real[i]
            - optical_real_list[i] is flow from A_real[i] to A_fake[i] (optional)
    """

    if use_cuda:
        A_fake = A_fake.cuda()
        A_real = A_real.cuda()

    num_frames = A_fake.shape[0]

    optical_fake_list, optical_real_list = [], []

    for j in range(int(num_frames-1)):
        # Flow from generated frame to real frame
        flow_fake_input = torch.stack([A_fake[j], A_real[j+1]], dim=2) 
        flow_fake_input = flow_fake_input.permute(0, 2, 1, 3)
        flow_fake_input = flow_fake_input.unsqueeze(0)

        flow_real_input = torch.stack([A_real[j], A_fake[j+1]], dim=2) 
        flow_real_input = flow_real_input.permute(0, 2, 1, 3)
        flow_real_input = flow_real_input.unsqueeze(0)        
        
        flow_fake = flow_model(flow_fake_input)
        optical_fake_list.append(flow_fake)

        # Flow from real frame to generated frame (逆序，可选)
        flow_real = flow_model(flow_real_input)
        optical_real_list.append(flow_real)

    optical_fake_data = torch.cat(optical_fake_list, dim=0)
    optical_real_data = torch.cat(optical_real_list, dim=0)        
        
    return optical_fake_data, optical_real_data


def batch_dep_tensor(A_fake, A_real, depth_model, use_cuda=True):
    """
    A_fake, A_real: [num_frames, C, H, W] Tensor
    depth_model, flow_model: torch model
    return: depth_real_list, depth_fake_list, optical_fake_list, optical_real_list,
            optical_fake_rev_list, optical_real_rev_list
    """

    if use_cuda:
        A_fake = A_fake.cuda()
        A_real = A_real.cuda()

    num_frames = A_fake.shape[0]

    depth_fake_list, depth_real_list = [], []

    for j in range(num_frames):
        # ------------------- Depth -------------------
        depth_fake_prev = depth_model(A_fake[j].unsqueeze(0))
        # depth_fake_curr = depth_model(A_fake[j+1].unsqueeze(0))
        depth_real_prev = depth_model(A_real[j].unsqueeze(0))
        # depth_real_curr = depth_model(A_real[j+1].unsqueeze(0))

        depth_fake_list.extend(depth_fake_prev[0].unsqueeze(0))
        depth_real_list.extend(depth_real_prev[0].unsqueeze(0))
        
    depth_fake_data = torch.cat(depth_fake_list, dim=0)
    depth_real_data = torch.cat(depth_real_list, dim=0)
        
    return depth_real_data, depth_fake_data


def batch_process_tensor_and_compute_loss(A_fake, A_real, depth_model, flow_model, criterion, use_cuda=True):
    """
    A_fake, A_real: [num_frames, C, H, W]  Tensor
    depth_model, flow_model: torch model
    criterion: loss function
    return: depth_loss_0, depth_loss_1, optical_loss_0, optical_loss_1
    """

    if use_cuda:
        A_fake = A_fake.cuda()
        A_real = A_real.cuda()

    num_frames = A_fake.shape[0]

    depth_loss_0, depth_loss_1 = [], []
    optical_loss_0, optical_loss_1 = [], []

    for j in range(num_frames - 1):
        # ------------------- Depth-------------------
        result_fake_prev = depth_model(A_fake[j].unsqueeze(0))
        result_fake_curr = depth_model(A_fake[j+1].unsqueeze(0))
        result_real_prev = depth_model(A_real[j].unsqueeze(0))
        result_real_curr = depth_model(A_real[j+1].unsqueeze(0))

        depth_loss_0.append(criterion(result_fake_prev[0], result_real_prev[0]))
        depth_loss_1.append(criterion(result_fake_curr[0], result_real_curr[0]))

        # ------------------- OPtical flow -------------------
        flow_fake_input = torch.stack([A_fake[j], A_fake[j+1]], dim=2) 
        flow_fake_input = flow_fake_input.permute(0, 2, 1, 3)
        flow_fake_input = flow_fake_input.unsqueeze(0)
        
        flow_real_input = torch.stack([A_real[j], A_real[j+1]], dim=2) 
        flow_real_input = flow_real_input.permute(0, 2, 1, 3)
        flow_real_input = flow_real_input.unsqueeze(0)
        
        flow_fake = flow_model(flow_fake_input)  # [1,2,H,W]
        flow_real = flow_model(flow_real_input)  # [1,2,H,W]

        optical_loss_0.append(criterion(flow_fake[:,0], flow_real[:,0]))  # x 分量
        optical_loss_1.append(criterion(flow_fake[:,1], flow_real[:,1]))  # y 分量

    return depth_loss_0, depth_loss_1, optical_loss_0, optical_loss_1
