import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class DISTS(nn.Module):
    def __init__(self):
        super(DISTS, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slices = nn.ModuleList()
        # 提取 VGG16 的 5 个阶段特征
        layer_ids = [4, 9, 16, 23, 30]
        prev = 0
        for lid in layer_ids:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, lid)]))
            prev = lid
        for param in self.parameters():
            param.requires_grad = False

    def forward_once(self, x):
        feats = []
        for slice in self.slices:
            x = slice(x)
            feats.append(x)
        return feats

    def forward(self, x, y):
        # x,y: (B,3,H,W)，值范围 [0,1]
        feats_x = self.forward_once(x)
        feats_y = self.forward_once(y)

        loss = 0
        for fx, fy in zip(feats_x, feats_y):
            # L2 差异 + 相关性
            loss += F.mse_loss(fx, fy)
        return loss

    
class LPIPSvgg(nn.Module):
    def __init__(self):
        super(LPIPSvgg, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slices = nn.ModuleList()
        layer_ids = [4, 9, 16, 23, 30]
        prev = 0
        for lid in layer_ids:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, lid)]))
            prev = lid
        for param in self.parameters():
            param.requires_grad = False

    def forward_once(self, x):
        feats = []
        for slice in self.slices:
            x = slice(x)
            feats.append(x)
        return feats

    def forward(self, x, y):
        feats_x = self.forward_once(x)
        feats_y = self.forward_once(y)

        loss = 0
        for fx, fy in zip(feats_x, feats_y):
            # LPIPS 计算范式：逐层 L2 差异
            loss += F.mse_loss(F.normalize(fx, dim=1), F.normalize(fy, dim=1))
        return loss