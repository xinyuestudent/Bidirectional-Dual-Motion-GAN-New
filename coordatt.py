import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class h_swish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32, fusion="avg"):
        super(CoordAtt, self).__init__()
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # 融合方式: avg / sum / cat
        self.fusion = fusion  

    def coord_att(self, x):
        n, c, h, w = x.size()

        x_h = F.adaptive_avg_pool2d(x, (h, 1))  # (n,c,h,1)
        x_w = F.adaptive_avg_pool2d(x, (1, w))  # (n,c,1,w)
        x_w = x_w.permute(0, 1, 3, 2)           # (n,c,w,1)

        y = torch.cat([x_h, x_w], dim=2)        # (n,c,h+w,1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)           # (n,mip,1,w)

        a_h = self.conv_h(x_h).sigmoid()        # (n,oup,h,1)
        a_w = self.conv_w(x_w).sigmoid()        # (n,oup,1,w)

        return a_h, a_w

    def forward(self, x, y):
        a_hx, a_wx = self.coord_att(x)
        a_hy, a_wy = self.coord_att(y)

        # 融合权重
        if self.fusion == "avg":
            a_h = (a_hx + a_hy*0.0001) / 2
            a_w = (a_wx + a_wy*0.0001) / 2
        elif self.fusion == "sum":
            a_h = a_hx + a_hy*0.0001
            a_w = a_wx + a_wy*0.0001
        elif self.fusion == "cat":  
            # 级联再压缩到原维度
            a_h = torch.cat([a_hx, a_hy*0.0001], dim=1)
            a_h = nn.Conv2d(a_h.size(1), x.size(1), kernel_size=1)(a_h).sigmoid()
            a_w = torch.cat([a_wx, a_wy*0.0001], dim=1)
            a_w = nn.Conv2d(a_w.size(1), x.size(1), kernel_size=1)(a_w).sigmoid()
        else:
            raise ValueError("fusion must be one of [avg, sum, cat]")

        # 输出 (融合输入 + 融合注意力)
        identity = (x + y*0.0001) / 2
        out = identity * a_h * a_w
        return out
    
    
# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))

#         mip = max(8, inp // reduction)

#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
        
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

#     def forward(self, x,y):
#         identity = x
        
#         n,c,h,w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)

#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y) 
        
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)

#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()

#         out = identity * a_w * a_h

#         return out
