import torch.nn as nn
import torch.nn.functional as F
import torch
import coordatt as crt
from models import *

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

##############################
#           RESNET
##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        # print(self.block)
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()
        channels = (input_shape[1]-1)*input_shape[2]
        channels_dep = input_shape[2]
        # Initial convolution block       
        out_features = 128
        layer = [
            nn.ReflectionPad2d(3),# (3,3,3,3)
            nn.Conv2d(channels, out_features, 7),
            # nn.InstanceNorm2d(out_features),
            # nn.ReLU(inplace=True),
        ]
        
        layer_1 = [
            nn.ReflectionPad2d(3),# (3,3,3,3)
            nn.Conv2d(channels_dep, out_features, 7),
            # nn.InstanceNorm2d(out_features),
            # nn.ReLU(inplace=True),
        ]
        
        in_features = out_features
        out_features //= 2
        # Downsampling
        for _ in range(2):
            out_features *= 2
            layer += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            
            layer_1 += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        
        # Residual blocks
        
        for _ in range(num_residual_blocks):
            layer += [ResidualBlock(out_features)]
            layer_1 += [ResidualBlock(out_features)]
            
        
        self.layer = nn.Sequential(*layer) 
        self.layer_1 = nn.Sequential(*layer_1)
        
        self.coord = crt.CoordAtt(in_features, out_features)
        model = [
            nn.ReflectionPad2d(1),# (3,3,3,3)
            nn.Conv2d(in_features, out_features, 3),
        ]
        # Upsampling
        # for _ in range(2):
        out_features //= 2
        model += [
            nn.ConvTranspose2d(in_features, out_features, (3, 3), stride=(2, 2), padding=(1, 1),output_padding=(1,1)),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        out_features = in_features * 2
        model += [
            nn.ConvTranspose2d(in_features, out_features, (3, 3), stride=(2, 2), padding=(1, 1),output_padding=(1,1)),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(out_features, 3, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

        
    def forward(self, x, y):

        x0 = self.layer(x)
        y0 = self.layer_1(y)
 
        x1 = self.coord(x0, y0)

        x2 = self.model(x1)

        return x2


##############################
#        Discriminator
##############################
class TriDiscriminatorA(nn.Module):
    def __init__(self, input_shape):
        super(TriDiscriminatorA, self).__init__()

        channels, height, width = input_shape[2:]

        # PatchGAN 输出尺寸
        self.output_shape = (1, height // 2**4, width // 2**4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 共享 backbone
        self.backbone = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1,0,1,0)),
        )

        # 三条流独立 head
        self.head_image = nn.Conv2d(512, 1, 4, padding=1)
        self.head_flow  = nn.Conv2d(512, 1, 4, padding=1)
        self.head_depth = nn.Conv2d(512, 1, 4, padding=1)

    def forward(self, x_image=None, x_flow=None, x_depth=None):
        """
        可以分别输入 image / flow / depth，返回三条流的输出
        """
        out_image, out_flow, out_depth = None, None, None

        if x_image is not None:
            feat = self.backbone(x_image)
            out_image = self.head_image(feat)

        if x_flow is not None:
            feat = self.backbone(x_flow)
            out_flow = self.head_flow(feat)

        if x_depth is not None:
            feat = self.backbone(x_depth)
            out_depth = self.head_depth(feat)

        return out_image, out_flow, out_depth

class TriDiscriminatorB(nn.Module):
    def __init__(self, input_shape):
        """
        input_shape: [batch, channels, height, width]，channels通常是3（RGB/光流/深度）
        """
        super(TriDiscriminatorB, self).__init__()

        channels, height, width = input_shape[2:]

        # PatchGAN 输出尺寸
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 三个共享 backbone
        self.backbone = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1,0,1,0))
        )

        # 三个独立输出 head
        self.head_image = nn.Conv2d(512, 1, 4, padding=1)
        self.head_flow  = nn.Conv2d(512, 1, 4, padding=1)
        self.head_depth = nn.Conv2d(512, 1, 4, padding=1)

    def forward(self, x_image=None, x_flow=None, x_depth=None):
        """
        可以分别输入 image / flow / depth，返回三条流的输出
        """
        out_image, out_flow, out_depth = None, None, None
        if x_image is not None:
            feat = self.backbone(x_image)
            out_image = self.head_image(feat)

        if x_flow is not None:
            feat = self.backbone(x_flow)
            out_flow = self.head_flow(feat)

        if x_depth is not None:
            feat = self.backbone(x_depth)
            out_depth = self.head_depth(feat)

        return out_image, out_flow, out_depth  

class DiscriminatorB(nn.Module):
    def __init__(self, input_shape):
        super(DiscriminatorB, self).__init__()

        channels = input_shape[1]*input_shape[2]
        height, width = input_shape[3:]

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
  
    
if __name__ == '__main__':
    img_height = img_width = 256

    input_shape = (1,5,3,img_height, img_width)

    G = GeneratorResNet(input_shape,9)

    test_T = torch.rand((2,12,img_height,img_width))
    from torchsummary import summary
    summary(G, input_size=(12,img_height, img_width) )
    output = G(test_T)
    print(output.size())
