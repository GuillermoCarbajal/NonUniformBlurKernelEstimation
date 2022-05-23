""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F



class TwoHeadsNetwork(nn.Module):
    def __init__(self, K=9, blur_kernel_size=33, bilinear=False,
                 no_softmax=False):
        super(TwoHeadsNetwork, self).__init__()

        self.no_softmax = no_softmax
        if no_softmax:
            print('Softmax is not being used')

        self.inc_rgb = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.inc_gray = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.blur_kernel_size = blur_kernel_size
        self.K=K

        self.down1 = Down(64, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)
        self.feat =   nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.up1 = Up(1024,1024, 512, bilinear)
        self.up2 = Up(512,512, 256, bilinear)
        self.up3 = Up(256,256, 128, bilinear)
        self.up4 = Up(128,128, 64, bilinear)
        self.up5 = Up(64,64, 64, bilinear)

        self.masks_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, K, kernel_size=3, padding=1),
            nn.Softmax(dim=1),
        )

        self.feat5_gap = PooledSkip(2)
        self.feat4_gap = PooledSkip(4)  
        self.feat3_gap = PooledSkip(8)  
        self.feat2_gap = PooledSkip(16)  
        self.feat1_gap = PooledSkip(32) 

        self.kernel_up1 = Up(1024,1024, 512, bilinear)
        self.kernel_up2 = Up(512,512, 256, bilinear)
        self.kernel_up3 = Up(256,256, 256, bilinear)
        self.kernel_up4 = Up(256,128, 128, bilinear)
        self.kernel_up5 = Up(128,64, 64, bilinear)
        if self.blur_kernel_size>33:
            self.kernel_up6 = Up(64, 0, 64, bilinear)

        self.kernels_end = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, K, kernel_size=3, padding=1)
            #nn.Conv2d(128, K*self.blur_kernel_size*self.blur_kernel_size, kernel_size=8),
        )
        self.kernel_softmax = nn.Softmax(dim=2)

    def forward(self, x):
        #Encoder
        if x.shape[1]==3:
            x1 = self.inc_rgb(x)
        else:
            x1 = self.inc_gray(x)
        x1_feat, x2 = self.down1(x1)
        x2_feat, x3 = self.down2(x2)
        x3_feat, x4 = self.down3(x3)
        x4_feat, x5 = self.down4(x4)
        x5_feat, x6 = self.down5(x5)
        x6_feat = self.feat(x6)

        #k = self.kernel_network(x3)
        feat6_gap = x6_feat.mean((2,3), keepdim=True) #self.feat6_gap(x6_feat)
        #print('x6_feat: ', x6_feat.shape,'feat6_gap: ' , feat6_gap.shape)
        feat5_gap = self.feat5_gap(x5_feat)
        #print('x5_feat: ', x5_feat.shape,'feat5_gap: ' , feat5_gap.shape)
        feat4_gap = self.feat4_gap(x4_feat)
        #print('x4_feat: ', x4_feat.shape,'feat4_gap: ' , feat4_gap.shape)
        feat3_gap = self.feat3_gap(x3_feat)
        #print('x3_feat: ', x3_feat.shape,'feat3_gap: ' , feat3_gap.shape)
        feat2_gap = self.feat2_gap(x2_feat)
        #print('x2_feat: ', x2_feat.shape,'feat2_gap: ' , feat2_gap.shape)
        feat1_gap = self.feat1_gap(x1_feat)
        #print(feat5_gap.shape, feat4_gap.shape)
        k1 = self.kernel_up1(feat6_gap, feat5_gap)
        #print('k1 shape', k1.shape)
        k2 = self.kernel_up2(k1, feat4_gap)
        #print('k2 shape', k2.shape)
        k3 = self.kernel_up3(k2, feat3_gap)
        #print('k3 shape', k3.shape)
        k4 = self.kernel_up4(k3, feat2_gap)
        #print('k4 shape', k4.shape)
        k5 = self.kernel_up5(k4, feat1_gap)

        if self.blur_kernel_size==65:
            k6 = self.kernel_up6(k5)
            k = self.kernels_end(k6)
        else:
            k = self.kernels_end(k5)
        N, F, H, W = k.shape  # H and W should be one
        k = k.view(N, self.K, self.blur_kernel_size * self.blur_kernel_size)

        if self.no_softmax:
            k = functional.leaky_relu(k)
            #suma = k5.sum(2, keepdim=True)
            #k = k5 / suma
        else:
            k = self.kernel_softmax(k)

        k = k.view(N, self.K, self.blur_kernel_size, self.blur_kernel_size)

        #Decoder
        x7 = self.up1(x6_feat, x5_feat)
        x8 = self.up2(x7, x4_feat)
        x9 = self.up3(x8, x3_feat)
        x10 = self.up4(x9, x2_feat)
        x11 = self.up5(x10, x1_feat)
        logits = self.masks_end(x11)

        return  k,logits
""" Parts of the U-Net model """


class Down(nn.Module):
    """double conv and then downscaling with maxpool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,  kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
           # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            #nn.BatchNorm2d(out_channels),
        )


        self.down_sampling = nn.MaxPool2d(2)


    def forward(self, x):
        feat = self.double_conv(x)
        down_sampled = self.down_sampling(feat)
        return feat, down_sampled


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, feat_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels,  in_channels, kernel_size=2, stride=2)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.feat = nn.Sequential(
            nn.Conv2d(feat_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x1, x2=None):
        #print('initial x1: ', x1.shape)
        x1 = self.up(x1)
        x1 = self.double_conv(x1)
        #print('x1 after upsampling: ', x1.shape)

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        if x2 is not None:
          
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        feat = self.feat(x)
        return feat

class PooledSkip(nn.Module):
    def __init__(self, output_spatial_size):
        super().__init__()

        self.output_spatial_size = output_spatial_size

    def forward(self, x):
        global_avg_pooling = x.mean((2,3), keepdim=True) #self.gap(x)
        #print('gap shape:' , global_avg_pooling.shape)
        return global_avg_pooling.repeat(1,1,self.output_spatial_size,self.output_spatial_size)
