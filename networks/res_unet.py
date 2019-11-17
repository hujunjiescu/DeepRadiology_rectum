import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb
from backbones.vision_resnet import resnet50


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
                                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True))
    
    def forward(self, de_feature, enc_feature):
        de_feature = F.interpolate(de_feature, size=enc_feature.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([de_feature, enc_feature], dim=1)
        x = self.conv(x)
        return x

    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv_relu = nn.Sequential(
                          nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(inplace=True))
        
    def forward(self, x, target_size=None):
        x = self.conv_relu(x)
        if target_size == None:
            x = F.interpolate(x, size=(x.shape[2]*2, x.shape[3]*2), mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x
    
class ResUNet50(nn.Module):
    def __init__(self, in_channels, num_classes, net_config):
        super(ResUNet50, self).__init__()
        self.feature = resnet50(in_channels, net_config)
        self.decoder0 = Decoder(2048 + 1024, 1024)
        self.decoder1 = Decoder(1024 + 512, 512)
        self.decoder2 = Decoder(512 + 256, 256)
        self.final_conv = nn.Sequential(Upsample(256, 256),
                                        Upsample(256, 256),
                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(2048, num_classes))
    
    def forward(self, x):
        down0, down1, down2, out = self.feature(x) #256, 512, 1024, 2048
        x = self.decoder0(out, down2)
        x = self.decoder1(x, down1)
        x = self.decoder2(x, down0)
        
        gap_feature = torch.squeeze(self.gap(out))
        return self.final_conv(x), self.fc(gap_feature)