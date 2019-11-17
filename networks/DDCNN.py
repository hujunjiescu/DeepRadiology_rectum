import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb
from backbones.vision_resnet import resnet50
from backbones.vision_vgg import vgg16_bn

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DDCNN(nn.Module):
    def __init__(self, in_channels, num_classes, net_config):
        super(DDCNN, self).__init__()
        self.vgg = vgg16_bn(in_channels, net_config)
        
        self.aspp1 = ASPP_module(512, 768, rate=6)
        self.aspp2 = ASPP_module(512, 768, rate=12)
        self.aspp3 = ASPP_module(512, 768, rate=18)
        self.aspp4 = ASPP_module(512, 768, rate=24)
        
        self.final_conv = nn.Conv2d(768, num_classes, kernel_size=1, stride=1)
        
    def forward(self, x):
        out = self.vgg(x)
        
        x1 = self.aspp1(out)
        x2 = self.aspp2(out)
        x3 = self.aspp3(out)
        x4 = self.aspp4(out)
        
        out = torch.stack([x1, x2, x3, x4], dim=0).sum(dim=0)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        out = self.final_conv(out)
        return out, None