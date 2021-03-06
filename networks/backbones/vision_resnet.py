import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_down0 = self.layer1(x)
        x_down1 = self.layer2(x_down0)
        x_down2 = self.layer3(x_down1)
        x_down3 = self.layer4(x_down2)

        return x_down0, x_down1, x_down2, x_down3


def resnet18(in_channels, net_config):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(in_channels, BasicBlock, [2, 2, 2, 2])
    if net_config["pretrained"]:
        pretrain_dict = torch.load(net_config["pretrain_checkpoint"])
        model_dict = {}
        state_dict = self.state_dict()
        ignore_prefixs = net_config["ignore_prefixs"]
        for k, v in pretrain_dict.items():
            k = k.replace("module.", "") # replace the module in checkpoint which is caused by the training using multiple GPUs
            passed = True
            for ignore_prefix in ignore_prefixs:
                if k.startswith(ignore_prefix):
                    passed = False
            
            if passed and k in state_dict:
                print("load %s"%(k))
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model


def resnet34(in_channels, net_config):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(in_channels, BasicBlock, [3, 4, 6, 3])
    if net_config["pretrained"]:
        pretrain_dict = torch.load(net_config["pretrain_checkpoint"])
        model_dict = {}
        state_dict = model.state_dict()
        ignore_prefixs = net_config["ignore_prefixs"]
        for k, v in pretrain_dict.items():
            k = k.replace("module.", "") # replace the module in checkpoint which is caused by the training using multiple GPUs
            passed = True
            for ignore_prefix in ignore_prefixs:
                if k.startswith(ignore_prefix):
                    passed = False
            
            if passed and k in state_dict:
                print("load %s"%(k))
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model


def resnet50(in_channels, net_config):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(in_channels, Bottleneck, [3, 4, 6, 3])
    if net_config["pretrained"]:
        pretrain_dict = torch.load(net_config["pretrain_checkpoint"])
        model_dict = {}
        state_dict = model.state_dict()
        ignore_prefixs = net_config["ignore_prefixs"]
        for k, v in pretrain_dict.items():
            k = k.replace("module.", "") # replace the module in checkpoint which is caused by the training using multiple GPUs
            passed = True
            for ignore_prefix in ignore_prefixs:
                if k.startswith(ignore_prefix):
                    passed = False
            
            if passed and k in state_dict:
                print("load %s"%(k))
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model


def resnet101(in_channels, net_config):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(in_channels, Bottleneck, [3, 4, 23, 3])
    if net_config["pretrained"]:
        pretrain_dict = torch.load(net_config["pretrain_checkpoint"])
        model_dict = {}
        state_dict = model.state_dict()
        ignore_prefixs = net_config["ignore_prefixs"]
        for k, v in pretrain_dict.items():
            k = k.replace("module.", "") # replace the module in checkpoint which is caused by the training using multiple GPUs
            passed = True
            for ignore_prefix in ignore_prefixs:
                if k.startswith(ignore_prefix):
                    passed = False
            
            if passed and k in state_dict:
                print("load %s"%(k))
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model


def resnet152(in_channels, net_config):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(in_channels, Bottleneck, [3, 8, 36, 3])
    if net_config["pretrained"]:
        pretrain_dict = torch.load(net_config["pretrain_checkpoint"])
        model_dict = {}
        state_dict = model.state_dict()
        ignore_prefixs = net_config["ignore_prefixs"]
        for k, v in pretrain_dict.items():
            k = k.replace("module.", "") # replace the module in checkpoint which is caused by the training using multiple GPUs
            passed = True
            for ignore_prefix in ignore_prefixs:
                if k.startswith(ignore_prefix):
                    passed = False
            
            if passed and k in state_dict:
                print("load %s"%(k))
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model
