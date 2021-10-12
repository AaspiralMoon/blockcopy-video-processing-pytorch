""" 
ResNet for 32 by 32 images (CIFAR)
"""
BATCHNORM_MOMENTUM = 0.1

import math
import logging 

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """Standard residual block """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        assert groups == 1
        assert dilation == 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


########################################
# Original ResNet                      #
########################################

class ResNet_32x32(nn.Module):
    def __init__(self, layers, num_classes=10, in_channels=3,width_factor=1):
        super(ResNet_32x32, self).__init__()
        

        assert len(layers) == 3
        block = BasicBlock
        

        self.in_channels = in_channels
        self.inplanes = int(16*width_factor)
        self.conv1 = conv3x3(in_channels, int(16*width_factor))
        self.bn1 = nn.BatchNorm2d(int(16*width_factor), momentum=BATCHNORM_MOMENTUM)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, int(16*width_factor), layers[0])
        self.layer2 = self._make_layer(block, int(32*width_factor), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(64*width_factor), layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.OUT_CHANNELS = int(64*width_factor)# self.inplanes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BATCHNORM_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def load_state_dict(self, state_dict, strict: bool):
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[:7] == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        
        
        logging.info('Resnet32x32 loading pretrained state dict')
        if self.in_channels != 3:
            del new_state_dict['conv1.weight']
        return super().load_state_dict(new_state_dict, strict=strict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x
        
def resnet8(pretrained=True, **kwargs):
    model = ResNet_32x32([1,1,1], **kwargs)
    if pretrained:
        checkpoint = torch.load('pretrained/resnet8.pth')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def resnet8_wf2(pretrained=True, **kwargs):
    kwargs['width_factor'] = 2
    model = ResNet_32x32([1,1,1], **kwargs)
    if pretrained:
        checkpoint = torch.load('pretrained/resnet8_wf2.pth')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model 

def resnet14(pretrained=True, **kwargs):
    model = ResNet_32x32([2,2,2], **kwargs)
    if pretrained:
        checkpoint = torch.load('pretrained/resnet14.pth')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def resnet20(pretrained=True, **kwargs):
    model = ResNet_32x32([3,3,3], **kwargs)
    if pretrained:
        checkpoint = torch.load('pretrained/resnet20.pth')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def resnet26(pretrained=True, **kwargs):
    model = ResNet_32x32([4,4,4], **kwargs)
    if pretrained:
        checkpoint = torch.load('pretrained/resnet26.pth')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model

def resnet32(pretrained=True, **kwargs):
    model = ResNet_32x32([5,5,5], **kwargs)
    if pretrained:
        state_dict = torch.load('pretrained/resnet32.pth')
        model.load_state_dict(state_dict, strict=False)
    return model
