import torch

import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from utils.util import conv3x3, conv1x1, softmaxNd

import torch.utils.model_zoo as model_zoo

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class BasicBlockFeatures(BasicBlock):
    '''
    BasicBlock that returns its last conv layer features.
    '''

    def forward(self, x):

        if isinstance(x, tuple):
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        conv2_rep = out
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, conv2_rep


class BottleneckFeatures(Bottleneck):
    '''
    Bottleneck that returns its last conv layer features.
    '''

    def forward(self, x):

        if isinstance(x, tuple):
            x = x[0]

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        conv3_rep = out
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, conv3_rep


class ResNetFeatures(ResNet):
    '''
    A ResNet that returns features instead of classification.
    '''

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, c2 = self.layer1(x)
        x, c3 = self.layer2(x)
        x, c4 = self.layer3(x)
        x, c5 = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        c5 = self.fc(x)

        return c2, c3, c4, c5


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet18_features(pretrained=False, **kwargs):
    '''Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNetFeatures(BasicBlockFeatures, [2, 2, 2, 2], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model


def resnet34_features(pretrained=False, **kwargs):
    '''Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNetFeatures(BasicBlockFeatures, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    return model


def resnet50_features(pretrained=False, **kwargs):
    '''Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


def resnet101_features(pretrained=False, **kwargs):
    '''Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNetFeatures(BottleneckFeatures, [3, 4, 23, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

    return model


def resnet152_features(pretrained=False, **kwargs):
    '''Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    '''
    model = ResNetFeatures(BottleneckFeatures, [3, 8, 36, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    return model


class FPN(BaseModel):
    def __init__(self, args):
        super(FPN, self).__init__()
        self.args = args

        self.resnet = globals()[args.backbone + '_features'](pretrained=True)

        # applied in a pyramid
        self.latter_connect_2 = conv1x1(256, 256)
        self.latter_connect_3 = conv1x1(512, 256)
        self.latter_connect_4 = conv1x1(1024, 256)
        self.latter_connect_5 = conv1x1(2048, 256)

        self.linear = nn.Linear(256*3, args.num_classes)

    def _upsample_add(self, x, y):
        # is this correct? You do lose information on the upscale...
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):

        # First tower: bottom up
        c2, c3, c4, c5 = self.resnet(x)

        # Second tower: top down
        g = self.latter_connect_5(c5)
        p3 = self.latter_connect_4(c4)
        p3 = self._upsample_add(g, p3)
        p2 = self.latter_connect_3(c3)
        p2 = self._upsample_add(p3, p2)
        p1 = self.latter_connect_2(c2)
        p1 = self._upsample_add(p2, p1)

        # From fpn to attention
        l1, l2, l3 = p1, p2, p3

        # Third tower: transform
        c1 = (l1 * g).sum(1).unsqueeze(1)
        c2 = (l2 * g).sum(1).unsqueeze(1)
        c3 = (l3 * g).sum(1).unsqueeze(1)

        a1 = softmaxNd(c1)
        a2 = softmaxNd(c2)
        a3 = softmaxNd(c3)

        p1 = (a1 * l1).sum(-1).sum(-1)
        p2 = (a2 * l2).sum(-1).sum(-1)
        p3 = (a3 * l3).sum(-1).sum(-1)

        p = torch.cat([p1, p2, p3], dim=1)

        out = self.linear(p)

        a1 = F.upsample(a1, x.size(-1), mode='bilinear')
        a2 = F.upsample(a2, x.size(-1), mode='bilinear')
        a3 = F.upsample(a3, x.size(-1), mode='bilinear')

        return out, a1, a2, a3


class ResNetAttention(BaseModel):
    def __init__(self, args):
        super(ResNetAttention, self).__init__()
        self.args = args

        self.resnet = globals()[args.backbone + '_features'](pretrained=True)

        # applied in a pyramid
        self.latter_connect_2 = conv1x1(256, 256)
        self.latter_connect_3 = conv1x1(512, 256)
        self.latter_connect_4 = conv1x1(1024, 256)
        self.latter_connect_5 = conv1x1(2048, 256)

        self.linear = nn.Linear(256*3, args.num_classes)

    def forward(self, x):

        # First tower: bottom up
        c2, c3, c4, c5 = self.resnet(x)

        # Second tower: top down
        g = self.latter_connect_5(c5)
        l3 = self.latter_connect_4(c4)
        l2 = self.latter_connect_3(c3)
        l1 = self.latter_connect_2(c2)

        # Third tower: transform
        c1 = (l1 * g).sum(1).unsqueeze(1)
        c2 = (l2 * g).sum(1).unsqueeze(1)
        c3 = (l3 * g).sum(1).unsqueeze(1)

        a1 = softmaxNd(c1)
        a2 = softmaxNd(c2)
        a3 = softmaxNd(c3)

        p1 = (a1 * l1).sum(-1).sum(-1)
        p2 = (a2 * l2).sum(-1).sum(-1)
        p3 = (a3 * l3).sum(-1).sum(-1)

        p = torch.cat([p1, p2, p3], dim=1)

        out = self.linear(p)

        a1 = F.upsample(a1, x.size(-1), mode='bilinear')
        a2 = F.upsample(a2, x.size(-1), mode='bilinear')
        a3 = F.upsample(a3, x.size(-1), mode='bilinear')

        return out, a1, a2, a3
