'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from base.base_model import BaseModel
from utils.util import softmaxNd, conv1x1
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models import vgg16_bn


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGGAttentionModel(BaseModel):
    cfg = {
        'VGGOriginal16': [(64, 64, 'M', 128, 128, 'M', 256, 256, 256), ('M', 512, 512, 512), ('M', 512, 512, 512), ('M',)],
        'VGG16': [(64, 64, 128, 128, 256, 256, (256, 2)), (512, 512, (512, 2)), (512, 512, (512, 2)), ((512, 2), (512, 2))],
    }

    def __init__(self, args, vgg_name):
        super(VGGAttentionModel, self).__init__()
        self.last_channel = None
        cfg = self.cfg
        self.features1 = self._make_layers(cfg[vgg_name][0])
        self.features2 = self._make_layers(cfg[vgg_name][1])
        self.features3 = self._make_layers(cfg[vgg_name][2])
        self.features4 = self._make_layers(cfg[vgg_name][3])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
        )
        self.transform1 = conv1x1(256, 512)
        self.transform2 = conv1x1(512, 512)
        self.transform3 = conv1x1(512, 512)
        self.linear = nn.Linear(512 * 3, args.num_classes)

    def forward(self, x):
        l1 = self.features1(x)
        l2 = self.features2(l1)
        l3 = self.features3(l2)
        l4 = self.features4(l3)
        g = self.classifier(l4.squeeze())

        l1 = self.transform1(l1)
        l2 = self.transform2(l2)
        l3 = self.transform3(l3)

        c1 = (l1 * g[:, :, None, None]).sum(1).unsqueeze(1)
        c2 = (l2 * g[:, :, None, None]).sum(1).unsqueeze(1)
        c3 = (l3 * g[:, :, None, None]).sum(1).unsqueeze(1)

        a1 = softmaxNd(c1)
        a2 = softmaxNd(c2)
        a3 = softmaxNd(c3)

        g1 = (a1 * l1).sum(-1).sum(-1)
        g2 = (a2 * l2).sum(-1).sum(-1)
        g3 = (a3 * l3).sum(-1).sum(-1)

        g = self.linear(torch.cat([g1, g2, g3], 1))

        a1 = F.upsample(a1, x.size(-1), mode='bilinear')
        a2 = F.upsample(a2, x.size(-1), mode='bilinear')
        a3 = F.upsample(a3, x.size(-1), mode='bilinear')

        return g, a1, a2, a3

    def _make_layers(self, cfg):
        layers = []
        if self.last_channel is None:
            in_channels = 3
        else:
            in_channels = self.last_channel

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(x, (tuple, list)):
                layers += [nn.Conv2d(in_channels, x[0], kernel_size=3, padding=1, stride=x[1]),
                           nn.BatchNorm2d(x[0]),
                           nn.ReLU(inplace=True)]
                in_channels = x[0]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        self.last_channel = x[0] if isinstance(x, (tuple, list)) else x
        return nn.Sequential(*layers)


class VGGModel(BaseModel):

    def __init__(self, args, pretrained=True):
        super(VGGModel, self).__init__()
        self.last_channel = None
        self.args = args
        vgg = vgg16_bn(pretrained)

        self.features1 = nn.Sequential(*list(vgg.features.children())[:23])
        self.features2 = nn.Sequential(*list(vgg.features.children())[23:33])
        self.features3 = nn.Sequential(*list(vgg.features.children())[33:43])
        self.features4 = nn.Sequential(*list(vgg.features.children())[43:44])

        if args.extract_only:
            for param in self.features1.parameters():
                param.requires_grad = False
            for param in self.features2.parameters():
                param.requires_grad = False
            for param in self.features3.parameters():
                param.requires_grad = False
            for param in self.features4.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
        )
        self.transform1 = conv1x1(256, 512)
        self.transform2 = conv1x1(512, 512)
        self.transform3 = conv1x1(512, 512)
        self.linear = nn.Linear(512 * 3, 512)

    def forward(self, x):
        if self.args.extract_only:
            x = Variable(x.data, volatile=True)

        l1 = self.features1(x)
        l2 = self.features2(l1)
        l3 = self.features3(l2)
        x = self.features4(l3)

        if self.args.extract_only:
            l1 = Variable(l1.data)
            l2 = Variable(l2.data)
            l3 = Variable(l3.data)
            x = Variable(x.data)

        x = F.adaptive_avg_pool2d(x, 1)
        g = self.classifier(x.squeeze())

        l1 = self.transform1(l1)
        l2 = self.transform2(l2)
        l3 = self.transform3(l3)

        c1 = (l1 * g[:, :, None, None]).sum(1).unsqueeze(1)
        c2 = (l2 * g[:, :, None, None]).sum(1).unsqueeze(1)
        c3 = (l3 * g[:, :, None, None]).sum(1).unsqueeze(1)

        a1 = softmaxNd(c1)
        a2 = softmaxNd(c2)
        a3 = softmaxNd(c3)

        g1 = (a1 * l1).sum(-1).sum(-1)
        g2 = (a2 * l2).sum(-1).sum(-1)
        g3 = (a3 * l3).sum(-1).sum(-1)

        g = self.linear(torch.cat([g1, g2, g3], 1))

        # a1 = F.upsample(a1, x.size(-1), mode='bilinear')
        # a2 = F.upsample(a2, x.size(-1), mode='bilinear')
        # a3 = F.upsample(a3, x.size(-1), mode='bilinear')

        return g

    def _make_layers(self, cfg):
        layers = []
        if self.last_channel is None:
            in_channels = 3
        else:
            in_channels = self.last_channel

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(x, (tuple, list)):
                layers += [nn.Conv2d(in_channels, x[0], kernel_size=3, padding=1, stride=x[1]),
                           nn.BatchNorm2d(x[0]),
                           nn.ReLU(inplace=True)]
                in_channels = x[0]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        self.last_channel = x[0] if isinstance(x, (tuple, list)) else x
        return nn.Sequential(*layers)


# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
