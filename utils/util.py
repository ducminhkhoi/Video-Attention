import os
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def init_conv_weights(layer, weights_std=0.01, bias=0):
    '''
    RetinaNet's layer initialization
    :layer
    :
    '''
    nn.init.normal(layer.weight.data, std=weights_std)
    nn.init.constant(layer.bias.data, val=bias)
    return layer


def conv1x1(in_channels, out_channels, **kwargs):
    '''Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)

    return layer


def conv3x3(in_channels, out_channels, **kwargs):
    '''Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)

    return layer


def softmaxNd(tensor):
    shape = tensor.shape
    tensor = tensor.view(shape[0], shape[1], -1)
    tensor = F.softmax(tensor, -1)
    tensor = tensor.view(*shape)
    return tensor


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)

    cam = heatmap + img
    cam = cam / np.max(cam)
    ret = np.uint8(255 * cam)
    return ret


invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]),
                               transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465],
                                                    std=[1., 1., 1.]),
                               transforms.Lambda(lambda x: x.permute(1, 2, 0).numpy())
                              ])
