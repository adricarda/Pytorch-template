"""Defines the neural network"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import deeplabv3_resnet50


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

    def forward(self, s):
        pass

def get_network(params):
    if params.architecture == 'deeplab_resnet101':
        return deeplabv3_resnet101(pretrained=False, num_classes=params.num_classes)
    if params.architecture == 'deeplab_resnet101':
        return deeplabv3_resnet50(pretrained=False, num_classes=params.num_classes)