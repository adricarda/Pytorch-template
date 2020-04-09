import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_fn(loss_name='crossentropy', **kwargs):

    if loss_name=='crossentropy':
        return nn.CrossEntropyLoss(**kwargs)
    else:
        return nn.CrossEntropyLoss(**kwargs)