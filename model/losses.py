import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_fn(params):

    if params.loss_fn=='crossentropy':
        return nn.CrossEntropyLoss(ignore_index=params.ignore_index)
    else:
        raise NotImplementedError