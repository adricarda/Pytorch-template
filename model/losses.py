import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def crossentropy(ignore_index=19):
    return nn.CrossEntropyLoss(ignore_index)

def get_loss_fn(loss_name):

    if loss_name=='crossentropy':
        return crossentropy
    else:
        return crossentropy