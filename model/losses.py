import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def crossentropy(outputs, labels):

    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples

def get_loss_fn(loss_name):

    if loss_name=='crossentropy':
        return crossentropy
    else:
        return crossentropy