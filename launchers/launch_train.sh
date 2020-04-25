#!/bin/bash

python3 ../train.py \
    --data_dir '/content/drive/My Drive/cityscapes' \
    --model_dir '/content/drive/My Drive/Pytorch-template/experiments/resnet50' \
    --checkpoint_dir '/content/drive/My Drive/Pytorch-template/experiments/resnet50/ckpt' \
    --tensorboard_dir '/content/drive/My Drive/Pytorch-template/experiments/resnet50/tensorboard'