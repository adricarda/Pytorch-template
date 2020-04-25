#!/bin/bash

python3 ../train.py \
    --data_dir '/content/drive/My Drive/cityscapes' \
    --model_dir '/content/drive/My Drive/Pytorch-template/experiments/baseline' \
    --checkpoint_dir '/content/drive/My Drive/Pytorch-template/experiments/baseline/ckpt' \
    --tensorboard_dir '/content/drive/My Drive/Pytorch-template/experiments/baseline/tensorboard' \
    --txt_train '../data/cityscapes_train.txt' \
    --txt_val '../data/cityscapes_val.txt'