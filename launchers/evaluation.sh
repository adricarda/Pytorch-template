#!/bin/bash

python3 evaluate.py \
    --data_dir '../data/cityscapes' \
    --model_dir '/content/drive/My Drive/Pytorch-template/experiments/baseline' \
    --checkpoint_dir '/content/drive/My Drive/Pytorch-template/experiments/baseline' \
    --txt_val '../data/cityscapes_val.txt'