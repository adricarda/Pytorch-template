# Pytorch-template
This is a template for semantic segmentation projects in Pytorch, but it can be easily adapted to other cases.
In this example, the Cityscapes dataset is used (it must be downloaded first). The network used is a predefined DeepLabv3 available from the torchvision package.

This template includes :
1) Learning rate finder
2) Reading parameters from a configuration file
3) Logging into a log file
4) Tensorboard summary for learning rate, loss, metrics

Many functions are taken from other repositories and the structure of the project is inspired by https://github.com/cs230-stanford/cs230-code-examples.
The goal of the template is to have something to start on when one wants to quickly implement some new ideas without rewriting everything from scratch.
