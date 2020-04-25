# Pytorch semantic segmentation template
This is a template for semantic segmentation projects in Pytorch, but it can be easily adapted to other cases.
In this example, the Cityscapes dataset is used (it must be downloaded first). The network used is a predefined DeepLabv3 available from the torchvision package.

This template includes :

1) Learning rate finder
2) Reading parameters from a configuration file
3) Logging into a log file
4) Tensorboard summary for learning rate, loss, metrics for both training and validation
5) Advanced learning rate decay (1cycle policy)
6) Early stopping

Many functions are taken from other repositories and the structure of the project is inspired by https://github.com/cs230-stanford/cs230-code-examples.
The goal of the template is to have something to start on when one wants to quickly implement some new ideas without rewriting everything from scratch. 
Performance is thereby not the main goal, but just for reference this model achieves about 77% of mIoU on the Cityscapes validation set, without using coarse annotations/ensembling/complex data augmentation.

The data folder should contain your dataset togheter with two txt files listing paths for the trainig and validation set.
Parameters of a training should be written in a json file called params.json in experiments/name_of_the_experiment. For each experiment you should create a dedicated directory with its corresponding json so that nothing it's lost.

Logs will be automatically saved in experiments/name_of_the_experiment/logs
Tensorboard files will be saved in experiments/name_of_the_experiment/tensorboard
last checkpoints and the best model checkpoint are stored in experiments/name_of_the_experiment/ckpt

Some launcher scripts are provided in launchers/