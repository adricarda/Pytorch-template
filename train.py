"""Train the model"""

import argparse
import copy
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataloader.dataloader as data_loader
from dataloader.dataloader import get_predictions_plot
import utils
from evaluate import evaluate
from model.losses import get_loss_fn
from model.metrics import get_metrics
from model.net import get_network

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/baseline',
                    help="Directory containing params.json")
parser.add_argument('--checkpoint_dir', default="experiments/baseline/checkpoints",
                    help="Directory containing weights to reload before \
                    training")
parser.add_argument('--tensorboard_dir', default="experiments/baseline/tensorboard",
                    help="Directory for Tensorboard data")

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def inference(model, batch):
    model.eval()
    with torch.no_grad():
        y_pred = model(batch.to(device))
        y_pred = y_pred["out"].cpu()
        y_pred = torch.argmax(y_pred,axis=1)
    return y_pred

def train_epoch(model,loss_fn,dataset_dl,opt=None, metrics=None, params=None):
    running_loss=utils.RunningAverage()
    num_batches=len(dataset_dl)

    if metrics is not None:
        metrics.reset()
    
    for (xb, yb) in tqdm(dataset_dl):
        xb=xb.to(params.device)
        yb=yb.to(params.device)    
        output=model(xb)['out']
        loss_b = loss_fn(output, yb)

        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()

        running_loss.update(loss_b.item())

        if metrics is not None:
            output=torch.argmax(output.detach(), dim=1)
            metrics.add(output, yb.detach())

    if metrics is not None:
        return loss_b.item(), metrics.value()
    else:   
        return running_loss(), None

def train_and_evaluate(model, train_dl, val_dl, opt, loss_fn, metrics, params, 
                    lr_scheduler, model_dir, ckpt_filename, writer):

    ckpt_file_path = os.path.join(model_dir, ckpt_filename)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    start_epoch=0

    for xb, yb in val_dl:
        batch_sample = xb
        batch_gt = yb
        break

    if os.path.exists(ckpt_file_path): 
        checkpoint = torch.load(ckpt_file_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint form {} (epoch {})".format(ckpt_file_path, checkpoint['epoch']))
    else:
        print("=> Initializing from scratch")

    for epoch in range(start_epoch, start_epoch + params.num_epochs-1 ):
        # Run one epoch
        current_lr=get_lr(opt)
        logging.info('Epoch {}/{}, current lr={}'.format(epoch, start_epoch+params.num_epochs-1, current_lr))

        model.train()
        train_loss, train_metric = train_epoch(model, loss_fn, train_dl, opt, metrics, params)

        # Evaluate for one epoch on validation set
        val_loss, val_metric = evaluate(model, loss_fn, val_dl, metrics=metrics, params=params)

        
        writer.add_scalars('Loss', {
                                    'Training': train_loss,
                                    'Validation': val_loss,
                                  }, epoch)
        writer.add_scalars('Metric', {
                                    'Training': train_metric[1],
                                    'Validation': val_metric[1],
                                  }, epoch)
        
        predictions = inference(model, batch_sample)
        plot = get_predictions_plot(batch_sample, predictions, batch_gt)
        writer.add_image('Predictions', plot, epoch, dataformats='HWC')                          
        
        is_best = val_loss >= best_loss

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint_dir=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_loss = val_loss

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts) 
            
        logging.info("\ntrain loss: %.3f, val loss: %.3f" %(train_loss, val_loss))
        logging.info("train metric: %.3f, val metric: %.3f" %(train_metric[1], val_metric[1]))
        logging.info("-"*20)             

        #     # Save best val metrics in a json file in the model directory
        #     best_json_path = os.path.join(
        #         model_dir, "metrics_val_best_weights.json")
        #     utils.save_dict_to_json(val_metrics, best_json_path)

        # # Save latest val metrics in a json file in the model directory
        # last_json_path = os.path.join(
        #     model_dir, "metrics_val_last_weights.json")
        # utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    ckpt_filename = "checkpoint.tar"
    writer = SummaryWriter(args.tensorboard_dir)

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params.device = device

    # Set the random seed for reproducible experiments
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    train_dl = data_loader.fetch_dataloader(args.data_dir, 'train', params)
    val_dl = data_loader.fetch_dataloader(args.data_dir, 'val', params)

    logging.info("- done.")

    # Define the model and optimizer
    model = get_network(params).to(params.device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=4,verbose=1)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(loss_name=params.loss_fn , ignore_index=19)
    #num_classes+1 for background. 
    metrics = get_metrics(metrics_name=params.metrics, num_classes=params.num_classes+1, ignore_index=params.ignore_index)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics,
            params, lr_scheduler, args.model_dir, ckpt_filename, writer)
