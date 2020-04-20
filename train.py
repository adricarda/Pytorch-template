"""Train the model"""

import argparse
import copy
import logging
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import dataloader.dataloader as data_loader
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
parser.add_argument('--checkpoint_dir', default="experiments/baseline",
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
        y_pred = torch.argmax(y_pred, axis=1)
    return y_pred


def train_epoch(model, loss_fn, dataset_dl, opt=None, lr_scheduler=None, metrics=None, params=None):
    running_loss = utils.RunningAverage()
    num_batches = len(dataset_dl)

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metric.reset()

    for (xb, yb) in tqdm(dataset_dl):
        xb = xb.to(params.device)
        yb = yb.to(params.device)
        output = model(xb)['out']
        loss_b = loss_fn(output, yb)

        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss.update(loss_b.item())

        if metrics is not None:
            output = torch.argmax(output, dim=1)
            for metric_name, metric in metrics.items():
                metric.add(output.detach(), yb)

    if metrics is not None:
        metrics_results = {}
        for metric_name, metric in metrics.items():
            metrics_results[metric_name] = metric.value()
        return running_loss(), metrics_results
    else:
        return running_loss(), None


def train_and_evaluate(model, train_dl, val_dl, opt, loss_fn, metrics, params,
                       lr_scheduler, checkpoint_dir, ckpt_filename, writer):

    # todo restore best checkpoint
    ckpt_file_path = os.path.join(checkpoint_dir, ckpt_filename)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_value = float('inf')
    start_epoch = 0

    for xb, yb in val_dl:
        batch_sample = xb
        batch_gt = yb
        break

    if os.path.exists(ckpt_file_path):
        model, opt, lr_scheduler, start_epoch, best_value = utils.load_checkpoint(model, opt, lr_scheduler,
                                                                                  start_epoch, False, best_value, checkpoint_dir, ckpt_filename)

        print("=> loaded checkpoint form {} (epoch {})".format(
            ckpt_file_path, start_epoch))
    else:
        print("=> Initializing from scratch")

    for epoch in range(start_epoch, start_epoch + params.num_epochs-1):
        # Run one epoch
        current_lr = get_lr(opt)
        logging.info('Epoch {}/{}, current lr={}'.format(epoch,
                                                         params.num_epochs, current_lr))
        writer.add_scalar('Learning_rate', current_lr, epoch)

        model.train()
        train_loss, train_metrics = train_epoch(
            model, loss_fn, train_dl, opt, lr_scheduler, metrics, params)

        # Evaluate for one epoch on validation set
        val_loss, val_metrics = evaluate(
            model, loss_fn, val_dl, metrics=metrics, params=params)

        writer.add_scalars('Loss', {
            'Training': train_loss,
            'Validation': val_loss,
        }, epoch)

        for (train_metric_name, train_metric_results), (val_metric_name, val_metric_results) in zip(train_metrics.items(), val_metrics.items()):
            writer.add_scalars(train_metric_name, {
                'Training': train_metric_results[0],
                'Validation': val_metric_results[0],
            }, epoch)

        # TODO inserisci get prediction nel dataloader
        predictions = inference(model, batch_sample)
        plot = train_dl.dataset.get_predictions_plot(batch_sample, predictions, batch_gt)
        writer.add_image('Predictions', plot, epoch, dataformats='HWC')

        # get value for first metric
        current_value = list(val_metrics.values())[0][0]
        is_best = current_value >= best_value
        print()

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_value = current_value
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                checkpoint_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': opt.state_dict(),
                               'scheduler_dict': lr_scheduler.state_dict(),
                               'best_value': best_value},
                              is_best=is_best,
                              checkpoint_dir=checkpoint_dir)

        logging.info("\ntrain loss: %.3f, val loss: %.3f" %
                     (train_loss, val_loss))
        for (train_metric_name, train_metric_results), (val_metric_name, val_metric_results) in zip(train_metrics.items(), val_metrics.items()):
            logging.info("train %s: %.3f, val %s: %.3f" % (
                train_metric_name, train_metric_results[0], val_metric_name, val_metric_results[0]))

        logging.info("-"*20)


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
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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
    opt = optim.AdamW(model.parameters(), lr=params.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=params.learning_rate, steps_per_epoch=len(train_dl), epochs=params.num_epochs, div_factor=20)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(loss_name=params.loss_fn, ignore_index=19)
    # num_classes+1 for background.
    metrics = {}
    for metric in params.metrics:
        metrics[metric] = get_metrics(metrics_name=metric,
                                      num_classes=params.num_classes+1, ignore_index=params.ignore_index)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, opt, loss_fn, metrics,
                       params, lr_scheduler, args.checkpoint_dir, ckpt_filename, writer)
