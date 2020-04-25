"""Evaluates the model"""

import argparse
import logging
import os
import random
import numpy as np
import torch
import utils.utils as utils
from model.net import get_network
from tqdm import tqdm
import dataloader.dataloader as dataloader
from model.losses import get_loss_fn
from model.metrics import get_metrics
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/baseline',
                    help="Directory containing params.json")
parser.add_argument('--checkpoint_dir', default="experiments/baseline/ckpt",
                    help="Directory containing weights to reload before \
                    training") 
parser.add_argument('--txt_val', default='/content/drive/My Drive/cityscapes/cityscapes_val.txt',
                    help="Txt file containing path to validation images")


def evaluate(model, dataset_dl, loss_fn=None, metrics=None, params=None):

    # set model to evaluation mode
    model.eval()
    metrics_results = {}

    if loss_fn is not None:
        running_loss = utils.RunningAverage()
    num_batches = len(dataset_dl)
    if metrics is not None:
        for metric_name, metric in metrics.items():
            metric.reset()

    with torch.no_grad():
        for (xb, yb) in tqdm(dataset_dl):
            xb = xb.to(params.device)
            yb = yb.to(params.device)
            output = model(xb)['out']

            if loss_fn is not None:
                loss_b = loss_fn(output, yb)
                running_loss.update(loss_b.item())
            if metrics is not None:
                for metric_name, metric in metrics.items():
                    metric.add(output, yb)

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metrics_results[metric_name] = metric.value()
    
    if loss_fn is not None:
        return running_loss(), metrics_results
    else:
        return None, metrics_results

if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
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

    # fetch dataloaders
    # fetch dataloaders
    val_dl = dataloader.fetch_dataloader(
        args.data_dir, args.txt_val, "val", params)
    # Define the model
    model = get_network(params).to(params.device)

    #num_classes+1 for background.
    metrics = OrderedDict({})
    for metric in params.metrics:
        metrics[metric]= get_metrics(metric, params)

    # Reload weights from the saved file
    model = utils.load_checkpoint(model, is_best=True, checkpoint_dir=args.checkpoint_dir)[0]

    # Evaluate
    eval_loss, val_metrics = evaluate(model, val_dl, loss_fn=None, metrics=metrics, params=params)
    
    best_json_path = os.path.join(args.model_dir, "logs/evaluation.json")
    for val_metric_name, val_metric_results in val_metrics.items(): 
        print("{}: {}".format(val_metric_name, val_metric_results))
    utils.save_dict_to_json(val_metrics, best_json_path)