"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.net as net
from tqdm import tqdm
import dataloader.dataloader as data_loader
from model.losses import get_loss_fn
from model.metrics import get_metrics

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/baseline',
                    help="Directory containing params.json")
parser.add_argument('--checkpoint_dir', default="experiments/baseline/checkpoints",
                    help="Directory containing weights to reload before \
                    training") 


def evaluate(model,loss_fn,dataset_dl,opt=None, metrics=None, params=None):

    # set model to evaluation mode
    model.eval()
    running_loss=utils.RunningAverage()
    num_batches=len(dataset_dl)
    metrics.reset()

    with torch.no_grad():        
        for (xb, yb) in tqdm(dataset_dl):
            xb=xb.to(params.device)
            yb=yb.to(params.device)    
            output=model(xb)['out']
            
            loss_b = loss_fn(output, yb)
            running_loss.update(loss_b.item())
            output=torch.argmax(output.detach(), dim=1)
            metrics.add(output, yb.detach())

    return loss_b.item(), metrics.value()

    #     # compute all metrics on this batch
    #     summary_batch = {metric: metrics[metric](output_batch, labels_batch)
    #                      for metric in metrics}
    #     summary_batch['loss'] = loss.item()
    #     summ.append(summary_batch)

    # # compute mean of all metrics in summary
    # metrics_mean = {metric: np.mean([x[metric]
    #                                  for x in summ]) for metric in summ[0]}
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
    #                             for k, v in metrics_mean.items())
    # logging.info("- Eval metrics : " + metrics_string)
    # return metrics_mean


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
    ckpt_filename = "checkpoint.tar"
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params.device = device

    # Set the random seed for reproducible experiments
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).to(params.device)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(params.loss_fn)
    metrics = get_metrics(params.metrics)

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(model, True, args.checkpoint_dir)

    # Evaluate
    eval_loss, test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    # save_path = os.path.join(args.model_dir, "metrics_test.json")
    # utils.save_dict_to_json(test_metrics, save_path)