import torch
import math
import argparse
import os
import utils
import matplotlib.pyplot as plt
import dataloader.dataloader as data_loader
from model.net import get_network
from model.losses import get_loss_fn
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/baseline',
                    help="Directory containing params.json")
parser.add_argument('--checkpoint_dir', default=None,
                    help="Directory containing weights to reload before \
                    training")

def find_lr(data_ld, opt, model, criterion, init_value = 1e-8, final_value=10., beta = 0.98):
    num = len(data_ld)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in data_ld:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        opt.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.data[0]
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        opt.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses

def plot_lr(log_lrs, losses):
    plt.plot(log_lrs[10:-5],losses[10:-5])
    plt.savefig('foo.png')
    

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params.device = device

    # Set the random seed for reproducible experiments
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    train_dl = data_loader.fetch_dataloader(args.data_dir, 'train', params)

    # Define the model and optimizer
    model = get_network(params).to(params.device)
    opt = optim.AdamW(model.parameters(), lr=params.learning_rate)
    loss_fn = get_loss_fn(loss_name=params.loss_fn , ignore_index=19)

    if args.checkpoint_dir:
        utils.load_checkpoint(model, False, args.checkpoint_dir) 
    
    log_lrs, losses = (train_dl, opt, model, loss_fn)
    plot_lr(log_lrs, losses)
