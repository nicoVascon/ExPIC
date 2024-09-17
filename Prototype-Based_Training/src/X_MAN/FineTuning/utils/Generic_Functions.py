import torch


import math

import sys
import os

def progressbar(it, prefix="", sufix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{} {}".format(prefix, u"█"*x, "."*(size-x), j, count, sufix), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)
    
def plot_metrics(history, metric, figures_dir):
    from matplotlib import pyplot as plt
    epochs = range(1, len(history["train"][metric]) + 1)
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 5))
    
    ax0.plot(epochs, history["train"][metric], 'bo-', label='Training ' + metric)
    ax0.set_title('Training ' + metric)
    ax1.plot(epochs, history["val"][metric], 'ro-', label='Validation ' + metric)
    ax1.set_title('Validation ' + metric)
    
    # set labels
    plt.setp((ax0, ax1), xlabel="Epochs")
    plt.setp((ax0, ax1), ylabel=metric)
    
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    plt.savefig(os.path.join(figures_dir, metric + ".png"))
    plt.close()

def make_data_parallel(net,gpu_ids):                                                                        
    """Make models data parallel"""                                                                        
    if len(gpu_ids) == 0:                                                                                       
        return                                                                                             
    else:                                                                                                       
        return torch.nn.DataParallel(net,gpu_ids)  # multi-GPUs 

def freeze_layers(net, layer_to_freeze_until):
    trainable = False
    for name, param in net.named_parameters():
        if layer_to_freeze_until in name:
            # break
            trainable = True
        
        param.requires_grad = trainable

class EarlyStopping:
    """ Autor: Bjarten; url: https://github.com/Bjarten/early-stopping-pytorch"""
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = math.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:e} --> {val_loss:e}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss