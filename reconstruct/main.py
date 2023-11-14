import config
import losses
import models
import training
import utils
import data_loader

import torch
import torch.nn as nn
import monai
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

def main():
    # get config
    num_workers = config.num_workers
    batch_size = config.batch_size
    train_sampler = config.train_sampler
    flag_pretrain = config.flag_pretrain
    lr = config.lr
    val_interval = config.val_interval
    max_epochs = config.max_epochs
    lr_decay_flag = config.lr_decay_flag

    # load data
    train_loader, val_loader = data_loader.get_loader(num_workers, batch_size, train_sampler)

    # load model
    device = torch.device("cuda:0")
    model = models.SEResNet50_model(flag_pretrain, spatial_dims=2, in_channels=1, pretrained=False, num_classes=1)
    model = nn.Sequential(model, losses.Bachnorm())
    model.to(device)
    loss = losses.Spearman_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Cosine learning rate scheduler
    if lr_decay_flag:
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-7)
    else:
        scheduler = None

    # training
    training(model, 
             train_loader, val_loader, 
             lr, batch_size, max_epochs, val_interval, device, 
             optimizer, scheduler, loss, utils.spearman_hard_eval, debug_flag=False)

if __name__ == "__main__":
    main()