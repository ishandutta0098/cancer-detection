import gc 
import torch
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from torch.cuda import amp
from sklearn.metrics import accuracy_score

import training.loss_fn as loss_fn

def train_one_epoch(cfg, model, optimizer, scheduler, dataloader, device, epoch, weights, scaler=None):
    """
    Function to train one epoch on the data

    Args:
        cfg (dict): Configuration File 
        model: Model definition
        optimizer: Optimizer for the network
        scheduler: Learning Rate Scheduler
        dataloader: PyTorch DataLoader
        device: GPU or CPU device for training
        epoch (int): Number of epochs
        weights (torch tensor): Class Weight values
        scaler: Scaler for AMP

    Returns:
        epoch_loss (float): Loss for the epoch
        epoch_accuracy(float): Percentage of correct model predictions
    """
    
    # Initialize model train mode
    model.train()
    
    # Initial parameters
    dataset_size = 0
    running_loss = 0.0
    train_correct = 0
    size_sampler = len(dataloader.sampler)
    
    # Progress bar
    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    # Iterate over batches
    for step, data in bar:

        # Get data
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        # When device = "cuda"
        # use automatic mixed precision
        if device == torch.device('cuda'):

            with amp.autocast():
            
                # Generate predictions
                outputs = model(images)

                # Calculate loss               
                loss = loss_fn.fetch_loss(cfg, outputs, targets)

            # Accumulate loss
            loss = loss / cfg['TRAIN']['N_ACCUMULATE']
                
            # Backpropagation
            scaler.scale(loss).backward()
        
            if (step + 1) % cfg['TRAIN']['N_ACCUMULATE'] == 0:

                scaler.step(optimizer)
                scaler.update()

                # zero the parameter gradients
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

        else:

            # When device = "cpu"
            # do not use automatic mixed precision

            # Generate predictions
            outputs = model(images)

            # Calculate loss
            loss = loss_fn.fetch_loss(cfg, outputs, targets)

            # Accumulate loss
            loss = loss / cfg['TRAIN']['N_ACCUMULATE']
                
            # Backpropagation
            loss.backward()
        
            if (step + 1) % cfg['TRAIN']['N_ACCUMULATE'] == 0:
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size

        # Progress bar details
        bar.set_postfix(
            Epoch=epoch, 
            Train_Loss=epoch_loss,
            LR=optimizer.param_groups[0]['lr'],
            )

    gc.collect()

    outputs = torch.round(torch.sigmoid(outputs))
    outputs = torch.squeeze(outputs)

    targets = (targets.detach().cpu().numpy()).tolist()
    outputs = (outputs.detach().cpu().numpy()).tolist()
    
    epoch_accuracy = accuracy_score(targets, outputs)

    return epoch_loss, epoch_accuracy

def valid_one_epoch(cfg, model, optimizer, dataloader, device, epoch, weights):
    """
    Function to validate on one epoch on the data

    Args:
        cfg (dict): Configuration File 
        model: Model definition
        optimizer: Optimizer for the network
        dataloader: PyTorch DataLoader
        device: GPU or CPU device for training
        epoch (int): Number of epochs
        weights (torch tensor): Class Weight values

    Returns:
        epoch_loss (float): Loss for the epoch
        epoch_accuracy(float): Percentage of correct model predictions
        final_targets (list): List of Targets
        final_outputs (list): List of Model Predictions
    """
    model.eval()

    final_targets = []
    final_outputs = []
    
    dataset_size = 0
    running_loss = 0.0
    val_correct = 0
    size_sampler = len(dataloader.sampler)
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:   

        # Get data
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        # Generate predictions
        outputs = model(images)

        # Calculate loss 
        # targets = targets.unsqueeze(1)
        loss = loss_fn.fetch_loss(cfg, outputs, targets)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        # Progress bar details
        bar.set_postfix(
            Epoch=epoch, 
            Valid_Loss=epoch_loss,
            LR=optimizer.param_groups[0]['lr']
            )   

        # Move targets and outputs to cpu 
        outputs = torch.round(torch.sigmoid(outputs))
        outputs = torch.squeeze(outputs)

        targets = (targets.detach().cpu().numpy()).tolist()
        outputs = (outputs.detach().cpu().numpy()).tolist()
        
        final_targets.extend(targets)
        final_outputs.extend(outputs)
    
    gc.collect()

    # Accuracy score for the epoch
    epoch_accuracy = accuracy_score(final_targets, final_outputs)
    
    return epoch_loss, epoch_accuracy, final_targets, final_outputs