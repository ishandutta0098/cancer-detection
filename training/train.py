import gc 
import time 
import copy 
import torch
import numpy as np
import wandb
import training.engine as engine 
from collections import defaultdict
from tqdm.auto import tqdm

import data.dataloader as dataloader
import utilities.utils as utils

from torch.cuda import amp 

def run_training(args, cfg, model, optimizer, scheduler, weights, device, num_epochs, df_train, df_valid, pred_csv, base_path):
    """
    Function to run the training and validation on a fold of data

    Args:
        args: Argparse Arguments 
        cfg (dict): Configuration file
        model (PyTorch Model): Model Class
        optimizer: Optimizer for the network
        scheduler: Learning Rate Scheduler
        weights (torch tensor): Class Weight values
        device (torch.device): GPU or CPU
        num_epochs (int): Number of Epochs
        df_train (pandas dataframe): Training DataFrame
        df_valid (pandas dataframe): Validation DataFrame
        pred_csv (str): Path to save the validation predictions
        base_path (str): Base path for the system
        
    """
    
    start = time.time()
    history = defaultdict(list)

    best_acc = 0

    # Get dataloaders
    train_loader, valid_loader, valid_ids = dataloader.prepare_loaders(
                                                            cfg, 
                                                            df_train, 
                                                            df_valid,
                                                            base_path
                                                        )

    wandb.watch(model)

    # Use amp scaler if model is running 
    # on a cuda enabled device
    if device == torch.device('cuda'):

        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

        # Scaler for Automatic Mixed Precision
        scaler = amp.GradScaler()

    else:

        print('\nTraining on CPU')
        scaler = None
    
    # Run training and validation for given epochs
    for epoch in range(1, num_epochs + 1): 
        print("\n---------------------------------------------------------")
        print(f"########## Epoch: {epoch}/{cfg['TRAIN']['EPOCHS']} ##########")
        print("---------------------------------------------------------")
        print()

        gc.collect()

        # Training
        train_epoch_loss, train_acc = engine.train_one_epoch(
                                        cfg,
                                        model, 
                                        optimizer, 
                                        scheduler, 
                                        dataloader=train_loader, 
                                        device=device, 
                                        epoch=epoch,
                                        weights=weights,
                                        scaler=scaler
                                        )
        
        # Validation
        val_epoch_loss, val_acc, targets, outputs = engine.valid_one_epoch(
                                        cfg,
                                        model, 
                                        optimizer, 
                                        valid_loader, 
                                        device=device, 
                                        epoch=epoch,
                                        weights=weights
                                        )

        val_acc = utils.format_decimal_places(val_acc * 100)
        train_epoch_loss = utils.format_decimal_places(train_epoch_loss)
        train_acc = utils.format_decimal_places(train_acc * 100)
        val_epoch_loss = utils.format_decimal_places(val_epoch_loss)


        # Save model based on validation accuracy
        if val_acc > best_acc:
            
            print(f'>> Validation Accuracy Improved - Val Acc: Old: {best_acc} | New: {val_acc}')
            best_model = copy.deepcopy(model)
            best_acc = val_acc

            best_model_path = base_path + cfg['MODEL']['MODEL_PATH'] + "/" + cfg['MODEL']['RUN_NAME'] + "_acc_" + str(best_acc) + ".bin"

        

        # Log metrics
        wandb.log(
            {
            'epoch': epoch,
            'train_epoch_loss': train_epoch_loss,
            'train_acc': train_acc,
            'val_epoch_loss': val_epoch_loss,
            'val_acc': val_acc,
            'best_val_acc': best_acc
            }
        )

        # Store the training history
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)

        print(f"> Epochs: {epoch}/{num_epochs} - Train Loss: {train_epoch_loss} - Train Acc: {train_acc} - Val Loss: {val_epoch_loss} - Val Acc: {val_acc}")
        print()

    print(f'>> Saving Best Model with Val Acc: {best_acc}')
    print()

    torch.save(
            best_model.state_dict(), 
            best_model_path
        )
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

    training_time = (time_elapsed/60)
    wandb.log(
        {
        'training_time(mins)': training_time
        }
    )

    # Save validation predictions
    utils.save_preds(
        valid_ids, 
        targets, 
        outputs, 
        pred_file_path = pred_csv,
    )

    return best_model