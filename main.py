import os
import numpy as np
from re import M
import pandas as pd
import torch
import argparse
import wandb 
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

import data.dataloader as dataloader
import utilities.utils as utils
import training.train as train 
import utilities.model_utils as model_utils
import utilities.inference_utils as inference_utils

if __name__ == "__main__":

    # Login to weights and biases
    wandb.login()

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, required=True, help='cuda or cpu')
    parser.add_argument('--environ', type=str, required=True, help='colab or local or ec2')
    parser.add_argument('--check', type=bool)

    args = parser.parse_args()

    # Load Configuration File
    cfg = utils.load_config(args.config)
    
    if args.environ == 'colab':
        # mount drive at drive.mount('/drive)
        BASE_PATH = cfg['DATA']['COLAB_BASE_PATH']

    elif args.environ == 'local':
        BASE_PATH = cfg['DATA']['BASE_PATH']

    elif args.environ == 'ec2':
        BASE_PATH = cfg['DATA']['BASE_PATH']

    # Get paths
    TRAIN_PATH = BASE_PATH + cfg['DATA']['TRAIN_CSV']
    VALID_PATH = BASE_PATH + cfg['DATA']['VALID_CSV']
    TEST_PATH = BASE_PATH + cfg['DATA']['TEST_CSV']
    VAL_PRED_PATH = BASE_PATH + cfg['PREDICT']['PRED_CSV']
    TEST_PRED_PATH = BASE_PATH + cfg['PREDICT']['TEST_CSV']
    
    # Device
    DEVICE = torch.device(args.device)

    # Make model
    model, optimizer, scheduler = model_utils.make_model(cfg)
    model.to(DEVICE)

    # Load data
    df_train = pd.read_csv(TRAIN_PATH)
    df_valid = pd.read_csv(VALID_PATH)
    df_test = pd.read_csv(TEST_PATH)

    # Get Class weights for calculating Loss
    if cfg['TRAIN']['WEIGHTS'] == True:
        weights = utils.get_class_weights(df_train)

    else: 
        weights = None

    if args.check == True:
        df_train = df_train.head(100)
        df_valid = df_valid.head(100)
        df_test = df_test.head(100)

    # Train and validate model

    print("\n---------------------------------------------------------")
    print(f"########## Initialize Training ##########")
    print("---------------------------------------------------------")
    print()

    # Initialize weights and biases
    run = wandb.init(
        project=cfg['MODEL']['PROJECT_NAME'], 
        config = cfg,
        # group = cfg['MODEL']['GROUP_NAME'] 
        )

    # Set wandb run name
    wandb.run.name = cfg['MODEL']['RUN_NAME']

    best_model = train.run_training(
        args,
        cfg,
        model, 
        optimizer, 
        scheduler,
        weights = weights,
        device=DEVICE,
        num_epochs=cfg['TRAIN']['EPOCHS'],
        df_train = df_train,
        df_valid = df_valid,
        pred_csv=VAL_PRED_PATH,
        base_path = BASE_PATH 
    )

    print("\n---------------------------------------------------------")
    print(f"########## Training Completed ##########")
    print("---------------------------------------------------------")
    print()

    print("\n---------------------------------------------------------")
    print(f"########## Initialize Model Testing ##########")
    print("---------------------------------------------------------")
    print()

    test_loader, test_ids = dataloader.prepare_test_loader(
                                                        cfg, 
                                                        df_test, 
                                                        BASE_PATH
                                                    )

    outputs, targets = inference_utils.test_fn(
                                                best_model, 
                                                test_loader, 
                                                DEVICE
                                            )

    # Save test predictions
    utils.save_preds(
        test_ids, 
        targets, 
        outputs, 
        pred_file_path = TEST_PRED_PATH,
    )

    test_accuracy = accuracy_score(targets, np.round(outputs))
    test_accuracy = utils.format_decimal_places(test_accuracy * 100)

    print("\n---------------------------------------------------------")
    print(f"########## Test Accuracy: {test_accuracy} ##########")
    print("---------------------------------------------------------")
    print()

    wandb.log(
        {
            'test_accuracy_score': test_accuracy
        }
    )

    print("\n---------------------------------------------------------")
    print(f"########## Model Testing Complete ##########")
    print("---------------------------------------------------------")
    print()

    wandb.finish()