import os
import yaml
import random 

import torch 

import numpy as np
import pandas as pd
import wandb

from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

def load_config(config_file, train=True):
    """
    Function to load a yaml configuration file.

    Args:
        config_file (str): Name of the configuration file

    Returns:
        config (dict): yaml config data
    """
    if train == True:
        config_path = './configuration/' + config_file

    else:
        config_path = config_file

    with open (config_path) as file:
        config = yaml.safe_load(file)

    return config

def set_seed(SEED = 42):
    """
    Set the seed for the entire project

    Args:
        seed (int, optional): Seed value to be set. Defaults to 42.
    """
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
set_seed()

def save_preds(id, target, pred, pred_file_path):
    """
    Function to save the model predictions.

    Args:
        id (list): List of image ids
        target (list): List of targets
        pred (list): List of model predictions
        pred_file_path (str): Path for the prediction file
    """

    df = pd.DataFrame()
    df['image_id'] = id
    df['target'] = target 
    df['pred'] = pred

    df.to_csv(pred_file_path, index = False)

def format_decimal_places(input, places = 3):

    if places == 1:
        return float("{:.1f}".format(input))

    elif places == 2:
        return float("{:.2f}".format(input))

    elif places == 3:
        return float("{:.2f}".format(input))