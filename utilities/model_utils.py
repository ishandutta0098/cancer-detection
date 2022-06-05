import glob
import torch

import training.optimizer_fn as optimizer_fn
import training.scheduler_fn as scheduler_fn
import models.baseline_model as baseline_model

def make_model(cfg, pretrained=True): 
    """
    Function to get the model, optimizer, scheduler

    Args:
        cfg: Configuration File

    Returns:
        model: Model Class
        optimizer: Optimizer for the network
        scheduler: Learning rate scheduler
    """

    if 'resnet' in cfg['MODEL']['MODEL_NAME']:
        model = baseline_model.ResNetModel(cfg, pretrained)

    elif 'resnext' in cfg['MODEL']['MODEL_NAME']:
        model = baseline_model.ResNextModel(cfg, pretrained)

    elif 'vgg' in cfg['MODEL']['MODEL_NAME']:
        model = baseline_model.VGGModel(cfg, pretrained)

    elif 'vit' in cfg['MODEL']['MODEL_NAME']:
        model = baseline_model.ViTModel(cfg, pretrained)

    elif 'efficientnet' in cfg['MODEL']['MODEL_NAME']:
        model = baseline_model.EfficientNetModel(cfg, pretrained)

    optimizer = optimizer_fn.get_optimizer(cfg, model)
    scheduler = scheduler_fn.fetch_scheduler(cfg, optimizer)

    return model, optimizer, scheduler

def setup_pretrained_model(cfg, base_path, device, pretrained=True):
    """
    Function to load a pretrained model.

    Args:
        cfg: Model Configuration file
        base_path (str): Base Path of the system
        device (torch.device): GPU or CPU

    Returns:
        model: Model with loaded weights
    """

    # Get model path
    model_path = [
        model_path for model_path in glob.glob(base_path + cfg['MODEL']['MODEL_PATH'] + "/*") \
        if cfg['MODEL']['RUN_NAME'] in model_path
    ]
    
    # model_path = [
    #     model_path for model_path in glob.glob(base_path + cfg['MODEL']['MODEL_PATH'] + "/*")]

    # Create model
    model, _, _ = make_model(cfg, pretrained)

    # Load weights
    model.load_state_dict(torch.load(model_path[0], map_location=device))

    return model