import torch

def get_optimizer(cfg, model):
    """
    Function to obtain the optimizer for the network.

    Args:
        cfg (dict): Configuration File
        model: Model class

    Returns:
        Adam optimizer
    """

    return torch.optim.Adam(
        model.parameters(), 
        lr=cfg['TRAIN']['LEARNING_RATE'], 
        weight_decay=cfg['TRAIN']['WEIGHT_DECAY']
        )