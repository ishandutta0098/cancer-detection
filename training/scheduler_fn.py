from torch.optim import lr_scheduler

def fetch_scheduler(cfg, optimizer):
    """
    Function to fetch the scheduler for training.

    Args:
        cfg (dict): Configuration File
        optimizer: Network optimizer

    Returns:
        scheduler: The Learning Rate Scheduler
    """
    
    if cfg['TRAIN']['SCHEDULER'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['TRAIN']['T_MAX'], 
            eta_min=cfg['TRAIN']['MIN_LR']
            )

    elif cfg['TRAIN']['SCHEDULER'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg['TRAIN']['T_0'], 
            eta_min=cfg['TRAIN']['MIN_LR']
            )

    elif cfg['TRAIN']['SCHEDULER'] == None:
        return None
        
    return scheduler