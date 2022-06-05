from torch.utils.data import DataLoader

import data.dataset as dataset
import data.augmentations as augmentations 

def prepare_loaders(cfg, df_train, df_valid, base_path):
    """
    Function to obtain the training and validation dataloaders 
    for given fold.

    Args:
        cfg (dict): Configuration File
        df_train (pandas dataframe): Training DataFrame
        df_valid (pandas dataframe): Validation DataFrame

    Returns:
        train_loader (DataLoader): Train DataLoader
        valid_loader (DataLaoder): Validation DataLoader
        valid_ids (list): List of images ids for validation
    """

    
    # Obtain dataset
    train_dataset = dataset.CancerDataset(
        df_train, 
        cfg['DATA']['IMG_DIR'],
        base_path,
        mode = "train", 
        transforms=augmentations.get_transforms(cfg, "train")
        )

    valid_dataset = dataset.CancerDataset(
        df_valid, 
        cfg['DATA']['IMG_DIR'],
        base_path,
        mode = "valid",
        transforms=augmentations.get_transforms(cfg, "valid")
        )

    # Get dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["TRAIN"]['TRAIN_BATCH_SIZE'], 
        num_workers=0, 
        shuffle=True, 
        pin_memory=True, 
        drop_last=True
        )

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=cfg["TRAIN"]['VALID_BATCH_SIZE'], 
        num_workers=0, 
        shuffle=False, 
        pin_memory=True
        )

    valid_ids = df_valid['image_id']
    
    return train_loader, valid_loader, valid_ids

def prepare_test_loader(cfg, df_test, base_path):
    """
    Function to obtain the test dataloader and image ids.

    Args:
        cfg (dict): Configuration File
        df_test (pandas dataframe): Test DataFrame

    Returns:
        test_loader (DataLoader): test DataLoader
        test_ids (list): List of test images ids
    """

    test_dataset = dataset.CancerDataset(
                df_test, 
                cfg['DATA']['IMG_DIR'],
                base_path,
                mode = "test",
                transforms=augmentations.get_transforms(cfg, "valid")
                )

    test_loader = DataLoader(
                test_dataset, 
                batch_size=cfg["TRAIN"]['VALID_BATCH_SIZE'], 
                num_workers=0, 
                shuffle=False, 
                pin_memory=True
                )

    test_ids = df_test['image_id']

    return test_loader, test_ids
