import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

def get_transforms(cfg, data):
    """
    Function to obtain the Training and Validation Transforms

    Args:
        cfg (dict): Configuration File
        data (str): "train" for Training data,
                    "valid " for Validation data

    Returns:
        Augmentations : Transforms to be applied
    """

    # Train Augmentations
    if data == "train":
        return A.Compose(
            [
                A.Resize(cfg['MODEL']['IMAGE_SIZE'], cfg['MODEL']['IMAGE_SIZE']),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                    ),
                ToTensorV2()
            ], 
        p=1.
    )

    # Validation Augmentations
    if data == "valid":
        return A.Compose(
            [
                A.Resize(cfg['MODEL']['IMAGE_SIZE'], cfg['MODEL']['IMAGE_SIZE']),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225], 
                    max_pixel_value=255.0, 
                    p=1.0
                    ),
                ToTensorV2()
            ], 
        p=1.
    )