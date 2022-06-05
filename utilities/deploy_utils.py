import os

import numpy as np
import pandas as pd 

from io import BytesIO
import cv2
from PIL import Image

import cv2 
import torch 
import json
import face_features.data.augmentations as augmentations 

import face_features.utilities.model_utils as model_utils
import face_features.utilities.blazeface_utils as blazeface_utils
import re
import base64

def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image

def preprocess_image(img, cfg):
    """
    Function to apply transformations, add batch dimension and convert image
    array to tensor.

    Args:
        img (numpy array): Image to be processed
        transforms (optional): Transformations to apply to the image. Defaults to augmentations.get_transforms(cfg, "valid").

    Returns:
        img (torch tensor): Transformed Image Tensor
    """
    transforms=augmentations.get_transforms(cfg, "valid")
    img = transforms(image=img)["image"]
    img = torch.unsqueeze(torch.tensor(img), 0)

    return img

def load_model(cfg, DEVICE):
    """
    Function to load the classification model
    with its weights

    Args:
        cfg (dict): Model configuration
        DEVICE (torch.device): Model device

    Returns:
        model: Model with loaded weights
    """
    model = model_utils.setup_pretrained_model(
        cfg, 
        cfg['DATA']['BASE_PATH'], 
        DEVICE,
        pretrained=False
        )

    model.to(DEVICE)

    return model
    
def get_prediction(data, model, device):
    """
    Function to obtain the model predictions

    Args:
        data (torch tensor): Tensor of Image
        model (nn.Module): Model with loaded weights
        device (torch.device): Device for data

    Returns:
        pred (int): Index of argument with highest value
    """

    model.eval()

    # Move data to device
    images = data.to(device, dtype=torch.float)

    # Obtain predictions
    outputs = model(images)

    # Obtain index of clas with highest prediction
    pred = torch.argmax(outputs)

    return pred

def run_inference(cfg, data):
    """
    Function to do the inference on the image

    Args:
        data (torch tensor): Tensor of image

    Returns:
        output (int): Index of output class
    """

    # Get device
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load model
    model = load_model(cfg, DEVICE) 

    # Obtain predictions
    output = get_prediction(data, model, DEVICE)
    
    return output

def predict(cfg, file_path, classes, crop = False):
    """
    Function to load the image, crop, preprocess and run inference

    Args:
        cfg (dict): Configuration File
        file_path (str): Image File Path
        classes (list): List of classes for the feature

    Returns:
        (dict): prediction
    """
    if crop == True:
        # Original Image array
        img_arr = cv2.imread(file_path)

        # Path to save cropped image
        crop_image_path = "/tmp/crop.jpg"

        # Load blazeface model for image cropping
        front_net = blazeface_utils.get_model()

        # Perform image cropping and save it
        blazeface_utils.data_crop_pipeline(
                    image_path=img_arr,
                    crop_image_path = crop_image_path,
                    front_net = front_net,
                )

        # Load cropped image as numpy array
        data = np.array(Image.open(crop_image_path))

    else:
        # Load cropped image as numpy array
        data = np.array(Image.open(file_path))

    # Preprocess cropped image
    data = preprocess_image(data, cfg)

    # Obtain output on cropped image
    outputs = run_inference(cfg, data)

    # Obtain prediction class
    prediction = classes[outputs]
        
    return prediction