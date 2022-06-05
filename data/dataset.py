import cv2 
import torch
import numpy as np
from torch.utils.data import Dataset

class CancerDataset(Dataset):
    def __init__(self, df, image_dir, base_path, mode, transforms=None):
        self.image_ids = df['image_id'].unique()
        self.df = df
        self.image_dir = image_dir
        self.targets = df['target'].values
        self.base_path = base_path
        self.mode = mode
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        # Get image_id
        image_id = self.image_ids[index]

        image_dir = self.image_dir + "/" + self.mode

        # Get image_path and load image in cv2 format
        image_path = self.base_path + f'{image_dir}/{image_id}'
        image = cv2.imread(
            image_path, 
            )
        try:
            image = cv2.cvtColor(
                image, 
                cv2.COLOR_BGR2RGB
            )

        except:
            print(image_path)

        target = self.targets[index]
        
        if self.transforms:
            image = self.transforms(image=image)["image"]

        return {
            'image': image,
            'target': target
        }