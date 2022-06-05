import gc
import torch
from tqdm.auto import tqdm
import torch.nn as nn

def test_fn(model, dataloader, device):
        """
        Function to test model accuracy

        Args:
            model: Model definition
            dataloader: PyTorch DataLoader
            device: GPU or CPU device for training

        Returns:
            final_targets (list): List of Targets
            final_outputs (list): List of Model Predictions
        """

        print("\n---------------------------------------------------------")
        print(f"########## Starting Evaluation ##########")
        print("---------------------------------------------------------")
        print()

        model.eval()

        final_outputs = []
        final_targets = []
        
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:   

            print(step)

            # Get data
            images = data['image'].to(device, dtype=torch.float)
            targets = data['target'].to(device, dtype=torch.long)

            # Generate predictions
            outputs = model(images)
            outputs = torch.squeeze(outputs)

            outputs = torch.round(torch.sigmoid(outputs))

            # Move targets and outputs to cpu 
            outputs = (outputs.detach().cpu().numpy()).tolist()
            targets = (targets.detach().cpu().numpy()).tolist()

            final_outputs.extend(outputs)
            final_targets.extend(targets)

        gc.collect()
        
        return final_outputs, final_targets
