import torch
import torch.nn as nn
import torch.nn.functional as F

def get_bce_with_logits_loss(outputs, labels):

        return nn.BCEWithLogitsLoss()(outputs, labels.view(-1, 1))

def fetch_loss(cfg, outputs, labels):
    if cfg['TRAIN']['CRITERION'] == 'BCEWithLogitsLoss':

        return nn.BCEWithLogitsLoss()(outputs, labels.view(-1, 1))

    elif cfg['TRAIN']['CRITERION'] == 'BCELoss':

        return nn.BCELoss()(outputs, labels.view(-1, 1))