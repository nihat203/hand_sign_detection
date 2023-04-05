from torchmetrics.classification import MulticlassF1Score
import torch

from device import device

def metrics(preds, target):
    metr = MulticlassF1Score(num_classes=29).to(device)

    return metr(preds, target)