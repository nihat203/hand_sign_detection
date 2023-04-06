from torchmetrics.classification import MulticlassF1Score
from torchmetrics.functional import accuracy
import torch

from defaults import device

def metrics(preds, target):
    metr = MulticlassF1Score(num_classes=29).to(device)

    return metr(preds, target)

def acc(preds, target):
    acc = accuracy(preds, target, num_classes=29, task="multiclass").to(device)

    return acc