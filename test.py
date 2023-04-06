import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.functional import accuracy

from dataloader import loader
from models.model import Net
from defaults import *

def metrics(preds, target):
    metr = MulticlassF1Score(num_classes=29).to(device)

    return metr(preds, target)

def acc(preds, target):
    acc = accuracy(preds, target, num_classes=29, task="multiclass").to(device)

    return acc

def val(model, data_val, loss_function, epoch):

    f1_score = 0
    accuracy = 0
    data_iterator = enumerate(data_val)
    with torch.no_grad():

        model.eval()
        tq = tqdm(total=len(data_val))
        tq.set_description('Validation:')
        
        total_loss = 0

        for _, batch in data_iterator:
            image, label = batch
            pred = model(image.to(device))
            
            loss = loss_function(pred.to(device), label.to(device))

            f1_score += metrics(pred, label.to(device))
            accuracy += acc(pred, label.to(device))
            total_loss += loss.item()
            tq.update(1)
    
    tq.close()

    print("\n\nF1 score: ", str(round((f1_score.item()*100)/len(data_val),2))+'%')
    print("Accuracy: ", str(round((accuracy.item()*100)/len(data_val),2))+'%\n')

    return None



if __name__ ==  '__main__':

	testloader=loader(mode = "test")

	net = Net().to(device)
	net.load_state_dict(torch.load(model_path))

	loss = nn.CrossEntropyLoss()

	val(net, testloader, loss, num_epoch)