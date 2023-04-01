import torch
import torchvision
import torch.optim as optim
import torch.nn as nn

from models.model import Net
from dataloader import loader
from defaults import *
from transforms import *    
from labels import *
from device import device


if __name__ ==  '__main__':

    trainloader=loader(mode = "train", image_path = "./datasets/train")

    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 

    net = Net().to(device)
    print("Device:", device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    end.record()
    torch.cuda.synchronize()

    print('Finished Training')
    print(start.elapsed_time(end))

    torch.save(net.state_dict(), model_path)
    print("Saved to:", model_path)