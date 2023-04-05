import torch

from dataloader import loader
from models.model import Net
from defaults import *
from labels import *
from device import device


if __name__ ==  '__main__':

	testloader=loader(mode = "test")

	net = Net().to(device)
	net.load_state_dict(torch.load(model_path))

	correct = 0
	total = 0
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        images, labels = images.to(device), labels.to(device)
	        outputs = net(images)
	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Accuracy: %d %%' % (100 * correct / total))
