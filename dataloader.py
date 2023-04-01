import torch
import torchvision

from defaults import *
from transforms import *

def loader(mode = "train", image_path = "./datasets/train"):

	resset = torchvision.datasets.ImageFolder(root=image_path, transform=transform)

	if mode == "train":
		res = torch.utils.data.DataLoader(resset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	if mode == "test":
		res = torch.utils.data.DataLoader(resset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

	return res