import torch
import torchvision

from defaults import *
from transforms import *
from dataset_retrieval import custom_dataset

def loader(mode = "train", image_path = "datasets/", label_path = "datasets/"):

	resset = custom_dataset(mode = mode, image_path = image_path, label_path = label_path)

	if mode == "train":
		res = torch.utils.data.DataLoader(resset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	if mode == "test":
		res = torch.utils.data.DataLoader(resset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

	return res