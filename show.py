import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(img):
	img = torchvision.utils.make_grid(img) / 2 + 0.5
	plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
	plt.show()