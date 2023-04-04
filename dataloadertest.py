from defaults import *
from labels import *
from dataloader import *
from show import imshow

if __name__ ==  '__main__':
  trainloader=loader(mode = "train", image_path = "./datasets/train")

  dataiter = iter(trainloader)
  images, labels = dataiter.next()
  imshow(images)

  print(' '.join('%s' % classes[labels[j]] for j in range(batch_size)))
