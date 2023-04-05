import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

from models.model import Net
from labels import *
from device import *
from defaults import model_path
from transforms import *

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

net = Net().to(device)
net.load_state_dict(torch.load(model_path))

def preprocess(frame):
    frame = cv2.resize(frame, (200, 200))
    frame = transform(frame)
    frame = frame.to(device)
    frame = frame.unsqueeze(0
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    border_color = (251, 0, 0)
    border_size = 3
    frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)
    frame = cv2.flip(frame, 1)
    frame2 = net(preprocess(frame))
    _, predicted = torch.max(frame2.data, 1)
    
    cv2.putText(frame, str(classes[predicted.item()]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    frame = cv2.resize(frame, (200, 200))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
