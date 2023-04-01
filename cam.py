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

state_dict = torch.load(model_path, map_location=torch.device(device))

model = Net().to(device)
model.load_state_dict(state_dict)
model.eval()

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

def preprocess(frame):
    frame = cv2.resize(frame, (200, 200))
    frame = frame.transpose((2, 0, 1))
    frame = frame.astype(np.float32)
    frame = frame / 255.0
    frame = torch.from_numpy(frame)
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    frame = transform(frame)
    frame = frame.unsqueeze(0)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    border_color = (251, 0, 0)
    border_size = 2
    frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)
    frame = cv2.flip(frame, 1)
    x = preprocess(frame).to(device)
    y = model(x).to(device)
    pred = torch.argmax(y, dim=1)
    cv2.putText(frame, str(classes[pred.item()]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
