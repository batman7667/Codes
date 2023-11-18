import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pyaudio

stream = pyaudio.PyAudio().open(format=pyaudio.paFloat32, channels=1, rate=22050, output=True)

class YOLOv5:
    def __init__(self, device):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='path/to/yolov5s.pt', force_reload=True)
        self.model.to(device)
        self.model.eval()

    def classify(self, image):
        results = self.model(image)
        output = []

        for *xyxy, conf, cls in results.xyxy[0]:
            output.append([*xyxy, conf.item(), cls.item()])
            stream.write(b'Audio output for detected object goes here.')

        return output
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLOv5(device)

    while True:
        image = cv2.imread('path/to/your/image.jpg')
        image = Image.fromarray(image)
        image_t = transforms.ToTensor()(image)
        image_t = image_t.to(device)
        image_t = image_t.unsqueeze(0)

        results = model.classify(image_t)

        for xyxy, conf, cls in results:
            cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
            cv2.putText(image, f'{conf:.4f}', (xyxy[0], xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('Image', image)
        cv2.waitKey(1)
