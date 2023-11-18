import pytorchyolo

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pytorchyolo import detect

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set up audio output
import pygame
pygame.mixer.init()

# Define a function to play a sound when a person is detected
def play_sound():
    sound = pygame.mixer.Sound('person.wav')
    sound.play()

# Define a function to process video frames
def process_frame(frame):
    # Convert frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect objects in the frame
    results = pytorchyolo.detect

    # Check if any people were detected
    if len(results.xyxy[0]) > 0:
        play_sound()

# Capture video frames from a webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Process the frame
    process_frame(frame)

    # Display the frame
    cv2.imshow('frame', frame)

    # Check if the `q` key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
