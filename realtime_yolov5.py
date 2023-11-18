import cv2
import torch
import numpy as np
import pygame
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F

def detect_objects(frame, model):
    # Preprocess the frame
    img = F.to_tensor(frame)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = img.permute(2, 0, 1).unsqueeze(0)
    img = img.to('cuda')

    # Perform object detection
    outputs = model(img)
    probas = torch.nn.functional.softmax(outputs['pred_logits'], -1)
    keep = (probas.max(-1).values > 0.2).nonzero().flatten()
    probas = probas[keep].tolist()
    bboxes = outputs['pred_boxes'][keep].cpu().numpy()

    # Map the bounding boxes back to the original frame
    width, height, _ = frame.shape
    bboxes = (bboxes * np.array([width, height, width, height])).round().astype(int)

    return probas, bboxes

def play_audio(label):
    if label == 'person':
        sound = pygame.mixer.Sound('C:\Users\ASUS\Documents\OptiCap\Audio\person.wav')
    elif label == 'car':
        sound = pygame.mixer.Sound('C:\Users\ASUS\Documents\OptiCap\Audio\person.wav')
    else:
        return

    sound.play()

def main():
    # Load the yolov5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.cuda()

    # Initialize pygame for audio playback
    pygame.mixer.init()

    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform object detection and obtain bounding boxes
        probas, bboxes = detect_objects(frame, model)

        # Play the corresponding audio for each detected object
        for i, (prob, bbox) in enumerate(zip(probas, bboxes)):
            max_prob_index = prob.index(max(prob))
            label = model.names[max_prob_index]
            play_audio(label)

        # Display the resulting frame
        cv2.imshow('Realtime YOLOv5', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()