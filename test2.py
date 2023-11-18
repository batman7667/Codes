import torch
import yolo5
import pyttsx3
import cv2
import numpy as np

# Initialize YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define object detection classes
classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

# Define audio output for each object detection class
audio_outputs = {
    'person': 'Person detected.',
    'bicycle': 'Bicycle detected.',
    'car': 'Car detected.',
    'motorcycle': 'Motorcycle detected.',
    'bus': 'Bus detected.',
    'truck': 'Truck detected.'
}

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Check if webcam is open
if not cap.isOpened():
    print('Error opening webcam')
    exit()

# Define frame processing function
def process_frame(frame):
    # Convert frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame to model input size
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)


    # Convert frame to tensor
    frame = torch.from_numpy(frame).unsqueeze(0).to('cuda')

    # Run object detection on frame
    results = model(frame)

    # Extract detected objects and their bounding boxes
    detections = results.xyxy[0]
    scores = detections[:, 4].cpu().numpy()
    boxes = detections[:, 5:9].cpu().numpy()

    # Apply non-maxima suppression to filter out overlapping detections
    indexes = non_max_suppression(boxes, scores, 0.45)

    # Process filtered detections
    for i in indexes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = boxes[i]

        # Scale coordinates to frame dimensions
        x1, y1 = scale_coords(frame.shape[2:], x1, y1).round()
        x2, y2 = scale_coords(frame.shape[2:], x2, y2).round()

        # Draw bounding box on frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Get object detection class
        object_class = classes[detections[i, 0].int()]

        # Generate audio output for detected object
        audio_output = audio_outputs[object_class]

        # Speak audio output using text-to-speech engine
        engine.say(audio_output)
        engine.runAndWait()

# Main loop
while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    if ret:
        # Process frame
        process_frame(frame)

        # Display processed frame
        cv2.imshow('Object Detection with Audio Output', frame)

        # Check if 'q' key is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Shut down text-to-speech engine
engine.shutdown()
