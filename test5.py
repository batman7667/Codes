import cv2
import numpy as np
import time
from moviepy.editor import *
from t2m.mistakes import *


# YOLOv5 Real-time object detection and audio feedback

class RealTimeObjectDetection:

    def __init__(self, config_path, weights_path, class_path):
        self.config_path = config_path
        self.weights_path = weights_path
        self.class_path = class_path
        self.labels = self.load_labels(class_path)
        self.model = self.load_model(config_path, weights_path)

    def load_model(self, config_path, weights_path):
        from models.yolo import Model
        model = Model(config_path)
        model.load_weights(weights_path)
        return model

    def load_labels(self, class_path):
        with open(class_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def process_audio(self, object_name):
        mistakes = [BlankInForm(), CoffeeBreak(), WrongBuilding(), TooClose(),
                    LoudTyping(), TypingInconsistencies(), LateInSession(), Inattentive()]
        error_sound = "sounds/error.mp3"
        audio_clip = AudioFileClip(error_sound)

        for mistake in mistakes:
            if mistake.is_detected(object_name):
                print(f"Error detected: {mistake.get_description()}")
                audio_clip.play()
                time.sleep(2) # Pause for 2 seconds

    def run(self):
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter("output.avi", codec, 30.0, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t1 = time.time()
            results = self.model.predict(frame)
            t2 = time.time()
            print(f"Inference time: {t2 - t1}")

            if results.get("detections") is not None:
                for detection in results.get("detections"):
                    object_name = self.labels[int(detection[6])]
                    print(f"Detected object: {object_name}")
                    self.process_audio(object_name)

            out.write(frame)
            cv2.imshow("Real-time Object Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    config_path = "models/yolov5s.yaml"
    weights_path = "models/yolov5s.pt"
    class_path = "models/coco.names"

    detector = RealTimeObjectDetection(config_path, weights_path, class_path)
    detector.run()