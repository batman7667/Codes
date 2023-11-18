import cv2
import numpy as np
import time
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16

# Define a function to load the VGG16 model
def load_model():
    model = VGG16(weights='imagenet', include_top=False)
    return model

# Define a function to process a video frame
def process_frame(frame):
    # Convert the frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to the input size of the VGG16 model
    frame = cv2.resize(frame, (224, 224))

    # Convert the frame to an array of pixels
    frame = img_to_array(frame)

    # Expand the dimensions of the frame to fit the model's input format
    frame = np.expand_dims(frame, axis=0)

    # Preprocess the frame by normalizing the pixel values
    frame /= 255.0

    # Make a prediction using the VGG16 model
    prediction = model.predict(frame)

    # Get the highest probability class label
    predicted_class = np.argmax(prediction[0])

    # Load the corresponding audio file for the detected class label
    sound_file = f'sounds/{predicted_class}.mp3'

    # Play the audio file
    if sound_file:
        cv2.mixer.init()
        sound = cv2.mixer.Sound(sound_file)
        sound.play()

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

    # Check if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
