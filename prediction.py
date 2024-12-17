# Import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Input

import cv2
import mediapipe as mp
import os

# Initialize MediaPipe's hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Function to extract hand landmarks from a frame
def extract_hand_landmarks(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)
        remaining_hands = 2 - len(results.multi_hand_landmarks)
        for _ in range(0,remaining_hands):
            for _ in range(0,63):
                landmarks.append(0)
        return landmarks
    else:
        return None

def play_video(location):
    # Open the video file
    cap = cv2.VideoCapture(location)

    # Create a list to store hand landmark sequences
    hand_sequences = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        landmarks = extract_hand_landmarks(frame)

        if landmarks:
            hand_sequences.append(landmarks)

        # Display the video frame (optional)
        cv2.imshow('MediaPipe Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return hand_sequences

# Load the saved model from the HDF5 format
model = tf.keras.models.load_model('shuffled_model.h5')

for layer in model.layers:
    if isinstance(layer, LSTM) and layer.stateful:
        layer.reset_states()

data = play_video("drink/17710.mp4")

max = -1
maxIndex = -1

for frame in range(len(data)):
    current_frame = np.array(data[frame]).reshape(1, 1, 126)
    prediction = model.predict(current_frame)
    for pred in range(5):
        if prediction[0][0][pred] > max:
            max = prediction[0][0][pred]
            maxIndex = pred 

labels = ["please", "drink", "computer", "before", "hot"]
print(f"Model is {max * 100}% confident that it was {labels[maxIndex]}")
