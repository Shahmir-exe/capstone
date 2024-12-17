import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np

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
            if(len(landmarks) != 126):
                print(location)
            hand_sequences.append(landmarks)

        # Display the video frame (optional)
        cv2.imshow('MediaPipe Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return hand_sequences

y_train = [] # A numpy array of shape (num_samples, num_classes)
dataset = ["please", "drink", "computer", "before", "hot"]
one_video_folder = dataset[0]
one_videos = [os.path.join(one_video_folder, file) for file in os.listdir(one_video_folder) if file.endswith('.mp4')]

two_video_folder = dataset[1]
two_videos = [os.path.join(two_video_folder, file) for file in os.listdir(two_video_folder) if file.endswith('.mp4')]

three_video_folder = dataset[2]
three_videos = [os.path.join(three_video_folder, file) for file in os.listdir(three_video_folder) if file.endswith('.mp4')]

four_video_folder = dataset[3]
four_videos = [os.path.join(four_video_folder, file) for file in os.listdir(four_video_folder) if file.endswith('.mp4')]

five_video_folder = dataset[4]
five_videos = [os.path.join(five_video_folder, file) for file in os.listdir(five_video_folder) if file.endswith('.mp4')]

data = []

for video in one_videos:
    video_data = play_video(video)
    data.append(video_data)
    y_train.append([1,0,0,0,0])

for video in two_videos:
    video_data = play_video(video)
    data.append(video_data)
    y_train.append([0,1,0,0,0])

for video in three_videos:
    video_data = play_video(video)
    data.append(video_data)
    y_train.append([0,0,1,0,0])

for video in four_videos:
    video_data = play_video(video)
    data.append(video_data)
    y_train.append([0,0,0,1,0])

for video in five_videos:
    video_data = play_video(video)
    data.append(video_data)
    y_train.append([0,0,0,0,1])

max_frames = 0
#testing_video = data[0]
for coordinates in data:
    if(len(coordinates) > max_frames):
        max_frames = len(coordinates)
        #testing_video = coordinates

filler_data = []
for _ in range(0,126):
    filler_data.append(0)

for coordinates in data:
    for _ in range(0,max_frames-len(coordinates)):
        coordinates.append(filler_data)

data = np.array(data)
print(data.shape)


##
##
##
##
## Everything before this just prepares the input data
##
##
##
## Everything under this makes and trains the AI model
##
##
##
##

# Pad the sequences
X_train = data
y_train = np.array(y_train)

num_samples = X_train.shape[0]
# Generate a random permutation of indices
permutation = np.random.permutation(num_samples)

# Shuffle both arrays using the same permutation
X_train = X_train[permutation]
y_train = y_train[permutation]

# X_train size is (8, 72, 126)
# Y_train size is (8, 5)

# Hyperparameters
batch_size = len(data)  # Number of videos (fixed during training)
time_steps = max_frames  # Frames per video during training
input_dim = 126  # Features per frame
lstm_units = 64  # LSTM units
num_classes = 5  # Number of output classes

# Define the stateful LSTM model
inputs = tf.keras.Input(batch_shape=(batch_size, time_steps, input_dim))
x = tf.keras.layers.LSTM(lstm_units, stateful=True, return_sequences=True)(inputs)
x = tf.keras.layers.LSTM(lstm_units, stateful=True)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Build the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=batch_size)

#
#
#
#
# Prepare the model for real-time prediction

# Define the input shape (batch_size=1, time_steps=1, input_dim=126)
input_layer = Input(batch_shape=(1, 1, input_dim))  # Batch size = 1, Time steps = 1
lstm_1 = LSTM(lstm_units, stateful=True, return_sequences=True)(input_layer)
lstm_2 = LSTM(lstm_units, stateful=True, return_sequences=True)(lstm_1)
output_layer = Dense(num_classes, activation='softmax')(lstm_2)

inference_model = Model(inputs=input_layer, outputs=output_layer)

inference_model.summary()

# Transfer weights from the trained model
inference_model.set_weights(model.get_weights())

for layer in inference_model.layers:
    if isinstance(layer, LSTM) and layer.stateful:
        layer.reset_states()

# Save the model in the HDF5 format
inference_model.save('shuffled_model.h5')

# # Simulate real-time prediction
# for frame in range(73):  # Simulating 73 frames in a live video
#     current_frame = np.array(testing_video[frame]).reshape(1, 1, 126)
#     prediction = inference_model.predict(current_frame)
#     print(f"Frame {frame + 1}: {prediction}")
