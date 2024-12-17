import cv2
import mediapipe as mp
import os
import time

# Initialize MediaPipe's hand tracking module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Function to extract hand landmarks from a frame
def extract_hand_landmarks(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
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

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z))

        if landmarks:
            hand_sequences.append(landmarks)

        # Display the video frame (optional)
        cv2.imshow('MediaPipe Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return hand_sequences

video_folder = "please"
videos = [os.path.join(video_folder, file) for file in os.listdir(video_folder) if file.endswith('.mp4')]
data = []

for video in videos:
    video_data = play_video(video)
    data.append(video_data)

# Save the hand sequences to a file (e.g., in JSON format)
# import json

# with open('hand_sequences.json', 'w') as f:
#     json.dump(data, f)
