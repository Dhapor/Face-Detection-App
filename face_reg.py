import cv2
import av
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# UI Styling
st.markdown("""
    <style>
        .header {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }
        .subheader {
            font-size: 20px;
            font-style: italic;
            text-align: center;
            margin-bottom: 20px;
        }
        .description {
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# UI Content
st.markdown("<div class='header'>Live Face, Palm & Gesture Recognition</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Built by datapsalm using Viola-Jones & MediaPipe</div>", unsafe_allow_html=True)
st.markdown("<div class='description'>Detects faces and hand gestures in real-time!</div>", unsafe_allow_html=True)

# Finger gesture mapping
def get_gesture(fingers):
    gestures = {
        (0, 0, 0, 0, 0): "Fist 👊",
        (0, 1, 0, 0, 0): "One ☝️",
        (0, 1, 1, 0, 0): "Peace ✌️",
        (1, 1, 1, 0, 0): "Three 🤟",
        (1, 1, 1, 1, 1): "Open Palm ✋",
        (1, 0, 0, 0, 0): "Thumbs Up 👍",
        (1, 0, 0, 0, 1): "Call Me 🤙",
        (0, 0, 0, 0, 1): "Pinky 🖕",
    }
    return gestures.get(tuple(fingers), f"{sum(fingers)} Fingers")

# Determine which fingers are up
def detect_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [2, 6, 10, 14, 18]
    fingers = []

    # Thumb (different logic)
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[pip_ids[0]].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers
    for tip, pip in zip(tips_ids[1:], pip_ids[1:]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Video processor
class FacePalmGestureDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, "Datapsalm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Hand detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = img.shape
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Draw box
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Get finger states and gesture
                fingers = detect_fingers(hand_landmarks)
                gesture = get_gesture(fingers)

                # Display gesture
                cv2.putText(img, gesture, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return img

# Launch the streamer
webrtc_streamer(key="face-gesture", video_processor_factory=FacePalmGestureDetector)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 18px; color: #888;'>Powered by Streamlit and MediaPipe | datapsalm 2025</div>", unsafe_allow_html=True)
