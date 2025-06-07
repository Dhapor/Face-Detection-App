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

# --- Page config and style ---
st.set_page_config(
    page_title="Face & Gesture Detector",
    page_icon="🖐️👤",
    layout="centered",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    /* General body font and background */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f7f9fc;
    }

    /* Title style */
    .title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #0366d6;
        margin-bottom: 0;
    }

    /* Subtitle style */
    .subtitle {
        font-size: 1.25rem;
        text-align: center;
        font-style: italic;
        color: #444;
        margin-top: 0;
        margin-bottom: 1rem;
    }

    /* Description style */
    .description {
        font-size: 1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }

    /* Footer style */
    .footer {
        text-align: center;
        font-size: 0.85rem;
        color: #999;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Centered header content with columns
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="title">Live Face & Gesture Recognition 🖐️👤</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Viola-Jones & MediaPipe</p>', unsafe_allow_html=True)
    st.markdown('<p class="description">Detect faces and hand gestures in real-time, right from your browser!</p>', unsafe_allow_html=True)

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

# Video processor class
class FacePalmGestureDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Flip image horizontally (mirror)
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Hand detection via MediaPipe
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

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                fingers = detect_fingers(hand_landmarks)
                gesture = get_gesture(fingers)

                cv2.putText(img, gesture, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return img

# Start webcam streamer with async processing for smoothness
webrtc_streamer(
    key="face-palm-gesture",
    video_processor_factory=FacePalmGestureDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Footer section
st.markdown(
    """
    <div class="footer">
        &copy; 2025 datapsalm &nbsp;|&nbsp; Built with Streamlit & MediaPipe &nbsp;|&nbsp; <a href="https://github.com/Dhapor" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
