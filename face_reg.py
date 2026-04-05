import cv2
import av
import os
import requests
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration

# --- Constants ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

_HAND_MODEL_PATH = "hand_landmarker.task"
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# MediaPipe hand skeleton connections (hardcoded — mp.solutions removed in 0.10+)
_HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
])


@st.cache_resource(show_spinner="Downloading hand landmark model (~9 MB)...")
def _load_hand_model_path() -> str:
    if not os.path.exists(_HAND_MODEL_PATH):
        response = requests.get(_HAND_MODEL_URL, stream=True)
        response.raise_for_status()
        with open(_HAND_MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return _HAND_MODEL_PATH


hand_model_path = _load_hand_model_path()

# --- Page config ---
st.set_page_config(
    page_title="Face & Gesture Detector",
    page_icon="🖐️👤",
    layout="centered",
)

st.markdown(
    """
    <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f7f9fc; }
    .title { font-size: 3rem; font-weight: 700; text-align: center; color: #0366d6; margin-bottom: 0; }
    .subtitle { font-size: 1.25rem; text-align: center; font-style: italic; color: #444; margin-top: 0; margin-bottom: 1rem; }
    .description { font-size: 1rem; text-align: center; color: #666; margin-bottom: 2rem; }
    .footer { text-align: center; font-size: 0.85rem; color: #999; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="title">Live Face & Gesture Recognition 🖐️👤</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by MediaPipe HandLandmarker & OpenCV</p>', unsafe_allow_html=True)
    st.markdown('<p class="description">Detect faces and hand gestures in real-time, right from your browser!</p>', unsafe_allow_html=True)


# --- Gesture helpers ---

def get_gesture(fingers: list) -> str:
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


def detect_fingers(landmarks: list) -> list:
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids  = [2, 6, 10, 14, 18]
    fingers  = []
    fingers.append(1 if landmarks[tips_ids[0]].x < landmarks[pip_ids[0]].x else 0)
    for tip, pip in zip(tips_ids[1:], pip_ids[1:]):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    return fingers


# --- Video processor ---

class FacePalmGestureDetector(VideoProcessorBase):
    def __init__(self):
        # Face detection via OpenCV Haar cascade (bundled with cv2, no download needed)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Hand detection via MediaPipe Tasks API HandLandmarker
        base_opts = mp_python.BaseOptions(model_asset_path=hand_model_path)
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # ── Face detection ──────────────────────────────────────────────
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, fw, fh) in faces:
            cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 255, 255), 2)
            cv2.putText(img, "Face", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # ── Hand landmark detection ──────────────────────────────────────
        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results  = self._hand_landmarker.detect(mp_image)

        for hand_landmarks in results.hand_landmarks:
            x_coords = [int(lm.x * w) for lm in hand_landmarks]
            y_coords = [int(lm.y * h) for lm in hand_landmarks]

            # Draw skeleton
            for start, end in _HAND_CONNECTIONS:
                cv2.line(img, (x_coords[start], y_coords[start]),
                              (x_coords[end],   y_coords[end]), (0, 255, 0), 2)

            # Draw landmark dots
            for cx, cy in zip(x_coords, y_coords):
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

            # Bounding box + gesture label
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            gesture = get_gesture(detect_fingers(hand_landmarks))
            cv2.putText(img, gesture, (x_min, max(0, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Webcam streamer ---
webrtc_streamer(
    key="face-palm-gesture",
    video_processor_factory=FacePalmGestureDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown(
    """
    <div class="footer">
        &copy; 2025 datapsalm &nbsp;|&nbsp; Built with Streamlit & MediaPipe &nbsp;|&nbsp;
        <a href="https://github.com/Dhapor" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
