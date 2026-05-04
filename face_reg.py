import cv2
import av
import os
import requests
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration

# ── Constants ──────────────────────────────────────────────────────────────────
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

_HAND_MODEL_PATH = "hand_landmarker.task"
_HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_HAND_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),
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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face & Gesture Detector",
    page_icon="🖐️",
    layout="wide",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }

  .hero-title { font-size: 2.8rem; font-weight: 700; color: #0366d6; text-align: center; margin-bottom: 0; }
  .hero-sub   { font-size: 1.05rem; color: #888; text-align: center; margin-top: 4px; }
  .section-header {
    font-size: 1.4rem; font-weight: 600; color: #0366d6;
    border-bottom: 2px solid #0366d6; padding-bottom: 6px; margin-top: 1.8rem;
  }
  .info-card {
    background: #f0f6ff; border-left: 4px solid #0366d6;
    border-radius: 6px; padding: 14px 18px; margin-bottom: 10px;
  }
  .info-card h4 { color: #0366d6; margin: 0 0 4px 0; font-size: 1rem; }
  .info-card p  { color: #555; margin: 0; font-size: 0.9rem; }
  .stat-card {
    background: #0366d6; color: white; border-radius: 10px;
    padding: 18px; text-align: center;
  }
  .stat-card .value { font-size: 1.6rem; font-weight: 700; }
  .stat-card .label { font-size: 0.85rem; opacity: 0.85; margin-top: 2px; }
  .gesture-chip {
    display: inline-block; background: #e8f0fe; color: #0366d6;
    border: 1px solid #0366d6; border-radius: 20px;
    padding: 3px 12px; margin: 3px; font-size: 0.85rem;
  }
  .disclaimer {
    background: #fff3cd; border-left: 4px solid #ffc107;
    border-radius: 6px; padding: 12px 16px; margin-top: 1rem;
    font-size: 0.88rem; color: #555;
  }
  hr.divider { border: none; border-top: 1px solid #e0e0e0; margin: 1.5rem 0; }
  .footer { text-align: center; font-size: 0.85rem; color: #999; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)


# ── Gesture detection helpers ──────────────────────────────────────────────────
GESTURES = {
    (0, 0, 0, 0, 0): "Fist 👊",
    (0, 1, 0, 0, 0): "One ☝️",
    (0, 1, 1, 0, 0): "Peace ✌️",
    (1, 1, 1, 0, 0): "Three 🤟",
    (1, 1, 1, 1, 1): "Open Palm ✋",
    (1, 0, 0, 0, 0): "Thumbs Up 👍",
    (1, 0, 0, 0, 1): "Call Me 🤙",
    (0, 0, 0, 0, 1): "Pinky 🖕",
}

def get_gesture(fingers: list) -> str:
    return GESTURES.get(tuple(fingers), f"{sum(fingers)} Fingers")

def detect_fingers(landmarks: list) -> list:
    tips = [4, 8, 12, 16, 20]
    pips  = [2, 6, 10, 14, 18]
    fingers = [1 if landmarks[tips[0]].x < landmarks[pips[0]].x else 0]
    for tip, pip in zip(tips[1:], pips[1:]):
        fingers.append(1 if landmarks[tip].y < landmarks[pip].y else 0)
    return fingers


# ── Video processor ────────────────────────────────────────────────────────────
class FacePalmGestureDetector(VideoProcessorBase):
    def __init__(self):
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        base_opts = mp_python.BaseOptions(model_asset_path=hand_model_path)
        hand_opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts, num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img  = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        h, w, _ = img.shape

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, fw, fh) in faces:
            cv2.rectangle(img, (x, y), (x + fw, y + fh), (0, 255, 255), 2)
            cv2.putText(img, "Face", (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results  = self._hand_landmarker.detect(mp_image)

        for hand_landmarks in results.hand_landmarks:
            xs = [int(lm.x * w) for lm in hand_landmarks]
            ys = [int(lm.y * h) for lm in hand_landmarks]

            for start, end in _HAND_CONNECTIONS:
                cv2.line(img, (xs[start], ys[start]), (xs[end], ys[end]), (0, 255, 0), 2)
            for cx, cy in zip(xs, ys):
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            gesture = get_gesture(detect_fingers(hand_landmarks))
            cv2.putText(img, gesture, (x_min, max(0, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">🖐️ Live Face & Gesture Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Real-time face detection and hand gesture recognition &nbsp;|&nbsp; Built by Datapsalm</p>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

tab_home, tab_detect, tab_about = st.tabs(["🏠 Overview", "📷 Live Detection", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    col_left, col_right = st.columns([1.4, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-header">What Does This App Do?</p>', unsafe_allow_html=True)
        st.markdown("""
This app runs **live** in your browser using your webcam to detect:

- **Faces** — draws a bounding box around every face in the frame using OpenCV's Haar Cascade classifier
- **Hands** — maps 21 landmark points on each hand using Google's MediaPipe HandLandmarker
- **Gestures** — recognizes 8 hand gestures in real-time from finger positions

Switch to the **Live Detection** tab to start your webcam and see it in action.
        """)

        st.markdown('<br>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="stat-card"><div class="value">Real-time</div><div class="label">Detection Speed</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="stat-card"><div class="value">2 Hands</div><div class="label">Max Simultaneous</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stat-card"><div class="value">8</div><div class="label">Gestures Recognized</div></div>', unsafe_allow_html=True)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">Recognized Gestures</p>', unsafe_allow_html=True)
        chips = " ".join(f'<span class="gesture-chip">{g}</span>' for g in GESTURES.values())
        st.markdown(chips, unsafe_allow_html=True)

    with col_right:
        st.markdown('<p class="section-header">Use Cases</p>', unsafe_allow_html=True)
        for title, desc in [
            ("🔒 Access Control", "Detect presence and count people in a camera feed"),
            ("🎮 Gesture Gaming", "Control games or interfaces with hand signs"),
            ("♿ Accessibility", "Enable gesture-based input for users with limited mobility"),
            ("📸 Photography", "Auto-trigger a camera when a specific gesture is detected"),
            ("🎓 Education", "Teach computer vision and pose estimation concepts"),
        ]:
            st.markdown(f"""
            <div class="info-card">
              <h4>{title}</h4>
              <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">🔒 <strong>Privacy note:</strong> All processing runs entirely in your browser session. No video frames are stored or transmitted to any server.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">How It Works</p>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, (step, label) in zip(cols, [
        ("1️⃣", "Your webcam stream is captured frame-by-frame via WebRTC"),
        ("2️⃣", "OpenCV Haar Cascade scans each frame for face-shaped patterns"),
        ("3️⃣", "MediaPipe HandLandmarker locates 21 key points on each detected hand"),
        ("4️⃣", "Finger tip positions are compared to joint positions to classify the gesture"),
    ]):
        with col:
            st.info(f"**{step}** {label}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.markdown('<p class="section-header">Live Webcam Detection</p>', unsafe_allow_html=True)
    st.markdown("""
- Allow browser access to your camera when prompted
- Point your face and hands at the camera
- Face boxes appear in **yellow**, hand skeleton in **green**, gesture labels in **blue**
    """)

    st.markdown('<br>', unsafe_allow_html=True)

    col_stream, col_legend = st.columns([2, 1])

    with col_stream:
        webrtc_streamer(
            key="face-palm-gesture",
            video_processor_factory=FacePalmGestureDetector,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_legend:
        st.markdown("**Detection key**")
        st.markdown("🟡 Yellow box — detected face")
        st.markdown("🟢 Green lines — hand skeleton")
        st.markdown("🔴 Red dots — landmark points")
        st.markdown("🔵 Blue box + label — gesture")
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("**Gesture reference**")
        for gesture in GESTURES.values():
            st.markdown(f"- {gesture}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<p class="section-header">Technologies Used</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        for tech, desc in [
            ("OpenCV Haar Cascade", "Classical computer vision algorithm that scans for face-shaped pixel patterns. Fast and works without a GPU."),
            ("MediaPipe HandLandmarker", "Google's ML model that detects 21 3D landmarks on each hand — fingertips, knuckles, and wrist."),
            ("streamlit-webrtc", "Enables real-time video streaming inside a Streamlit app using the WebRTC protocol."),
        ]:
            st.markdown(f"""
            <div class="info-card">
              <h4>{tech}</h4>
              <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
**Face detection vs. face recognition:**

| | Face Detection | Face Recognition |
|---|---|---|
| What it does | Locates faces in an image | Identifies *who* the face belongs to |
| This app | ✅ Yes | ❌ No |
| Privacy risk | Low | High |

**This app only detects faces — it does not identify or store any biometric data.**
        """)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown(f'<div class="footer">&copy; 2025 Datapsalm &nbsp;|&nbsp; Built with Streamlit, MediaPipe & OpenCV &nbsp;|&nbsp; <a href="https://github.com/Dhapor" target="_blank">GitHub</a></div>', unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Face & Gesture Detector")
    st.markdown("Real-time detection powered by MediaPipe and OpenCV.")
    st.markdown("---")
    st.markdown("**Face detection:** OpenCV Haar Cascade")
    st.markdown("**Hand tracking:** MediaPipe HandLandmarker")
    st.markdown("**Gestures:** 8 recognized")
    st.markdown("**Webcam:** Browser WebRTC")
    st.markdown("---")
    st.markdown("🔒 No video is stored or transmitted")
    st.caption("Built by Datapsalm")
