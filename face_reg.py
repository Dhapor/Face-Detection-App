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

# UI Styling
st.markdown("""
    <style>
        .header {
            font-size: 40px;
            color: #FF5733;
            font-weight: bold;
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
        }
        .subheader {
            font-size: 20px;
            color: #F2921D;
            font-style: italic;
            text-align: center;
            margin-bottom: 20px;
        }
        .description {
            color: #777;
            font-size: 18px;
            text-align: center;
            margin-bottom: 30px;
        }
        .frame-container {
            text-align: center;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }
        .button {
            background-color: #FF5733;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .button:hover {
            background-color: #F2921D;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# UI Content
st.markdown("<div class='header'>Live Face & Palm Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Built by datapsalm using Viola-Jones & MediaPipe</div>", unsafe_allow_html=True)
st.markdown("<div class='description'>This app detects faces and palms in real-time using your webcam!</div>", unsafe_allow_html=True)

# Video processor class
class FacePalmDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Palm detection using MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box from landmarks
                h, w, _ = img.shape
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(img, "Palm", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return img

# Launch Streamlit WebRTC streamer
webrtc_streamer(key="face-palm", video_processor_factory=FacePalmDetector)

# Add a footer with a cool message
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 18px; color: #888;'>Powered by Streamlit and MediaPipe | datapsalm 2025</div>", unsafe_allow_html=True)

