import cv2
import streamlit as st
import torch
import av
import mediapipe as mp
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load YOLOv5 model (small version for speed, change to 'yolov5m', 'yolov5l' for more accuracy)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize MediaPipe Hands for palm detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)

# UI
st.markdown("<h1 style='color: #FFACAC'>Live Face & Palm Detection with YOLOv5</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='color: #F2921D'>Built by datapsalm using YOLOv5 & MediaPipe</h6>", unsafe_allow_html=True)
st.markdown("<hr><br>", unsafe_allow_html=True)

# Video processor class with YOLOv5 for face and palm detection
class FacePalmDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Perform YOLOv5 object detection
        results = model(img_rgb)
        
        # Annotate detected objects (face, palm, etc.)
        results.render()  # Render the bounding boxes on the image
        
        # Display the results (bounding boxes and labels)
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Detect palm using MediaPipe if YOLOv5 misses it
        palm_results = hands.process(img_rgb)
        if palm_results.multi_hand_landmarks:
            for hand_landmarks in palm_results.multi_hand_landmarks:
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
