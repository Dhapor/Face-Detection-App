import cv2
import av
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Streamlit UI
st.markdown("<h1 style='color: #FFACAC'>Live Face Detection App</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='color: #F2921D'>Built by datapsalm using Viola-Jones</h6>", unsafe_allow_html=True)
st.markdown("<hr><br>", unsafe_allow_html=True)

# Load the Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define a Video Transformer
class FaceDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        return img

# Stream video from webcam
webrtc_streamer(key="face-detect", video_processor_factory=FaceDetector)
