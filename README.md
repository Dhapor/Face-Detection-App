# Face, Palm & Gesture Recognition 🖐️👤

A real-time **face detection** and **hand gesture recognition** app built with OpenCV, MediaPipe Tasks API, and Streamlit — deployable on Streamlit Cloud.

## Features

- **Face Detection** — OpenCV Haar cascade (Viola-Jones), runs entirely on-device
- **Hand Gesture Recognition** — MediaPipe `HandLandmarker` (Tasks API) with 21-point skeleton
- **Gesture Labels** — Fist, Peace, Thumbs Up, Open Palm, Call Me, and more
- **Real-time Webcam Stream** — powered by `streamlit-webrtc` with WebRTC
- **Streamlit Cloud Ready** — model downloaded automatically on first run

## Gestures Supported

| Gesture | Label |
|---------|-------|
| All fingers closed | Fist 👊 |
| Index finger up | One ☝️ |
| Index + middle up | Peace ✌️ |
| Thumb + index + middle | Three 🤟 |
| All fingers open | Open Palm ✋ |
| Thumb only | Thumbs Up 👍 |
| Thumb + pinky | Call Me 🤙 |

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/Dhapor/Face-Detection-App.git
cd Face-Detection-App
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run face_reg.py
```

The app opens at `http://localhost:8501`. On first run it downloads the hand landmark model (~9 MB) — this is automatic and only happens once.

**4. Use the app**
- Click **START** on the webcam widget
- Allow camera access in your browser
- Point your face and hands at the camera
- Detected faces get a yellow bounding box
- Detected hands show the skeleton + gesture label

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `face_reg.py`
5. Click **Deploy**

Streamlit Cloud will install `packages.txt` (system deps) and `requirements.txt` (Python deps) automatically.

## Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `streamlit-webrtc` | Real-time webcam stream via WebRTC |
| `mediapipe` | HandLandmarker Tasks API |
| `opencv-python-headless` | Face detection + image processing |
| `av` | Video frame handling |
| `requests` | Model file download |
