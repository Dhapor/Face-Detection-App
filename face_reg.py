import cv2
import streamlit as st

st.set_page_config(page_title="Face Detector", layout="centered", initial_sidebar_state="auto")

st.markdown("<h1 style = 'color: #FFACAC'>FACE DETECTION APPLICATION</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'margin-top: 0rem; color: #F2921D'>Built by datapsalm</h6>", unsafe_allow_html = True)

st.image('my.png', caption='FACE DETECTOR', width=400)
st.markdown('<hr><hr><br>', unsafe_allow_html=True)

if st.button('Read the usage Instructions below'):
    st.success('Hello User, these are the guidelines for the app usage')
    st.write('Press the camera button for our model to detect your face')
    st.write('Use the MinNeighbour slider to adjust sensitivity')
    st.write('Use the Scaler slider to control detection scaling')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    st.error("Failed to load Haar cascade. Check the XML file is in the repo.")

min_Neighbours = st.slider('Adjust Min Neighbour', 1, 10, 5)
Scale_Factor = st.slider('Adjust Scale Factor', 1.01, 2.0, 1.3)

if st.button('FACE DETECT'):
    camera = cv2.VideoCapture(0)
    st.info("Press 'q' in the OpenCV window to quit.")
    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access the camera.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=Scale_Factor, minNeighbors=min_Neighbours)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.imshow('Face Detection - Press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
