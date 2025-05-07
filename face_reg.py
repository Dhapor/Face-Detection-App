import cv2
import streamlit as st
import numpy as np

st.markdown("<h1 style='color: #FFACAC'>FACE DETECTION APPLICATION</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='top_margin: 0rem; color: #F2921D'>Built by datapsalm</h6>", unsafe_allow_html=True)

st.image('my.png', caption='FACE DETECTOR', width=400)
st.markdown('<hr><hr><br>', unsafe_allow_html=True)

if st.button('Read the usage Instructions below'):
    st.success('Hello User, these are the guidelines for the app usage')
    st.write('Take a photo for our model to detect your face')
    st.write('Use the MinNeighbour slider to adjust how many neighbors each candidate rectangle should have to retain it')
    st.write('Use the Scaler slider to specify how much the image size is reduced at each image scale')

min_Neighbours = st.slider('Adjust Min Neighbour', 1, 10, 5)
Scale_Factor = st.slider('Adjust Scale Factor', 1.1, 3.0, 1.3)

st.markdown('<br>', unsafe_allow_html=True)

uploaded_image = st.camera_input("Take a photo")

if uploaded_image is not None:
    img_array = np.frombuffer(uploaded_image.getvalue(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=Scale_Factor,
        minNeighbors=min_Neighbours,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    st.image(image, caption=f"{len(faces)} face(s) detected", channels="BGR")
