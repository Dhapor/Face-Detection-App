import cv2
import streamlit as st

st.markdown("<h1 style = 'color: #FFACAC'>FACE DETECTION APPLICATION</h1> ", unsafe_allow_html = True)
st.markdown("<h6 style = 'top_margin: 0rem; color: #F2921D'>Built by datapsalm</h6>", unsafe_allow_html = True)

# Add an image to the page
st.image('my.png', caption = 'FACE DETECTOR', width = 400)

# Create a line and a space underneath
st.markdown('<hr><hr><br>', unsafe_allow_html= True)

# Add instructions to the Streamlit app interface to guide the user on how to use the app.
if st.button('Read the usage Instructions below'):
    st.success('Hello User, these are the guidelines for the app usage')
    st.write('Press the camera button for our model to detect your face')
    st.write('Use the MinNeighbour slider to adjust how many neighbors each candidate rectangle should have to retain it')
    st.write('Use the Scaler slider to specify how much the image size is reduced at each image scale')

st.markdown('<br>', unsafe_allow_html= True)
# Start the face detectionimport cv2
import streamlit as st

# Set title and author information
st.markdown("<h1 style='color: #FFACAC'>FACE DETECTION APPLICATION</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='margin-top: 0rem; color: #F2921D'>Built by datapsalm</h6>", unsafe_allow_html=True)

# Add an image to the page
st.image('my.png', caption='FACE DETECTOR', width=400)

# Add instructions for using the app
if st.button('Read the usage instructions below'):
    st.success('Hello User, these are the guidelines for the app usage')
    st.write('Press the "FACE DETECT" button to detect faces using your webcam')
    st.write('Use the sliders to adjust the minimum neighbors and scale factor for face detection')

# Add sliders for adjusting parameters
min_neighbours = st.slider('Adjust Min Neighbours', 1, 10, 5)
scale_factor = st.slider('Adjust Scale Factor', 0.0, 3.0, 1.3)

if st.button('FACE DETECT'):
    # Initialize webcam
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbours, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        # Display frame with detected faces
        st.image(frame, channels="BGR", caption='Face Detection using Viola-Jones Algorithm')

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close all windows
    camera.release()
    cv2.destroyAllWindows()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')
camera = cv2.VideoCapture(0)

# set the minNeighbours abd Scale Factor buttons
min_Neighbours = st.slider('Adjust Min Neighbour', 1, 10, 5)
Scale_Factor = st.slider('Adjust Scale Factor', 0.0, 3.0, 1.3)

st.markdown('<br>', unsafe_allow_html= True)

if st.button('FACE DETECT'):
# Initialize the webcam
    while True:
        _, camera_view = camera.read()   #....................................... Initiate the camera
        gray = cv2.cvtColor(camera_view, cv2.COLOR_BGR2GRAY) #.................. Grayscale it using the cv grayscale library
    #   Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor= Scale_Factor, minNeighbors= min_Neighbours, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    #   Draw rectangles around the detected faces
        for (x, y, width, height) in faces:
            cv2.rectangle(camera_view, (x, y), (x + width, y + height), (225, 255, 0), 2)
    # Display the camera_views
        cv2.imshow('Face Detection using Viola-Jones Algorithm', camera_view)
    # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    # Release the webcam and close all windows
    camera.release()
    cv2.destroyAllWindows()

# let's assume the number of images gotten is 0
# img_counter = 0
# if k%256  == 32:
#     # the format for storing the images scrreenshotted
#     img_name = f'opencv_frame_{img_counter}'
#     # saves the image as a png file
#     cv2.imwrite(img_name, frame)
#     print('screenshot taken')
#     # the number of images automaticallly increases by 1
#     img_counter += 1
