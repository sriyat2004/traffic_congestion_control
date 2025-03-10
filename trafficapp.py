import streamlit as st
import pandas as pd
import joblib
from pothole_detection import detect_potholes
import matplotlib.pyplot as plt
import cv2

# Load the trained traffic congestion model
model = joblib.load('traffic_model.pkl')

# Title and description
st.title('Traffic Congestion & Pothole Detection App')
st.markdown('This app helps you to check traffic congestion and pothole presence on different roads.')

# Traffic prediction section
st.header('Traffic Congestion Prediction')
time_of_day = st.slider('Time of Day', 0, 23, 8)
day_of_week = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
traffic_volume = st.slider('Traffic Volume (Vehicles per hour)', 50, 1000, 300)

day_of_week_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

# Predict traffic congestion
input_data = pd.DataFrame([[time_of_day, day_of_week_mapping[day_of_week], traffic_volume]],
                          columns=['Time of Day', 'Day of Week', 'Traffic Volume'])
congestion_level = model.predict(input_data)[0]

st.write(f'The predicted traffic congestion level is: {congestion_level}')

# Pothole detection section
st.header('Pothole Detection')
uploaded_image = st.file_uploader("Upload a road image", type=["jpg", "png"])

if uploaded_image is not None:
    # Read and display the uploaded image
    img = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", caption="Uploaded Road Image", use_column_width=True)

    # Detect potholes
    potholes = detect_potholes(uploaded_image)
    st.write(f'Number of potholes detected: {potholes}')

    # Display the result on the image
    if potholes > 0:
        st.success("Potholes detected!")
    else:
        st.success("No potholes detected.")