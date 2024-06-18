import streamlit as st
import cv2
import easyocr
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer
import helper 

# Page title
st.set_page_config(page_title='Car Parking System', page_icon='🖲')
st.title('Car Parking System 🖲')

# Sidebar
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.01)

# File upload
source = st.sidebar.radio("Source", ["Upload Image", "Upload Video", "Webcam"])
if source == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        result_img,car_plate,output_log = helper.process_image(uploaded_file,conf_threshold,iou_threshold)
            # Display the annotated image and detection log
        st.subheader("Detection Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(uploaded_file, caption="Detected Car", use_column_width=True)
        with col2:
            st.image(result_img, caption="Detected Car License Plate", use_column_width=True)
        with col3:
            st.text_area("Detection Log", value=output_log, height=300)
elif source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi"])
else:
    webrtc_streamer(key="Webcam", sendback_audio=False)
# Display the uploaded image
