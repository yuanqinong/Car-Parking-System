import streamlit as st
import cv2
import easyocr
from ultralytics import YOLO
import helper 
import time
import tempfile

    
# Page title
st.set_page_config(page_title='Car Parking System', page_icon='🖲')
st.title('Car Parking System 🖲')

# Sidebar
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.7, 0.01)

# File upload
source = st.sidebar.radio("Source", ["Upload Image", "Upload Video", "Webcam"])
result_img=[]
car_plate=None
output_log =None

if source == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        result_img,car_plate,output_log,detected_img = helper.process_image(uploaded_file,conf_threshold,iou_threshold)
            # Display the annotated image and detection log
        st.subheader("Detection Result")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(detected_img, caption="Detected Image", use_column_width=True)
        with col2:
            if len(result_img)>0:
                st.image(result_img, caption="Detected Car License Plate", use_column_width=True)      
        with col3:
             if len(result_img)>0:
                st.text_area("Detection Log", value=output_log, height=300)
             else:
                 st.text_area("Detection Log", value="No car plate in image...", height=300)


elif source == "Upload Video":
    car_model,license_plate_detector,plate_reader= helper.load_model()
    uploaded_file = st.sidebar.file_uploader("Choose a video", type=["mp4", "avi"])
    if uploaded_file is not None:
    # Create a temporary file for the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Create a video writer for the output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
        # Process the video frames
        stframe = st.empty()
        col1, col2, col3 = st.columns(3)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Perform object detection on the frame
            license_plate_results = license_plate_detector(frame,conf=conf_threshold,iou=iou_threshold, stream=True)
            if license_plate_results: 
                for license_plate_result in license_plate_results:
                    license_boxes = license_plate_result.boxes.xyxy.to('cpu').numpy().astype(int)
                    # Annotate the frame with bounding boxes and labels
                    annotated_frame = license_plate_result.plot()
                    for box in license_boxes:
                        x_min, y_min, x_max, y_max = box
                        plate_crop = frame[y_min:y_max, x_min:x_max]
                        car_plate = helper.perform_ocr(plate_crop)
                        output_log = helper.process_car_plate(car_plate)
                    # Write the annotated frame to the output video
                    out.write(annotated_frame)
                with col1:
                    # Display the annotated frame in Streamlit
                    stframe.image(annotated_frame, channels="BGR", use_column_width=True)
            with col2:
                if len(plate_crop)>0:
                    st.image(plate_crop, caption="Detected Car License Plate", use_column_width=True)    
            with col3:
                if len(plate_crop)>0:
                    st.text_area("Detection Log", value=output_log, height=300)
                else:
                    st.text_area("Detection Log", value="No car plate in image...", height=300)
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

elif source == "Webcam":
    car_model,license_plate_detector,plate_reader= helper.load_model()
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        # Perform object detection on the frame
        license_plate_results = license_plate_detector(frame,conf=conf_threshold,iou=iou_threshold, stream=True)
        if license_plate_results:  # Check if license_plate_results is not empty

            for license_plate_result in license_plate_results:
                license_boxes = license_plate_result.boxes.xyxy.to('cpu').numpy().astype(int)
                for box in license_boxes:
                    x_min, y_min, x_max, y_max = box
                    plate_crop = frame[y_min:y_max, x_min:x_max]
                    car_plate = helper.perform_ocr(plate_crop)
        # Annotate the frame with bounding boxes and labels
        annotated_frame = license_plate_results[0].plot()
        # Display the annotated frame in Streamlit
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)
        time.sleep(5)

