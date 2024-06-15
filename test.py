import cv2  # type: ignore
import easyocr  # type: ignore
from ultralytics import YOLO  # type: ignore
import supervision as sv  # type: ignore
import numpy as np

# Load models
car_model = YOLO('./src/model/yolov8n.pt')
license_plate_detector = YOLO('./src/model/license_plate_detector.pt')
reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader

def detect_video(video_path):

        sv.process_video(source_path=video_path, target_path="result_short.mp4", callback=process_frame)



def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = car_model(frame, imgsz=1280)[0]
    
    detections = sv.Detections.from_ultralytics(results)

    corner_annotator = sv.BoxCornerAnnotator(thickness=4)

    #labels = [f"{car_model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
    annotated_frame  = corner_annotator.annotate(scene=frame, detections=detections)

    return annotated_frame

if __name__ == "__main__":

    detect_video("./sample_short.mp4")