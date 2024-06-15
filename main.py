import cv2
import easyocr
from ultralytics import YOLO
import filetype

# Load models
car_model = YOLO('./src/model/yolov8n.pt')
license_plate_detector = YOLO('./src/model/license_plate_detector.pt')
reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader

def check_file_type(src):
    kind = filetype.guess(src)
    mime_type = kind.mime
    file_type = mime_type.split("/")[0]
    if file_type not in ["image", "video"]:
        print('Please input video or image file only!')
        return
    else:
        return file_type

def detect_cars(img):
    car_results = car_model(img, agnostic_nms=True)
    return car_results

def detect_license_plates(car_crop):
    license_plate_results = license_plate_detector(car_crop)
    return license_plate_results

def perform_ocr(plate_crop):
    license_plate_text = reader.readtext(plate_crop, detail=0)
    print(f"Detected license plate: {license_plate_text}")

def process_image(src):
    car_image = cv2.imread(src)
    car_results = detect_cars(car_image)

    for car_result in car_results:
        boxes = car_result.boxes.xyxy.to('cpu').numpy().astype(int)

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            car_crop = car_image[y_min:y_max, x_min:x_max]

            license_plate_results = detect_license_plates(car_crop)

            for license_plate_result in license_plate_results:
                license_boxes = license_plate_result.boxes.xyxy.to('cpu').numpy().astype(int)
                for box in license_boxes:
                    x_min, y_min, x_max, y_max = box
                    plate_crop = car_crop[y_min:y_max, x_min:x_max]
                    perform_ocr(plate_crop)

def process_video(src):
    cap = cv2.VideoCapture(src)
    ret = True

    while ret:
        ret, frame = cap.read()
        if ret:
            car_results = detect_cars(frame)

            for car_result in car_results:
                boxes = car_result.boxes.xyxy.to('cpu').numpy().astype(int)

                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    car_crop = frame[y_min:y_max, x_min:x_max]

                    license_plate_results = detect_license_plates(car_crop)

                    for license_plate_result in license_plate_results:
                        license_boxes = license_plate_result.boxes.xyxy.to('cpu').numpy().astype(int)
                        for box in license_boxes:
                            x_min, y_min, x_max, y_max = box
                            plate_crop = car_crop[y_min:y_max, x_min:x_max]
                            perform_ocr(plate_crop)

    cap.release()

def detect(src):
    file_type = check_file_type(src)
    if file_type == "image":
        process_image(src)
    elif file_type == "video":
        process_video(src)

if __name__ == "__main__":
    detect("./sample_short.mp4")