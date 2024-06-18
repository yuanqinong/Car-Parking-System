import cv2
import easyocr
from ultralytics import YOLO
import filetype
import sqlite3
from datetime import datetime
import numpy as np
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

def detect_cars(img,conf_threshold,iou_threshold):
    car_results = car_model(img ,conf=conf_threshold,iou=iou_threshold, agnostic_nms=True)
    return car_results

def detect_license_plates(car_crop,conf_threshold,iou_threshold):
    license_plate_results = license_plate_detector(car_crop,conf=conf_threshold,iou=iou_threshold)
    return license_plate_results

def perform_ocr(plate_crop):
    license_plate_text = reader.readtext(plate_crop, detail=0)
    license_plate_text = license_plate_text[0].replace(" ", "")
    #print(f"Detected license plate: {license_plate_text}")
    return license_plate_text

def calculate_parking_fee(parking_duration):
    # Calculate total duration in minutes
    total_minutes = parking_duration.total_seconds() / 60
    
    # Check if parking duration is less than 5 minutes
    if total_minutes < 5:
        return 0
    
    # Get the total hours rounded up
    total_hours = total_minutes / 60
    hours_rounded_up = int(total_hours) + (1 if total_hours % 1 > 0 else 0)
    
    # Calculate the parking fee
    parking_fee = hours_rounded_up * 2
    
    return parking_fee

def process_image(src,conf_threshold,iou_threshold):
    #car_image = cv2.imread(src)
    # Convert the file to OpenCV image
    file_bytes = np.asarray(bytearray(src.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    car_results = detect_cars(opencv_image,conf_threshold,iou_threshold)

    if car_results:
        for car_result in car_results:
            boxes = car_result.boxes.xyxy.to('cpu').numpy().astype(int)

            for box in boxes:
                x_min, y_min, x_max, y_max = box
                car_crop = opencv_image[y_min:y_max, x_min:x_max]

                license_plate_results = detect_license_plates(car_crop,conf_threshold,iou_threshold)
                if license_plate_results:  # Check if license_plate_results is not empty

                    for license_plate_result in license_plate_results:
                        license_boxes = license_plate_result.boxes.xyxy.to('cpu').numpy().astype(int)
                        for box in license_boxes:
                            x_min, y_min, x_max, y_max = box
                            plate_crop = car_crop[y_min:y_max, x_min:x_max]
                            car_plate = perform_ocr(plate_crop)
                            output_log = process_car_plate(car_plate)
    return plate_crop,car_plate,output_log

def process_video(src):
    cap = cv2.VideoCapture(src)
    ret = True

    while ret:
        ret, frame = cap.read()
        if ret:
            car_results = detect_cars(frame)

            if car_results:
                for car_result in car_results:
                    boxes = car_result.boxes.xyxy.to('cpu').numpy().astype(int)

                    for box in boxes:
                        x_min, y_min, x_max, y_max = box
                        car_crop = frame[y_min:y_max, x_min:x_max]

                        license_plate_results = detect_license_plates(car_crop)
                        if license_plate_results:
                            for license_plate_result in license_plate_results:
                                license_boxes = license_plate_result.boxes.xyxy.to('cpu').numpy().astype(int)
                                for box in license_boxes:
                                    x_min, y_min, x_max, y_max = box
                                    plate_crop = car_crop[y_min:y_max, x_min:x_max]
                                    car_plate = perform_ocr(plate_crop)
                                    process_car_plate(car_plate)

    cap.release()

def insert_parking_log(car_plate):
    # Connect to the SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect('car_park.db')
    c = conn.cursor()
    # Create the table
    c.execute('''CREATE TABLE IF NOT EXISTS parking_logs
             (id INTEGER PRIMARY KEY AUTOINCREMENT, car_plate TEXT, check_in_time TEXT)''')
    check_in_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO parking_logs (car_plate, check_in_time) VALUES (?, ?)",
              (car_plate, check_in_time))
    conn.commit()
    conn.close()

# Function to process a car plate
def process_car_plate(car_plate):
    # Connect to the SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect('car_park.db')
    c = conn.cursor()
    # Create the parking_logs table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS parking_logs
                (id INTEGER PRIMARY KEY AUTOINCREMENT, car_plate TEXT, check_in_time TEXT)''')

    # Create the car_park_record table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS car_park_record
                (id INTEGER PRIMARY KEY AUTOINCREMENT, car_plate TEXT, check_in_time TEXT, check_out_time TEXT, parking_fee REAL)''')
    # Check if the car plate exists in parking_logs
    c.execute("SELECT id, check_in_time FROM parking_logs WHERE car_plate = ?", (car_plate,))
    result = c.fetchone()

    if result:
        # Car plate exists, calculate parking fee and move the record to car_park_record
        log_id, check_in_time = result
        check_in_time = datetime.strptime(check_in_time, '%Y-%m-%d %H:%M:%S')
        check_out_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        parking_duration = datetime.now() - check_in_time
        parking_fee = calculate_parking_fee(parking_duration)

        # Insert the record into car_park_record
        c.execute("INSERT INTO car_park_record (car_plate, check_in_time, check_out_time, parking_fee) VALUES (?, ?, ?, ?)",
                  (car_plate, check_in_time, check_out_time, parking_fee))

        # Remove the record from parking_logs
        c.execute("DELETE FROM parking_logs WHERE id = ?", (log_id,))
        conn.commit()

        print(f"Car Plate: {car_plate}")
        print(f"Check-in Time: {check_in_time}")
        print(f"Check-out Time: {check_out_time}")
        print(f"Parking Fee: RM{parking_fee:.2f}")

        output = (
            f"Car Plate: {car_plate}\n"
            f"Check-in Time: {check_in_time}\n"
            f"Check-out Time: {check_out_time}\n"
            f"Parking Fee: RM{parking_fee:.2f}"
        )
        
    else:
        check_in_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Car plate doesn't exist, insert a new record into parking_logs
        insert_parking_log(car_plate)
        print(f"New car plate {car_plate} added to the parking logs.")
        output = (
            f"Car Plate: {car_plate}\n"
            f"Check-in Time: {check_in_time}\n"
            f"Check-out Time: - \n"
            f"Parking Fee: -"
        )
    conn.close()
    return output
   

def view_db():
    # Connect to the SQLite database
    conn = sqlite3.connect('car_park.db')
    c = conn.cursor()

    # View the records in the parking_logs table
    print("parking_logs table:")
    c.execute("SELECT * FROM parking_logs")
    rows = c.fetchall()
    for row in rows:
        print(row)

    # View the records in the car_park_record table
    print("\ncar_park_record table:")
    c.execute("SELECT * FROM car_park_record")
    rows = c.fetchall()
    for row in rows:
        print(row)

    conn.close()

def detect(src):
    file_type = check_file_type(src)
    if file_type == "image":
        process_image(src)
    elif file_type == "video":
        process_video(src)

if __name__ == "__main__":
    detect("./src/sample/images/JWD6338.jpeg")
    view_db()