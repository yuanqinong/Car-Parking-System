# Car Park System

This is a car park management system built using YOLOv8 object detection model and EasyOCR for license plate recognition. The system supports image, video, and real-time webcam input for detecting vehicles and extracting their license plate numbers. The detected license plates and their corresponding check-in and check-out times are stored in a SQLite database, and the parking fee is calculated and displayed to the customer in a log screen. 

## Features

- License plate recognition using EasyOCR
- Support for image, video, and real-time webcam input
- Storage of license plate numbers and check-in/check-out times in SQLite database
- Calculation and display of parking fees

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/car-park-system.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Run the main script:

```
streamlit run app.py
```

2. Choose the input mode (image, video, or webcam) from the provided options.

3. The system will detect vehicles in the input, extract their license plate numbers using EasyOCR, and record the check-in/check-out times in the database.

4. The parking fee will be calculated based on the duration of stay and displayed in the log screen.

## Future Improvements

- Implement computer vision technology to monitor the remaining car park lots and display the availability.
- Introduce a detection zone where license plate recognition will only be triggered after a vehicle enters the designated area, preventing false detections.
