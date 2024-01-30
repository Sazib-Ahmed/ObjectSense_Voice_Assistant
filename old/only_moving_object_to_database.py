from collections import defaultdict
import cv2
import numpy as np
import time
from ultralytics import YOLO
from mysql.connector import connect, Error
from datetime import datetime

# class names present in the model
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# Define moving object IDs
moving_object_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79
]

# Define the list of stationary object IDs
stationary_object_ids = [13, 56, 57, 58, 59, 60, 61, 62, 68, 69, 70, 71, 72]


# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Open the video file or webcam
video_path = 0

try:

    cap = cv2.VideoCapture(video_path)
except Exception as e:
    print(f"Error opening video source: {e}")
    exit(1)

# Store the track history
track_history = defaultdict(lambda: [])

# For FPS calculation
prev_frame_time = 0
new_frame_time = 0

# Connect to the database
try:
    db = connect(
        host="localhost",
        user="root",
        password="",
        database="assistant"
    )
    cursor = db.cursor()
except Error as e:
    print(f"Error connecting to MySQL: {e}")
    exit(1)

# Loop through the video frames
while cap.isOpened():
    try:    
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            print("Failed to grab frame")
            break

        if success:
            new_frame_time = time.time()

            # Run YOLOv8 tracking on the frame
            results = model.track(frame, persist=True)

            # Check if detections and track IDs are present
            if results[0].boxes is not None and getattr(results[0].boxes, 'id', None) is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()  # Correct attribute to access class values

                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                
                # Inside the loop where detections are processed
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box
                    if class_id in moving_object_ids:
                        x_value = float(x)
                        y_value = float(y)
                        w_value = float(w)
                        h_value = float(h)
                        class_name = class_names[class_id]

                        try:
                            # Check if the tracker_id already exists in the database
                            cursor.execute("SELECT * FROM detections WHERE tracker_id = %s", (track_id,))
                            existing_record = cursor.fetchone()

                            if existing_record:
                                # Update the existing record
                                cursor.execute(
                                    "UPDATE detections SET class_id = %s, class_name = %s, x = %s, y = %s, width = %s, height = %s, timestamp = %s WHERE tracker_id = %s",
                                    (class_id, class_name, x_value, y_value, w_value, h_value, datetime.now(), track_id)
                                )
                            else:
                                # Insert a new record
                                cursor.execute(
                                    "INSERT INTO detections (tracker_id, class_id, class_name, x, y, width, height, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                                    (track_id, class_id, class_name, x_value, y_value, w_value, h_value, datetime.now())
                                )
                            db.commit()
                        except Error as e:
                            print(f"Error interacting with database: {e}")


                # Calculate and display the FPS
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

                # Show the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    except Exception as e:
        print(f"An error occurred during video processing: {e}")
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()
