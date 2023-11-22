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

# Dictionary to hold the bounding boxes of stationary objects
stationary_objects_bboxes = {}

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Open the video file or webcam
video_path = 0
try:
    cap = cv2.VideoCapture(video_path)
except Exception as e:
    print(f"Error opening video source: {e}")
    exit(1)

# Store the track history and for FPS calculation
track_history = defaultdict(lambda: [])
prev_frame_time = 0

# Connect to the database
try:
    db = connect(host="localhost", user="root", password="", database="assistant")
    cursor = db.cursor()
except Error as e:
    print(f"Error connecting to MySQL: {e}")
    exit(1)


def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # Compute the area of intersection
    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]

    # Compute the intersection over union
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou

def generate_location(moving_bbox, stationary_bboxes, class_names):
    mx, my, mw, mh = moving_bbox  # Moving object bounding box

    for stationary_id, stationary_bbox in stationary_bboxes.items():
        sx, sy, sw, sh = stationary_bbox  # Stationary object bounding box

        # Calculate IoU
        iou = calculate_iou((mx, my, mw, mh), (sx, sy, sw, sh))

        # Check if IoU is greater than 50%
        if iou > 0.5:
            return f"on {class_names[stationary_id]}"
        # Check if the bottom of the moving object is near the top of the stationary object
        if my + mh > sy and my + mh < sy + sh and mx < sx + sw and mx + mw > sx:
            return f"on {class_names[stationary_id]}"

        # Check if the moving object is to the left of the stationary object
        if mx + mw < sx and my + mh > sy and my < sy + sh:
            return f"left of {class_names[stationary_id]}"

        # Check if the moving object is to the right of the stationary object
        if mx > sx + sw and my + mh > sy and my < sy + sh:
            return f"right of {class_names[stationary_id]}"

        # Check if the moving object is in front of the stationary object
        if my < sy and mx + mw > sx and mx < sx + sw:
            return f"in front of {class_names[stationary_id]}"

    return "unknown"



# Loop through the video frames
while cap.isOpened():
    try:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        new_frame_time = time.time()

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and getattr(results[0].boxes, 'id', None) is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            annotated_frame = results[0].plot()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id in moving_object_ids or class_id in stationary_object_ids:
                    x, y, w, h = box
                    x_value = float(x)
                    y_value = float(y)
                    w_value = float(w)
                    h_value = float(h)
                    class_name = class_names[class_id]
                    class_type = 1 if class_id in moving_object_ids else 0
                    # Check if it's a stationary object and update its bbox
                    if class_id in stationary_object_ids:
                        stationary_objects_bboxes[class_id] = (x_value, y_value, w_value, h_value)
                    
                    # Process only moving objects
                    if class_id in moving_object_ids: # or class_id in stationary_object_ids:
                        # class_name = class_names[class_id]
                        # class_type = 1  # Moving object

                        # Initialize location string
                        # Use the function to generate the location
                        location = generate_location((x_value, y_value, w_value, h_value), stationary_objects_bboxes, class_names)

                        try:
                            cursor.execute("SELECT * FROM detections WHERE tracker_id = %s", (track_id,))
                            existing_record = cursor.fetchone()

                            if existing_record:
                                cursor.execute(
                                    "UPDATE detections SET class_id = %s, class_name = %s, class_type = %s, x = %s, y = %s, width = %s, height = %s, location = %s, timestamp = %s WHERE tracker_id = %s",
                                    (class_id, class_name, class_type, x_value, y_value, w_value, h_value, location, datetime.now(), track_id)
                                )
                            else:
                                cursor.execute(
                                    "INSERT INTO detections (tracker_id, class_id, class_name, class_type, x, y, width, height, location, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                    (track_id, class_id, class_name, class_type, x_value, y_value, w_value, h_value, location, datetime.now())
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

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        print(f"An error occurred during video processing: {e}")

cap.release()
cv2.destroyAllWindows()
if cursor and db:
    cursor.close()
    db.close()