from collections import defaultdict
from itertools import combinations
from pathlib import Path
import torch
import cv2
import numpy as np
import time
from ultralytics import YOLO
from mysql.connector import connect, Error
from datetime import datetime

# class names present in the model
class_names = (
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
)

all_class_ids = tuple(range(80))


# Define moving object
mobile_object_ids=(15,16,24,25,26,27,28,32,39,41,42,43,44,45,63,64,65,66,67,73,74,75,76,77,78,79)
# Define stationary object
stationary_object_ids=(13,56,57,58,59,60,62,68,69,72)

# Dictionary to hold the bounding boxes of stationary objects
stationary_objects_bboxes = {}

# Load the YOLOv8 model
model = YOLO('../yolov8x-seg.pt')

# Open the video file or webcam
#video_path = 0
video_path = '../Sequence01.mp4'
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

def check_overlap(moving_bbox, stationary_bbox):
    mx, my, mw, mh = moving_bbox
    sx, sy, sw, sh = stationary_bbox

    horizontal_overlap = (mx < sx + sw) and (mx + mw > sx)
    vertical_overlap = (my < sy + sh) and (my + mh > sy)

    return horizontal_overlap and vertical_overlap

def check_relative_position(moving_bbox, stationary_bbox):
    mx, my, mw, mh = moving_bbox
    sx, sy, sw, sh = stationary_bbox
    moving_center = (mx + mw / 2, my + mh / 2)
    stationary_center = (sx + sw / 2, sy + sh / 2)

    if moving_center[1] > stationary_center[1]:  # Moving object is below the stationary object
        return "below"
    elif moving_center[1] < stationary_center[1]:  # Moving object is above the stationary object
        return "above"
    elif moving_center[0] < stationary_center[0]:  # Moving object is to the left of the stationary object
        return "left of"
    elif moving_center[0] < stationary_center[0]:  # Moving object is to the right of the stationary object
        return "right of"
    else:  
        return "near"

def generate_location(moving_bbox, stationary_bboxes, class_names):
    for stationary_id, stationary_bbox in stationary_bboxes.items():
        if check_overlap(moving_bbox, stationary_bbox):
            relative_position = check_relative_position(moving_bbox, stationary_bbox)
            return f"{relative_position} {class_names[stationary_id]}"

    return "unknown"



class_masks = {}
mask_areas = {}
bounding_boxes = {}




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
            #boxes = results[0].boxes.xywh.cpu()
            masks = results[0].masks.data
            boxes = results[0].boxes.data
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            for i in class_ids:
                seg_class = class_names[i]
                obj_indices = torch.where(clss == i)
                obj_masks = masks[obj_indices]
                obj_mask = torch.any(obj_masks, dim=0).int() * 255
                class_masks[seg_class] = obj_mask
                mask_areas[seg_class] = torch.sum(obj_mask).item()
                #cv2.imwrite(f'./test_output/{seg_class}s.jpg', obj_mask.cpu().numpy())

                # Extract bounding boxes
                obj_bbox = boxes[obj_indices].squeeze(0)
                if obj_bbox.ndim == 1:  # In case squeezing leads to a 1D tensor
                    bounding_boxes[seg_class] = obj_bbox.unsqueeze(0)  # Add an extra dimension
                else:
                    bounding_boxes[seg_class] = obj_bbox






            annotated_frame = results[0].plot()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id in mobile_object_ids or class_id in stationary_object_ids:
                    x, y, w, h = box
                    x_value = float(x)
                    y_value = float(y)
                    w_value = float(w)
                    h_value = float(h)
                    class_name = class_names[class_id]
                    class_type = 1 if class_id in mobile_object_ids else 0
                    # Check if it's a stationary object and update its bbox
                    if class_id in stationary_object_ids:
                        stationary_objects_bboxes[class_id] = (x_value, y_value, w_value, h_value)
                    
                    # Process only moving objects
                    if class_id in mobile_object_ids: # or class_id in stationary_object_ids:
                        # class_name = class_names[class_id]
                        # class_type = 1  # Moving object

                        # Initialize location string
                        # Use the function to generate the location
                        location = generate_location((x_value, y_value, w_value, h_value), stationary_objects_bboxes, class_names)
                        print(location)

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