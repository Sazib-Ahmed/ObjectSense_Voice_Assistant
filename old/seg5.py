from collections import defaultdict
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from mysql.connector import connect, Error
from datetime import datetime
from itertools import combinations

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


# Load the YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')  # Ensure this model variant provides segmentation masks

# Open the video file or webcam
video_path = 0  # replace this with your video path or keep 0 for webcam
try:
    cap = cv2.VideoCapture(video_path)
except Exception as e:
    print(f"Error opening video source: {e}")
    exit(1)

# Connect to the database
try:
    db = connect(host="localhost", user="root", password="", database="assistant")
    cursor = db.cursor()
except Error as e:
    print(f"Error connecting to MySQL: {e}")
    exit(1)

# Initialize variables for tracking and FPS calculation
track_history = defaultdict(lambda: [])
prev_frame_time = 0
last_processed_time = 0
stationary_objects_masks = {}  # Dictionary to hold the masks of stationary objects

def check_mask_overlap(mask1, mask2):
    overlap = mask1 & mask2
    return torch.any(overlap)

def check_mask_overlap(moving_mask, stationary_mask):
    overlap = moving_mask & stationary_mask
    return torch.any(overlap)

def mask_center(mask):
    """Calculate the center of a mask."""
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    center_x = np.mean(x_indices)
    center_y = np.mean(y_indices)
    return center_x, center_y

def calculate_distance(center1, center2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

def generate_location_with_masks(moving_mask, stationary_masks, class_names):
    moving_center = mask_center(moving_mask)
    if moving_center is None:
        return "unknown"

    for stationary_id, stationary_mask in stationary_masks.items():
        stationary_center = mask_center(stationary_mask)
        if stationary_center is None:
            continue

        if check_mask_overlap(moving_mask, stationary_mask):
            relative_x = moving_center[0] - stationary_center[0]
            relative_y = moving_center[1] - stationary_center[1]
            distance = calculate_distance(moving_center, stationary_center)

            # Define a threshold for proximity
            proximity_threshold = 50  # This can be adjusted based on specific requirements

            if distance < proximity_threshold:
                proximity = "near"
            else:
                proximity = "far"

            if abs(relative_y) > abs(relative_x):  # More vertical movement
                if relative_y > 0:
                    return f"{proximity} below {class_names[stationary_id]}"
                else:
                    return f"{proximity} above {class_names[stationary_id]}"
            else:  # More horizontal movement
                if relative_x > 0:
                    return f"{proximity} right of {class_names[stationary_id]}"
                else:
                    return f"{proximity} left of {class_names[stationary_id]}"

    return "unknown"

while cap.isOpened():
    try:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        new_frame_time = time.time()
        if new_frame_time - last_processed_time >= 0.1:
            last_processed_time = new_frame_time

            # Run YOLOv8 segmentation on the frame
            results = model.track(frame, persist=True)

            # Check if masks are present in the results
            if hasattr(results[0], 'masks'):
                masks_result = results[0].masks

                # Process masks and bounding boxes
                if results[0].boxes:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
                    class_ids = results[0].boxes.cls.int().cpu().tolist() if results[0].boxes.cls is not None else []
                else:
                    boxes, track_ids, class_ids = [], [], []

                annotated_frame = results[0].plot()

                class_masks = {}
                for i, mask in enumerate(masks_result.xy):
                    mask_img = mask.astype(np.uint8)
                    class_id = class_ids[i]
                    class_masks[class_id] = mask_img

                    #mask = masks[i]  # Get the mask for the current object
                    box = boxes[i]
                    x, y, w, h = box
                    x_value, y_value, w_value, h_value = float(x), float(y), float(w), float(h)
                    track_id = track_ids[i]
                    class_id = class_ids[i]
                    class_name = class_names[class_id]
                    class_type = 1 if class_id in moving_object_ids else 0
                    

                # Check for overlapping masks
                for (class1, mask1), (class2, mask2) in combinations(class_masks.items(), 2):
                    if check_mask_overlap(mask1, mask2):
                        location = generate_location_with_masks(mask1, stationary_objects_masks, class_names)
                        # Database operations
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
            else:
                print("Masks not found in the results.")

    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        break

cap.release()
cv2.destroyAllWindows()
if cursor and db:
    cursor.close()
    db.close()
