from collections import defaultdict
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from mysql.connector import connect, Error
from datetime import datetime
from pathlib import Path

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
video_path = 0
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
last_processed_time = 0  # Variable to store the last time a frame was processed
stationary_objects_masks = {}  # Dictionary to hold the masks of stationary objects

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

        print(f"Frame shape: {frame.shape}")

        new_frame_time = time.time()
        if new_frame_time - last_processed_time >= 0.1:
            last_processed_time = new_frame_time  # Update the last processed time

            # Run YOLOv8 segmentation on the frame
            results = model.track(frame, persist=True)

            # Check if masks are present in the results
            if hasattr(results[0], 'masks'):
                print("Masks are available in the results.")
                print("Masks object type:", type(results[0].masks))
                try:
                    masks_result = results[0].masks  # This is the Masks object from Ultralytics

                    num_masks = len(masks_result.xy)  # Assuming this is the correct way to get the number of masks
                    resized_masks = np.zeros((num_masks, frame.shape[0], frame.shape[1]), dtype=np.uint8)

                    for i in range(num_masks):
                        # Process each individual mask
                        mask = masks_result.xy[i]  # This might be a list of pixel coordinates for the mask

                        # Create an empty mask
                        mask_img = np.zeros((masks_result.height, masks_result.width), dtype=np.uint8)

                        # Fill the mask based on the coordinates
                        for segment in mask:
                            cv2.fillPoly(mask_img, np.array([segment], dtype=np.int32), 255)

                        # Resize the mask
                        resized_mask = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]))
                        resized_masks[i] = resized_mask

                except Exception as e:
                    print("An error occurred while processing masks:")
                    print(e)
                    continue  # Skip the rest of the loop if there's an issue with masks

                if len(resized_masks.shape) == 3 and (resized_masks.shape[1] != frame.shape[0] or resized_masks.shape[2] != frame.shape[1]):
                    raise ValueError("Resized masks array shape does not match frame dimensions")

                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                annotated_frame = results[0].plot()

                # Iterate over each detected object
                for i in range(num_objects):
                    mask = masks[i]  # Get the mask for the current object
                    box = boxes[i]
                    x, y, w, h = box
                    x_value, y_value, w_value, h_value = float(x), float(y), float(w), float(h)
                    track_id = track_ids[i]
                    class_id = class_ids[i]
                    class_name = class_names[class_id]
                    class_type = 1 if class_id in moving_object_ids else 0
                    
                    # Use the mask to generate location
                    if class_id in moving_object_ids:
                        moving_mask = mask
                        location = generate_location_with_masks(moving_mask, stationary_objects_masks, class_names)

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

                    elif class_id in stationary_object_ids:
                        stationary_objects_masks[class_id] = mask  # Store or update the mask

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
                continue  # Skip the rest of the loop if masks are not available

    except Exception as e:
        print(f"An error occurred during video processing: {e}")
        break  # Add break here to exit loop on error

# Release resources
cap.release()
cv2.destroyAllWindows()
if cursor and db:
    cursor.close()
    db.close()
