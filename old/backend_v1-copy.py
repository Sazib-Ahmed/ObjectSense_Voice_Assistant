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
model = YOLO('yolov8n-seg.pt')

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


# Function to determine spatial relationship
def determine_spatial_relationship(class1, boxes1, area1, class2, boxes2, area2):
    for box1 in boxes1:
        for box2 in boxes2:
            x1, y1, x2, y2 = box1[:4]
            x3, y3, x4, y4 = box2[:4]

            horizontal_overlap = (x1 < x4 and x2 > x3)
            vertical_overlap = (y1 < y4 and y2 > y3)

            # Check if one box is completely inside the other
            is_box1_inside_box2 = x1 >= x3 and x2 <= x4 and y1 >= y3 and y2 <= y4
            is_box2_inside_box1 = x3 >= x1 and x4 <= x2 and y3 >= y1 and y4 <= y2

            if is_box1_inside_box2:
                if area1 < area2:
                    return f"{class1} is on {class2}"
                else:
                    return f"{class1} surrounds {class2}"
            elif is_box2_inside_box1:
                if area2 < area1:
                    return f"{class2} is on {class1}"
                else:
                    return f"{class2} surrounds {class1}"

            # Calculate overlap percentage
            overlap_width = min(x2, x4) - max(x1, x3)
            overlap_height = min(y2, y4) - max(y1, y3)
            overlap_area = max(overlap_width, 0) * max(overlap_height, 0)

            # Check size difference
            size_difference = abs(area1 - area2) / max(area1, area2)

            # Determine the spatial relationship based on size, position, and overlap
            if horizontal_overlap and vertical_overlap:
                if overlap_area / min(area1, area2) > 0.5:  # Significant overlap
                    if area1 > area2:
                        return f"{class1} is covering {class2}"
                    else:
                        return f"{class2} is covering {class1}"
                else:
                    if y1 < y3:
                        return f"{class1} is above {class2}"
                    else:
                        return f"{class1} is below {class2}"
            elif horizontal_overlap:
                if size_difference > 0.5:
                    return f"{class1} and {class2} are of significantly different sizes and beside each other"
                else:
                    return f"{class1} is beside {class2}"
            elif vertical_overlap:
                if size_difference > 0.5:
                    return f"{class1} and {class2} are of significantly different sizes and near each other"
                else:
                    return f"{class1} is near {class2}"

    return f"No direct spatial relationship detected between {class1} and {class2}"



# Function to check if bounding boxes overlap
def do_bounding_boxes_overlap(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x3, y3, x4, y4 = box2[:4]
    return (x1 < x4 and x2 > x3) and (y1 < y4 and y2 > y3)
# Additional utility functions like check_mask_overlap, mask_center, calculate_distance


while cap.isOpened():
    # [Existing code to read frames and run YOLO]

    if hasattr(results[0], 'masks'):
        # Process masks and bounding boxes
        class_masks = {}
        mask_areas = {}
        bounding_boxes = {}

        for result in results:
            masks = result.masks.data
            boxes = result.boxes.data
            clss = boxes[:, 5]

            for i in range(len(masks)):
                if clss[i] in seg_class_indices:
                    seg_class = seg_classes[clss[i]]
                    mask = masks[i].int() * 255
                    bbox = boxes[i]
                    area = torch.sum(mask).item()

                    class_masks[seg_class] = mask
                    mask_areas[seg_class] = area
                    bounding_boxes[seg_class] = bbox

        # Process spatial relationships
        for (class1, mask1), (class2, mask2) in combinations(class_masks.items(), 2):
            spatial_relationship = determine_spatial_relationship(
                class1, bounding_boxes[class1], mask_areas[class1],
                class2, bounding_boxes[class2], mask_areas[class2]
            )
            print(spatial_relationship)

            # Add database logging and other processing based on spatial_relationship

    # [Existing code to display frame and handle key events]

cap.release()
cv2.destroyAllWindows()
if cursor and db:
    cursor.close()
    db.close()