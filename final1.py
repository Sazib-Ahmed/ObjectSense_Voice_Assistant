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

def sep(mes="---"):
    print("==================================="+mes+"===================================")
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
model = YOLO('../yolov8n-seg.pt')

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



class_masks = {}
mask_areas = {}
bounding_boxes = {}

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
                    return f"{class1} and {class2} are beside each other"
                else:
                    return f"{class1} is beside {class2}"
            elif vertical_overlap:
                if size_difference > 0.5:
                    return f"{class1} and {class2} are near each other"
                else:
                    return f"{class1} is near {class2}"

    return f"No direct spatial relationship detected between {class1} and {class2}"



# Function to check if bounding boxes overlap
def do_bounding_boxes_overlap(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x3, y3, x4, y4 = box2[:4]
    return (x1 < x4 and x2 > x3) and (y1 < y4 and y2 > y3)

# Function to process overlaps and determine spatial relationships
def process_overlaps(class_masks, mask_areas, bounding_boxes):
    for (class1, mask1), (class2, mask2) in combinations(class_masks.items(), 2):
        overlap = mask1 & mask2
        if torch.any(overlap):
            area1 = mask_areas[class1]
            area2 = mask_areas[class2]
            overlap_area = torch.sum(overlap).item()

            if area1 == 0 or area2 == 0:
                continue

            overlap_percentage_class1 = (overlap_area / area1) * 100 if area1 > 0 else 0
            overlap_percentage_class2 = (overlap_area / area2) * 100 if area2 > 0 else 0

            print(f"Overlap percentage for {class1}: {overlap_percentage_class1:.2f}%")
            print(f"Overlap percentage for {class2}: {overlap_percentage_class2:.2f}%")

            # if area2 > area1:
            #     print(f"Overlap detected: {class1} is in front of {class2}")
            # else:
            #     print(f"Overlap detected: {class2} is in front of {class1}")

            # Determine and print spatial relationship
            spatial_relationship = determine_spatial_relationship(class1, bounding_boxes[class1], area1, class2, bounding_boxes[class2], area2)
            print(spatial_relationship)
        else:
            # Check for bounding box overlap if masks do not overlap
            if any(do_bounding_boxes_overlap(box1, box2) for box1 in bounding_boxes[class1] for box2 in bounding_boxes[class2]):
                print(f"{class1} is near {class2}")




# Dictionary to hold the locations of mobile objects
mobile_objects_locations = {}

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to find the nearest stationary object for a given mobile object
def find_nearest_stationary_object(mobile_object_location, stationary_objects_locations):
    min_distance = float('inf')
    nearest_stationary_object = None

    for stationary_object, stationary_location in stationary_objects_locations.items():
        distance = calculate_distance(mobile_object_location, stationary_location)
        if distance < min_distance:
            min_distance = distance
            nearest_stationary_object = stationary_object

    return nearest_stationary_object









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
            #track_ids = results[0].boxes.id.int().cpu().tolist()
            track_ids = results[0].boxes.id
            #class_ids = results[0].boxes.cls.int().cpu().tolist()
            class_ids = results[0].boxes.cls

            for i, track_id in enumerate(track_ids):
                track_id = int(track_id)
                print("Tracking ID:",track_id)
                print(type(track_id))
                class_id = class_ids[i].int().item()
                print("Class ID:",class_id)
                print(type(class_id))
                class_name=class_names[class_id]
                print("Clase Name:",class_name)
                print(type(class_name))
                if class_id in mobile_object_ids:
                    print("Mobile object")
                else:
                    print("Stationary object")

                seg_class = class_names[i]
                obj_indices = torch.where(track_ids == track_id)
                obj_masks = masks[obj_indices]
                obj_mask = torch.any(obj_masks, dim=0).int() * 255
                class_masks[track_id] = obj_mask
                mask_areas[track_id] = torch.sum(obj_mask).item()
                #cv2.imwrite(f'./test_output/{seg_class}s.jpg', obj_mask.cpu().numpy())

                # Extract bounding boxes
                obj_bbox = boxes[obj_indices].squeeze(0)
                if obj_bbox.ndim == 1:  # In case squeezing leads to a 1D tensor
                    bounding_boxes[track_id] = obj_bbox.unsqueeze(0)  # Add an extra dimension
                else:
                    bounding_boxes[track_id] = obj_bbox

                # Get the location of mobile objects
                if i in mobile_object_ids:
                    sep("IN if i ")
                    print(i)
                    mobile_object_location = (int((obj_bbox[0] + obj_bbox[2]) / 2), int((obj_bbox[1] + obj_bbox[3]) / 2))
                    mobile_objects_locations[seg_class] = mobile_object_location

            sep()
            # Find the nearest stationary object for each mobile object
            for mobile_object, mobile_location in mobile_objects_locations.items():
                nearest_stationary_object = find_nearest_stationary_object(mobile_location, stationary_objects_bboxes)
                print(f"{mobile_object} is near {nearest_stationary_object}")

            sep()
            process_overlaps(class_masks, mask_areas, bounding_boxes)
            annotated_frame = results[0].plot()
            
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