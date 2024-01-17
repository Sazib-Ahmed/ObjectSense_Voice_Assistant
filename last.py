from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
from mysql.connector import connect, Error
from datetime import datetime

# Load the YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

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

# Function to determine spatial relationships
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

# Store the track history and for FPS calculation
prev_frame_time = 0
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    try:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        new_frame_time = time.time()

        # Run YOLOv8 segmentation on the frame
        #results = model(frame)
        #results = model.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.50, imgsz=640, iou=0.50) 
        #results = model(frame, conf=0.50, imgsz=640, iou=0.50) 

        results = model.track(frame, conf=0.5, iou=0.6, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()


        # Process the results to get masks and bounding boxes
        class_masks = {}
        mask_areas = {}
        bounding_boxes = {}
        seg_classes = list(results[0].names.values())
        seg_class_indices = set(results[0].names.keys())
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        for result in results:
            masks = result.masks.data
            boxes = result.boxes.data
            clss = boxes[:, 5]

            for i in seg_class_indices:
                if i in clss:
                    seg_class = seg_classes[i]
                    obj_indices = torch.where(clss == i)
                    obj_masks = masks[obj_indices]
                    obj_mask = torch.any(obj_masks, dim=0).int() * 255
                    class_masks[seg_class] = obj_mask
                    mask_areas[seg_class] = torch.sum(obj_mask).item()

                    # Extract bounding boxes
                    obj_bbox = boxes[obj_indices].squeeze(0)
                    if obj_bbox.ndim == 1:  # In case squeezing leads to a 1D tensor
                        bounding_boxes[seg_class] = obj_bbox.unsqueeze(0)  # Add an extra dimension
                    else:
                        bounding_boxes[seg_class] = obj_bbox

        # Determine spatial relationships
        process_overlaps(class_masks, mask_areas, bounding_boxes)

        # Calculate and display the FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # Show the frame
        cv2.imshow("YOLOv8 Segmentation and Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        print(f"An error occurred during video processing: {e}")

cap.release()
cv2.destroyAllWindows()
if cursor and db:
    cursor.close()
    db.close()
