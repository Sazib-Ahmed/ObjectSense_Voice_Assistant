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
from ultralytics.utils.plotting import Annotator
import sys

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



# Load the YOLOv8 model
model = YOLO('../yolov8m-seg.pt')

# Open the video file or webcam
video_path = 0
#video_path = '../Sequence01.mp4'
try:
    cap = cv2.VideoCapture(video_path)
except Exception as e:
    print(f"Error opening video source: {e}")
    exit(1)

# Store the track history and for FPS calculation
track_history = defaultdict(lambda: [])
prev_frame_time = 0

# Define the desired frame rate (5 frames per second)
desired_fps = 10
frame_delay = 1 / desired_fps

# Connect to the database
try:
    db = connect(host="localhost", user="root", password="", database="assistant")
    cursor = db.cursor()
except Error as e:
    print(f"Error connecting to MySQL: {e}")
    exit(1)


detected_class_names = {}
detected_class_ids = {}

class_masks = {}
mask_areas = {}
bounding_boxes = {}
# Dictionary to hold the bounding boxes of stationary objects
stationary_objects_boxes = {}
mobile_objects_boxes={}

# Function to determine spatial relationship
def determine_spatial_relationship( boxes1, area1, boxes2, area2):
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
                    return "on"
                else:
                    return "surrounds"
            elif is_box2_inside_box1:
                if area2 < area1:
                    return "on"
                else:
                    return "surrounds"

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
                        return "covering"
                    else:
                        return "covering"
                else:
                    if y1 < y3:
                        return "above"
                    else:
                        return "below"
            elif horizontal_overlap:
                if size_difference > 0.5:
                    return "beside"
                else:
                    return "beside"
            elif vertical_overlap:
                if size_difference > 0.5:
                    return "near"
                else:
                    return "near"

    return "around"



# Function to check if bounding boxes overlap
def do_bounding_boxes_overlap(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x3, y3, x4, y4 = box2[:4]
    return (x1 < x4 and x2 > x3) and (y1 < y4 and y2 > y3)


def calculate_distance(box1, box2):
    x1, y1, w1, h1 = box1[:4].detach().numpy()
    x2, y2, w2, h2 = box2[:4].detach().numpy()

    # Define the endpoints of the lines for box1
    line1_start = (x1, y1)
    line1_end = (x1 + w1, y1)
    line2_start = (x1, y1)
    line2_end = (x1, y1 + h1)
    line3_start = (x1 + w1, y1)
    line3_end = (x1 + w1, y1 + h1)
    line4_start = (x1, y1 + h1)
    line4_end = (x1 + w1, y1 + h1)

    # Define the endpoints of the lines for box2
    line5_start = (x2, y2)
    line5_end = (x2 + w2, y2)
    line6_start = (x2, y2)
    line6_end = (x2, y2 + h2)
    line7_start = (x2 + w2, y2)
    line7_end = (x2 + w2, y2 + h2)
    line8_start = (x2, y2 + h2)
    line8_end = (x2 + w2, y2 + h2)

    # List of line segments
    lines1 = [(line1_start, line1_end), (line2_start, line2_end), (line3_start, line3_end), (line4_start, line4_end)]
    lines2 = [(line5_start, line5_end), (line6_start, line6_end), (line7_start, line7_end), (line8_start, line8_end)]

    distances = []

    for line1 in lines1:
        for line2 in lines2:
            distance = calculate_line_distance(line1, line2)
            distances.append(distance)

    # print(distances)
    return distances

def calculate_line_distance(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # Calculate the minimum distance between two lines
    numerator = abs((x2 - x1)*(y3 - y4) + (x3 - x4)*(y2 - y1) + (x4 - x3)*(y1 - y3))
    denominator = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    distance = numerator / denominator

    return distance


def check_relative_position(mobile_box, stationary_box):
    close_threshold = 200
    nearby_threshold = 500
    for box1 in mobile_box:
        for box2 in stationary_box:
            distances = calculate_distance(box1[:4], box2[:4])
            min_distance = min(distances)
            # print(distances)
            # print(min_distance)

            if min_distance < close_threshold:
                #position = get_relative_position(box1[:4], box2[:4], distances)
                #return f"Close to {position}"
                return f"Close to"
            elif min_distance < nearby_threshold:
                #position = get_relative_position(box1[:4], box2[:4], distances)
                #return f"Nearby {position}"
                return f"Nearby"
            
            else:
                return "Far away from"

    
# Function to process overlaps and determine spatial relationships
def process_overlaps( mask1, area1, bounding_boxes1, mask2, area2, bounding_boxes2):

    overlap = mask1 & mask2
    #print(overlap)
    if torch.any(overlap):

        # Determine and print spatial relationship
        spatial_relationship="Overlap spatial relationship not working "

        spatial_relationship = determine_spatial_relationship(bounding_boxes1, area1, bounding_boxes2, area2)
        #print(spatial_relationship)
        return spatial_relationship
    else:

        # Determine and print spatial relationship
        relative_position="Bounding Box spatial relationship not working "

        #relative_position = determine_spatial_relationship(bounding_boxes1, area1, bounding_boxes2, area2)
        
        relative_position = check_relative_position(bounding_boxes1, bounding_boxes2)
        # print(relative_position)
        return relative_position



# Function to find the nearest stationary object for a given mobile object
def get_closest_stationary_object(mobile_object_boxes, stationary_object_boxes):
    closest_stationary_objects = {}

    for mobile_track_id, mobile_box in mobile_object_boxes.items():
        mobile_x, mobile_y, _, _ = mobile_box

        # Initialize variables to keep track of the closest stationary object
        closest_stationary_track_id = None
        min_distance = float('inf')

        for stationary_track_id, stationary_box in stationary_object_boxes.items():
            stationary_x, stationary_y, _, _ = stationary_box

            # Calculate Euclidean distance between mobile and stationary objects
            distance = np.sqrt((mobile_x - stationary_x)**2 + (mobile_y - stationary_y)**2)

            # Update closest stationary object if distance is smaller
            if distance < min_distance:
                closest_stationary_track_id = stationary_track_id
                min_distance = distance

        # Save the closest stationary object for the current mobile object
        closest_stationary_objects[mobile_track_id] = closest_stationary_track_id

    return closest_stationary_objects


# Loop through the video frames
while cap.isOpened():
    try:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            cap.release()
            cv2.destroyAllWindows()
            break

        new_frame_time = time.time()

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True,verbose=False)
        # Set dictionaries to null
        detected_class_namess = {}
        class_masks = {}
        mask_areas = {}
        bounding_boxes = {}
        stationary_objects_boxes = {}
        mobile_objects_boxes = {}
        closest_stationary_objects = {}
        

        if results[0].boxes is not None and getattr(results[0].boxes, 'id', None) is not None:
            boxes_xywh = results[0].boxes.xywh.cpu()
            masks = results[0].masks.data
            boxes = results[0].boxes.data
            track_ids_list = results[0].boxes.id.int().cpu().tolist()
            track_ids = results[0].boxes.id
            class_ids_list = results[0].boxes.cls.int().cpu().tolist()
            class_ids = results[0].boxes.cls
            clss = results[0].boxes.cls.cpu().tolist()

            for i, track_id in enumerate(track_ids):
                track_id = int(track_id)
                class_id = class_ids[i].int().item()
                detected_class_ids[track_id]=class_id

                class_name=class_names[class_id]
                detected_class_names[track_id]=class_name
                obj_indices = torch.where(track_ids == track_id)
                obj_masks = masks[obj_indices]
                obj_mask = torch.any(obj_masks, dim=0).int() * 255
                class_masks[track_id] = obj_mask
                mask_areas[track_id] = torch.sum(obj_mask).item()
                #cv2.imwrite(f'./test_output/{class_name}s.jpg', obj_mask.cpu().numpy())

                # Extract bounding boxes
                obj_bbox = boxes[obj_indices].squeeze(0)
                if obj_bbox.ndim == 1:  # In case squeezing leads to a 1D tensor
                    bounding_boxes[track_id] = obj_bbox.unsqueeze(0)  # Add an extra dimension
                else:
                    bounding_boxes[track_id] = obj_bbox
                
                x, y, w, h = boxes_xywh[i]
                x_value = float(x)
                y_value = float(y)
                w_value = float(w)
                h_value = float(h)

                # Get the location of mobile objects
                if class_id in mobile_object_ids:
                    mobile_objects_boxes[track_id] = (x_value, y_value, w_value, h_value)

                if class_id in stationary_object_ids:
                    stationary_objects_boxes[track_id] = (x_value, y_value, w_value, h_value)
            
            if mobile_objects_boxes and stationary_objects_boxes:
                closest_stationary_objects = get_closest_stationary_object(mobile_objects_boxes, stationary_objects_boxes)
                # Check if closest_stationary_objects has values
            if closest_stationary_objects:
                for mobile_object_track_id, stationary_object_track_id in closest_stationary_objects.items():
                    location = process_overlaps(class_masks[mobile_object_track_id], mask_areas[mobile_object_track_id], bounding_boxes[mobile_object_track_id],class_masks[stationary_object_track_id], mask_areas[stationary_object_track_id], bounding_boxes[stationary_object_track_id])
                    #location = process_overlaps(detected_class_names[key],class_masks[key], mask_areas[key], bounding_boxes[key],detected_class_names[value],class_masks[value], mask_areas[value], bounding_boxes[value])
                    x_value, y_value, w_value, h_value = mobile_objects_boxes[mobile_object_track_id]
                    try:
                        cursor.execute("SELECT * FROM detections WHERE mobile_object_tracker_id = %s", (mobile_object_track_id,))
                        existing_record = cursor.fetchone()

                        if existing_record:
                            cursor.execute(
                                "UPDATE detections SET mobile_object_class_id = %s, mobile_object_class_name = %s, stationary_object_tracker_id = %s , stationary_object_class_id = %s, stationary_object_class_name = %s, x = %s, y = %s, width = %s, height = %s, location = %s, timestamp = %s WHERE mobile_object_tracker_id = %s",
                                (detected_class_ids[mobile_object_track_id], detected_class_names[mobile_object_track_id],stationary_object_track_id,detected_class_ids[stationary_object_track_id], detected_class_names[stationary_object_track_id], x_value, y_value, w_value, h_value, location, datetime.now(), mobile_object_track_id)
                            )
                        else:
                            cursor.execute(
                                "INSERT INTO detections (mobile_object_tracker_id, mobile_object_class_id, mobile_object_class_name, stationary_object_tracker_id, stationary_object_class_id, stationary_object_class_name, x, y, width, height, location, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                (mobile_object_track_id, detected_class_ids[mobile_object_track_id], detected_class_names[mobile_object_track_id],stationary_object_track_id,detected_class_ids[stationary_object_track_id], detected_class_names[stationary_object_track_id], x_value, y_value, w_value, h_value, location, datetime.now())
                            )
                        db.commit()
                    except Error as e:
                        print(f"Error interacting with database: {e}")

                    
                    print(detected_class_names[mobile_object_track_id],mobile_object_track_id,location,detected_class_names[stationary_object_track_id],stationary_object_track_id)


            
            annotator = results[0].plot()
            
            # annotator = Annotator(frame, line_width=2, example=str(model.names))
            # for box, cls in zip(boxes_xywh, clss):
            #     x, y, w, h = box
            #     label = int(cls)
            #     x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            #     annotator.box_label([x1, y1, x2, y2], label=str(label), color=(0, 0, 255))
            
        # Calculate and display the FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(annotator, f"FPS: {fps:.2f}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # Show the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotator)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
        # Introduce delay to achieve the desired frame rate
        time.sleep(frame_delay)
    except Exception as e:
        print(f"An error occurred during video processing: {e}")

cap.release()
cv2.destroyAllWindows()
if cursor and db:
    cursor.close()
    db.close()







