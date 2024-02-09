from collections import defaultdict

import torch
import cv2
import numpy as np
import time
from ultralytics import YOLO
from mysql.connector import connect, Error
from datetime import datetime
# from ultralytics.utils.plotting import Annotator
# import sys
# from gui.widget import Widget
# Display the processed frame in the GUI
# from itertools import combinations
# from pathlib import Path


def sep(mes="---"):
    print("==================================="+mes+"===================================")

# Function to determine spatial relationship between two sets of bounding boxes
def determine_spatial_relationship(boxes1, area1, boxes2, area2):
    # Iterate through each bounding box in the first set
    for box1 in boxes1:
        # Iterate through each bounding box in the second set
        for box2 in boxes2:
            # Extract coordinates of the bounding boxes
            x1, y1, x2, y2 = box1[:4]
            x3, y3, x4, y4 = box2[:4]

            # Check for horizontal and vertical overlap between the bounding boxes
            horizontal_overlap = (x1 < x4 and x2 > x3)
            vertical_overlap = (y1 < y4 and y2 > y3)

            # Check if one box is completely inside the other
            is_box1_inside_box2 = x1 >= x3 and x2 <= x4 and y1 >= y3 and y2 <= y4
            is_box2_inside_box1 = x3 >= x1 and x4 <= x2 and y3 >= y1 and y4 <= y2

            # Determine spatial relationship based on containment
            if is_box1_inside_box2:
                if area1 < area2:
                    return "on"  # Box 1 is completely inside Box 2
                else:
                    return "surrounds"  # Box 1 completely surrounds Box 2
            elif is_box2_inside_box1:
                if area2 < area1:
                    return "on"  # Box 2 is completely inside Box 1
                else:
                    return "surrounds"  # Box 2 completely surrounds Box 1

            # Calculate overlap percentage and size difference
            overlap_width = min(x2, x4) - max(x1, x3)
            overlap_height = min(y2, y4) - max(y1, y3)
            overlap_area = max(overlap_width, 0) * max(overlap_height, 0)
            size_difference = abs(area1 - area2) / max(area1, area2)

            # Determine the spatial relationship based on size, position, and overlap
            if horizontal_overlap and vertical_overlap:
                if overlap_area / min(area1, area2) > 0.5:  # Significant overlap
                    if area1 > area2:
                        return "covering"  # Box 1 is covering Box 2
                    else:
                        return "covering"  # Box 2 is covering Box 1
                else:
                    if y1 < y3:
                        return "above"  # Box 1 is above Box 2
                    else:
                        return "below"  # Box 1 is below Box 2
            elif horizontal_overlap:
                if size_difference > 0.5:
                    return "beside"  # Boxes are beside each other
                else:
                    return "beside"  # Boxes are beside each other
            elif vertical_overlap:
                if size_difference > 0.5:
                    return "near"  # Boxes are near each other
                else:
                    return "near"  # Boxes are near each other

    return "around"  # No significant spatial relationship



# Function to check if bounding boxes overlap
def do_bounding_boxes_overlap(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x3, y3, x4, y4 = box2[:4]
    return (x1 < x4 and x2 > x3) and (y1 < y4 and y2 > y3)


# Function to calculate the Euclidean distance between two bounding boxes represented as lines.
def calculate_distance(box1, box2):
    # Extract coordinates and dimensions of the bounding boxes
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

    # List of line segments for both bounding boxes
    lines1 = [(line1_start, line1_end), (line2_start, line2_end), (line3_start, line3_end), (line4_start, line4_end)]
    lines2 = [(line5_start, line5_end), (line6_start, line6_end), (line7_start, line7_end), (line8_start, line8_end)]

    distances = []

    # Iterate through each pair of lines and calculate the distance
    for line1 in lines1:
        for line2 in lines2:
            distance = calculate_line_distance(line1, line2)
            distances.append(distance)

    return distances

# Function to calculate the distance between two line segments using the formula for the minimum distance between two lines.
def calculate_line_distance(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    # Calculate the numerator and denominator for the formula of minimum distance between two lines
    numerator = abs((x2 - x1)*(y3 - y4) + (x3 - x4)*(y2 - y1) + (x4 - x3)*(y1 - y3))
    denominator = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    # Calculate the distance using the formula
    distance = numerator / denominator

    return distance



# Function to determine the relative relationship between a mobile object and a stationary object based on their minimum distance.
def determine_relative_relationship(mobile_box, stationary_box):
    close_threshold = 200  # Threshold for considering objects as "close"
    nearby_threshold = 500  # Threshold for considering objects as "nearby"
    # Iterate through each bounding box of the mobile object
    for box1 in mobile_box:
        # Iterate through each bounding box of the stationary object
        for box2 in stationary_box:
            # Calculate the distances between the two objects
            distances = calculate_distance(box1[:4], box2[:4])
            # Find the minimum distance
            min_distance = min(distances)
            # Determine the relative relationship based on the minimum distance
            if min_distance < close_threshold:
                return f"Close to"  # Mobile object is close to the stationary object
            elif min_distance < nearby_threshold:
                return f"Nearby"  # Mobile object is nearby the stationary object
            else:
                return "Far away from"  # Mobile object is far away from the stationary object


    
# Function to process overlaps and determine relationships
def find_location( mask1, area1, bounding_boxes1, mask2, area2, bounding_boxes2):
    overlap = mask1 & mask2
    if torch.any(overlap):
        # Determine spatial relationship
        spatial_relationship="Overlap spatial relationship not working "

        spatial_relationship = determine_spatial_relationship(bounding_boxes1, area1, bounding_boxes2, area2)
        return spatial_relationship
    
    else:
        # Determine relative relationship
        relative_relationship="Bounding Box spatial relationship not working "
        
        relative_relationship = determine_relative_relationship(bounding_boxes1, bounding_boxes2)
        return relative_relationship




# Function to find the nearest stationary object for a given mobile object
def get_closest_stationary_object(mobile_object_boxes, stationary_object_boxes):
    # Dictionary to store the closest stationary object for each mobile object
    closest_stationary_objects = {}

    # Iterate through each mobile object and its bounding box coordinates
    for mobile_track_id, mobile_box in mobile_object_boxes.items():
        mobile_x, mobile_y, _, _ = mobile_box  # Extract x and y coordinates of the mobile object

        # Initialize variables to keep track of the closest stationary object
        closest_stationary_track_id = None
        min_distance = float('inf')  # Initialize minimum distance as positive infinity

        # Iterate through each stationary object and its bounding box coordinates
        for stationary_track_id, stationary_box in stationary_object_boxes.items():
            stationary_x, stationary_y, _, _ = stationary_box  # Extract x and y coordinates of the stationary object

            # Calculate Euclidean distance between mobile and stationary objects
            distance = np.sqrt((mobile_x - stationary_x) ** 2 + (mobile_y - stationary_y) ** 2)

            # Update closest stationary object if distance is smaller
            if distance < min_distance:
                closest_stationary_track_id = stationary_track_id
                min_distance = distance

        # Save the closest stationary object for the current mobile object
        closest_stationary_objects[mobile_track_id] = closest_stationary_track_id

    return closest_stationary_objects



def process_video(widget_instance, frame_callback=None):
    # Define class names present in the COCO dataset
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

    # Define moving object IDs
    mobile_object_ids = (
        15,16,24,25,26,27,28,32,39,41,42,43,44,
        45,63,64,65,66,67,73,74,75,76,77,78,79
    )

    # Define stationary object IDs
    stationary_object_ids = (
        13,56,57,58,59,60,62,68,69,72
    )

    # Retrieve selected settings from the widget_instance
    selected_video_source = widget_instance.selected_video_source
    selected_video_file = widget_instance.selected_video_file
    selected_live_video_input = widget_instance.selected_live_video_input
    selected_detection_model = widget_instance.selected_detection_model
    selected_pixel_size = widget_instance.selected_pixel_size
    selected_tracker = widget_instance.selected_tracker
    selected_confidence = widget_instance.selected_confidence
    selected_iou = widget_instance.selected_iou

    # Set video path based on the selected source
    if selected_video_source == "file":
        video_path = selected_video_file
    elif selected_video_source == "live":
        video_path = selected_live_video_input
    else:
        video_path = 0

    # Set default values if certain settings are not provided
    if not selected_detection_model:
        selected_detection_model = "yolov8s-seg.pt"
    if not selected_pixel_size:
        selected_pixel_size = 640
    if not selected_tracker:
        selected_tracker = "botsort.yaml"
    if not selected_confidence:
        selected_confidence = 0.25
    if not selected_iou:
        selected_iou = 0.7

    # Load the YOLOv8 model
    model = YOLO(selected_detection_model)

    # Open the video capture object
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        print(f"Error opening video source: {e}")
        exit(1)

    # Initialize variables for track history and FPS calculation
    track_history = defaultdict(lambda: [])
    prev_frame_time = 0
    annotator = None  # Initialize annotator variable

    # Define the desired frame rate (5 frames per second)
    desired_fps = 20
    frame_delay = 1 / desired_fps

    # Connect to the database
    try:
        db = connect(host="localhost", user="root", password="", database="assistant")
        cursor = db.cursor()
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        exit(1)

    # Initialize variables for object detection and tracking
    detected_class_names = {}
    detected_class_ids = {}
    class_masks = {}
    mask_areas = {}
    bounding_boxes = {}
    stationary_objects_boxes = {}
    mobile_objects_boxes = {}
    frame_num = 0

    # Loop through the video frames
    while cap.isOpened():
        try:
            success, frame = cap.read()
            frame_num += 1

            # Display frame number
            print("")
            mes = "Frame Number: " + str(frame_num)
            sep(mes)

            if not success:
                print("Failed to grab frame")
                cap.release()
                cv2.destroyAllWindows()
                break

            new_frame_time = time.time()

            # Run YOLOv8 tracking on the frame
            results = model.track(frame, persist=True, verbose=False, imgsz=selected_pixel_size, tracker=selected_tracker,
                                  conf=selected_confidence, iou=selected_iou)
            # Initialize dictionaries
            class_masks = {}
            mask_areas = {}
            bounding_boxes = {}
            stationary_objects_boxes = {}
            mobile_objects_boxes = {}
            closest_stationary_objects = {}

            # Process detection results
            if results[0].boxes is not None and getattr(results[0].boxes, 'id', None) is not None:
                boxes_xywh = results[0].boxes.xywh.cpu()
                masks = results[0].masks.data
                boxes = results[0].boxes.data
                track_ids = results[0].boxes.id
                class_ids = results[0].boxes.cls

                for i, track_id in enumerate(track_ids):
                    track_id = int(track_id)
                    class_id = class_ids[i].int().item()
                    detected_class_ids[track_id] = class_id

                    # Extract class name and related information
                    class_name = class_names[class_id]
                    detected_class_names[track_id] = class_name
                    obj_indices = torch.where(track_ids == track_id)
                    obj_masks = masks[obj_indices]
                    obj_mask = torch.any(obj_masks, dim=0).int() * 255
                    class_masks[track_id] = obj_mask
                    mask_areas[track_id] = torch.sum(obj_mask).item()

                    obj_bbox = boxes[obj_indices].squeeze(0)
                    if obj_bbox.ndim == 1:
                        bounding_boxes[track_id] = obj_bbox.unsqueeze(0)
                    else:
                        bounding_boxes[track_id] = obj_bbox

                    x, y, w, h = boxes_xywh[i]
                    x_value = float(x)
                    y_value = float(y)
                    w_value = float(w)
                    h_value = float(h)

                    if class_id in mobile_object_ids:
                        mobile_objects_boxes[track_id] = (x_value, y_value, w_value, h_value)

                    if class_id in stationary_object_ids:
                        stationary_objects_boxes[track_id] = (x_value, y_value, w_value, h_value)

                # Find closest stationary objects to each mobile object
                if mobile_objects_boxes and stationary_objects_boxes:
                    closest_stationary_objects = get_closest_stationary_object(mobile_objects_boxes,stationary_objects_boxes)
                # Update database with detection records
                if closest_stationary_objects:
                    for mobile_object_track_id, stationary_object_track_id in closest_stationary_objects.items():
                        location = find_location(class_masks[mobile_object_track_id],
                                                 mask_areas[mobile_object_track_id],
                                                 bounding_boxes[mobile_object_track_id],
                                                 class_masks[stationary_object_track_id],
                                                 mask_areas[stationary_object_track_id],
                                                 bounding_boxes[stationary_object_track_id])
                        x_value, y_value, w_value, h_value = mobile_objects_boxes[mobile_object_track_id]
                        try:
                            cursor.execute("SELECT * FROM detections WHERE mobile_object_tracker_id = %s",
                                           (mobile_object_track_id,))
                            existing_record = cursor.fetchone()
                            if existing_record:
                                cursor.execute(
                                    "UPDATE detections SET mobile_object_class_id = %s,"
                                    " mobile_object_class_name = %s,"
                                    " stationary_object_tracker_id = %s,"
                                    " stationary_object_class_id = %s,"
                                    " stationary_object_class_name = %s,"
                                    " x = %s,"
                                    " y = %s,"
                                    " width = %s,"
                                    " height = %s,"
                                    " location = %s,"
                                    " timestamp = %s"
                                    " WHERE mobile_object_tracker_id = %s",
                                    (
                                        detected_class_ids[mobile_object_track_id],
                                        detected_class_names[mobile_object_track_id],
                                        stationary_object_track_id,
                                        detected_class_ids[stationary_object_track_id],
                                        detected_class_names[stationary_object_track_id],
                                        x_value,
                                        y_value,
                                        w_value,
                                        h_value,
                                        location,
                                        datetime.now(),
                                        mobile_object_track_id
                                    )
                                )
                            else:
                                cursor.execute(
                                    "INSERT INTO detections (mobile_object_tracker_id,"
                                    " mobile_object_class_id,"
                                    " mobile_object_class_name,"
                                    " stationary_object_tracker_id,"
                                    " stationary_object_class_id,"
                                    " stationary_object_class_name,"
                                    " x,"
                                    " y,"
                                    " width,"
                                    " height,"
                                    " location,"
                                    " timestamp)"
                                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                                    (
                                        mobile_object_track_id,
                                        detected_class_ids[mobile_object_track_id],
                                        detected_class_names[mobile_object_track_id],
                                        stationary_object_track_id,
                                        detected_class_ids[stationary_object_track_id],
                                        detected_class_names[stationary_object_track_id],
                                        x_value,
                                        y_value,
                                        w_value,
                                        h_value,
                                        location,
                                        datetime.now()
                                    )
                                )
                            db.commit()
                        except Error as e:
                            print(f"Error interacting with database: {e}")
                        print(detected_class_names[mobile_object_track_id], mobile_object_track_id, location,
                              detected_class_names[stationary_object_track_id], stationary_object_track_id)
    
                annotator = results[0].plot()


            else:
                annotator = frame.copy()

            # Calculate and display FPS
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(annotator, f"FPS: {fps:.2f}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3,
                        cv2.LINE_AA)

            # Show annotated frame
            widget_instance.display_video_frame(annotator)

            # Call frame callback function
            frame_callback(annotator)

            # Check if video thread is stopped
            if widget_instance.video_thread.stopped:
                break

        except Exception as e:
            print(f"An error occurred during video processing: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()
