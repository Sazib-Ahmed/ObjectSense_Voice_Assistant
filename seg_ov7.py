from itertools import combinations
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

# Load a pretrained YOLOv8n-seg Segment model
model = YOLO('yolov8m-seg.pt')

# Run inference on an image
results = model('test_images/IMG_5064.JPG')  # results list
result = results[0]  # Get the first result, assuming a single image

# Process the results
seg_classes = list(result.names.values())
seg_class_indices = set(result.names.keys())

class_masks = {}
mask_areas = {}
bounding_boxes = {}

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

# Define moving and stationary object IDs
moving_object_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79
]

# Define the list of stationary object IDs
stationary_object_ids = [13, 56, 57, 58, 59, 60, 61, 62, 68, 69, 70, 71, 72]

# Store stationary objects' data
stationary_objects_data = {}

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


for result in results:
    masks = result.masks.data
    boxes = result.boxes.data
    clss = boxes[:, 5]

    for i in stationary_object_ids:
        if i in clss:
            obj_indices = torch.where(clss == i)
            obj_bbox = boxes[obj_indices].squeeze(0)
            if obj_bbox.ndim == 1:
                stationary_objects_data[i] = obj_bbox.unsqueeze(0)
            else:
                stationary_objects_data[i] = obj_bbox

# Function to determine spatial relationship with stationary objects
def relative_position_to_stationary(moving_box, stationary_data):
    relationships = []
    for stationary_id, stationary_boxes in stationary_data.items():
        for stationary_box in stationary_boxes:
            # Ensure both moving_box and stationary_box are at least 1-dimensional
            if moving_box.ndim < 1 or stationary_box.ndim < 1:
                print("Error: One of the bounding boxes is 0-dimensional.")
                print("Moving Box Shape:", moving_box.shape)
                print("Stationary Box Shape:", stationary_box.shape)
                continue  # Skip this iteration

            relation = determine_spatial_relationship("Moving Object", moving_box, 0, 
                                                      f"Stationary Object {stationary_id}", stationary_box, 0)
            relationships.append(relation)
    return relationships

# ... [previous code] ...

# Process moving objects and compare with stationary objects
for result in results:
    masks = result.masks.data
    boxes = result.boxes.data
    clss = boxes[:, 5]

    for i in moving_object_ids:
        if i in clss:
            obj_indices = torch.where(clss == i)
            obj_bboxes = boxes[obj_indices].squeeze(0)  # obj_bboxes contains all bboxes for this class
            print("obj_bboxes shape after extraction:", obj_bboxes.shape)

            # Skip if obj_bboxes is empty
            if obj_bboxes.nelement() == 0:
                print(f"No bounding boxes found for moving object ID {i}")
                continue

            # If obj_bboxes is a single bbox (1D), use it directly
            if obj_bboxes.ndim == 1:
                obj_bboxes = obj_bboxes.unsqueeze(0)  # Make it 2D for consistency

            for bbox in obj_bboxes:
                relationships = relative_position_to_stationary(bbox, stationary_objects_data)
                for relation in relationships:
                    print(relation)

# ... [Rest of your existing code] ...
