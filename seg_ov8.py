from itertools import combinations
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

# Load a pretrained YOLOv8n-seg Segment model
model = YOLO('yolov8m-seg.pt')

# Run inference on an image
results = model('test_images/IMG_5061.JPG')  # results list
result = results[0]  # Get the first result, assuming a single image

# Create a directory for output
#Path("./test_output/").mkdir(parents=True, exist_ok=True)

# Save the original image
#cv2.imwrite("./test_output/original_image.jpg", result.orig_img)

# Process the results
seg_classes = list(result.names.values())
seg_class_indices = set(result.names.keys())

# Define moving and stationary object IDs
moving_object_ids = [1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79]
stationary_object_ids = [13, 56, 57, 58, 59, 60, 61, 62, 68, 69, 70, 71, 72]

# Process the results to extract moving and stationary objects
moving_objects = {}
stationary_objects = {}

for result in results:
    masks = result.masks.data
    boxes = result.boxes.data
    clss = boxes[:, 5]

    for i, seg_class in enumerate(seg_classes):
        if i in moving_object_ids:
            obj_indices = torch.where(clss == i)
            obj_bbox = boxes[obj_indices].squeeze(0)
            if obj_bbox.ndim == 1:  # In case squeezing leads to a 1D tensor
                moving_objects[seg_class] = obj_bbox.unsqueeze(0)  # Add an extra dimension
            else:
                moving_objects[seg_class] = obj_bbox
        elif i in stationary_object_ids:
            obj_indices = torch.where(clss == i)
            obj_bbox = boxes[obj_indices].squeeze(0)
            if obj_bbox.ndim == 1:  # In case squeezing leads to a 1D tensor
                stationary_objects[seg_class] = obj_bbox.unsqueeze(0)  # Add an extra dimension
            else:
                stationary_objects[seg_class] = obj_bbox

# Function to determine spatial relationship between moving and stationary objects
def determine_spatial_relationship(moving, stationary):
    spatial_relationships = []
    for moving_class, moving_boxes in moving.items():
        for stationary_class, stationary_boxes in stationary.items():
            for moving_box in moving_boxes:
                for stationary_box in stationary_boxes:
                    spatial_relationship = check_spatial_relation(moving_box, stationary_box, moving_class, stationary_class)
                    if spatial_relationship:
                        spatial_relationships.append(spatial_relationship)
    return spatial_relationships

# Helper function to check spatial relationship between two boxes
def check_spatial_relation(moving_box, stationary_box, moving_class, stationary_class):
    x1, y1, x2, y2 = moving_box[:4]
    x3, y3, x4, y4 = stationary_box[:4]

    horizontal_overlap = (x1 < x4 and x2 > x3)
    vertical_overlap = (y1 < y4 and y2 > y3)

    if horizontal_overlap and vertical_overlap:
        return f"{moving_class} is near {stationary_class}"

    return None

# Process overlaps and determine spatial relationships
spatial_relationships = determine_spatial_relationship(moving_objects, stationary_objects)
for relation in spatial_relationships:
    print(relation)