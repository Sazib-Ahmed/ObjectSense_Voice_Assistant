from itertools import combinations
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

# Load a pretrained YOLOv8n-seg Segment model
model = YOLO('yolov8n-seg.pt')

# Run inference on an image
results = model('Photo.jpg')  # results list
result = results[0]  # Get the first result, assuming a single image

# Create a directory for output
Path("./test_output/").mkdir(parents=True, exist_ok=True)

# Save the original image
cv2.imwrite("./test_output/original_image.jpg", result.orig_img)

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
            cv2.imwrite(f'./test_output/{seg_class}s.jpg', obj_mask.cpu().numpy())

            # Extract bounding boxes
            obj_bbox = boxes[obj_indices].squeeze(0)
            if obj_bbox.ndim == 1:  # In case squeezing leads to a 1D tensor
                bounding_boxes[seg_class] = obj_bbox.unsqueeze(0)  # Add an extra dimension
            else:
                bounding_boxes[seg_class] = obj_bbox

# Helper functions for spatial relationships
def is_overlapping(box1, box2):
    """Check if two boxes are overlapping."""
    x1, y1, x2, y2 = box1[:4]
    x3, y3, x4, y4 = box2[:4]
    return (x1 < x4 and x2 > x3) and (y1 < y4 and y2 > y3)

def horizontal_relationship(x1, x2, x3, x4):
    """Determine the horizontal relationship between two boxes."""
    if x2 < x3:
        return 'to the left of'
    elif x1 > x4:
        return 'to the right of'
    return 'horizontally aligned with'

def vertical_relationship(y1, y2, y3, y4):
    """Determine the vertical relationship between two boxes."""
    if y2 < y3:
        return 'below'
    elif y1 > y4:
        return 'above'
    return 'vertically aligned with'

def determine_spatial_relationship(class1, boxes1, class2, boxes2):
    """Determine the spatial relationship between two classes of objects."""
    for box1 in boxes1:
        for box2 in boxes2:
            x1, y1, x2, y2 = box1[:4]
            x3, y3, x4, y4 = box2[:4]

            if is_overlapping(box1, box2):
                horizontal_rel = horizontal_relationship(x1, x2, x3, x4)
                vertical_rel = vertical_relationship(y1, y2, y3, y4)

                if horizontal_rel == 'horizontally aligned with' and vertical_rel == 'vertically aligned with':
                    return f"{class1} is overlapping {class2}"
                else:
                    return f"{class1} is {horizontal_rel} and {vertical_rel} {class2}"
            else:
                horizontal_rel = horizontal_relationship(x1, x2, x3, x4)
                vertical_rel = vertical_relationship(y1, y2, y3, y4)
                return f"{class1} is {horizontal_rel} and {vertical_rel} {class2}"

# Function to process and determine spatial relationships
def process_relationships(bounding_boxes):
    for (class1, boxes1), (class2, boxes2) in combinations(bounding_boxes.items(), 2):
        spatial_relationship = determine_spatial_relationship(class1, boxes1, class2, boxes2)
        print(f"Spatial relationship between {class1} and {class2}: {spatial_relationship}")

process_relationships(bounding_boxes)
