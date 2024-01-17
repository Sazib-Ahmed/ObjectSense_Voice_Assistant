from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
from itertools import combinations
import numpy as np


# Load a pretrained YOLOv8n-seg Segment model
model = YOLO('yolov8n-seg.pt')

# Run inference on an image
results = model('Photo.jpg')  # results list

# Get the first result, assuming a single image
result = results[0]

# Create a directory for output
Path("./test_output/").mkdir(parents=True, exist_ok=True)

# Save the original image
cv2.imwrite("./test_output/original_image.jpg", result.orig_img)

# Get the class names and class indices
seg_classes = list(result.names.values())
seg_class_indices = set(result.names.keys())
print("seg_classes:", seg_classes)
print("seg_class_indices:", seg_class_indices)

# Dictionary to store masks and their areas for each class
class_masks = {}
mask_areas = {}

def calculate_overlap_ratio(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    if torch.sum(union) == 0:
        return 0
    return torch.sum(intersection) / torch.sum(union)

def estimate_depth(size1, size2):
    return size1 > size2

def is_contour_in_front(mask1, mask2):
    # Convert PyTorch tensor to NumPy array and ensure it's in the right format
    mask1_np = mask1.cpu().numpy().astype(np.uint8)
    mask2_np = mask2.cpu().numpy().astype(np.uint8)

    contours1, _ = cv2.findContours(mask1_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt1 in contours1:
        for point in cnt1:
            if mask2_np[point[0][1], point[0][0]] == 0:
                return True

    return False

# Iterate over results to process masks and bounding boxes
for result in results:
    masks = result.masks.data
    boxes = result.boxes.data
    print("masks:", masks)
    print("boxes:", boxes)

    # Get the class indices
    clss = boxes[:, 5]
    print("Class Indices:", clss)

    # Extract a single mask with all the classes
    obj_indices = torch.where(clss != -1)
    obj_masks = masks[obj_indices]
    obj_mask = torch.any(obj_masks, dim=0).int() * 255
    cv2.imwrite('./test_output/all-masks.jpg', obj_mask.cpu().numpy())

    # Process masks for each detected class
    for i in seg_class_indices:
        if i in clss:
            seg_class = seg_classes[i]
            obj_indices = torch.where(clss == i)
            obj_masks = masks[obj_indices]
            obj_mask = torch.any(obj_masks, dim=0).int() * 255
            class_masks[seg_class] = obj_mask
            mask_areas[seg_class] = torch.sum(obj_mask).item()
            # After mask saving
            cv2.imwrite(f'./test_output/{seg_class}s_mask.jpg', obj_mask.cpu().numpy())

           

            # Add more such print statements to understand the flow and values in your logic.





# Enhanced Overlap Detection
for (class1, mask1), (class2, mask2) in combinations(class_masks.items(), 2):
    overlap_ratio1 = calculate_overlap_ratio(mask1, mask2)
    overlap_ratio2 = calculate_overlap_ratio(mask2, mask1)
    area1 = mask_areas[class1]
    area2 = mask_areas[class2]
    depth1_in_front = estimate_depth(area1, area2)
    contour1_in_front = is_contour_in_front(mask1, mask2)
     # Inside the overlap detection loop
    print(f"Checking overlap between {class1} and {class2}")
    overlap_ratio1 = calculate_overlap_ratio(mask1, mask2)
    overlap_ratio2 = calculate_overlap_ratio(mask2, mask1)
    print(f"Overlap Ratios: {overlap_ratio1}, {overlap_ratio2}")

    if overlap_ratio1 > 0.1 or overlap_ratio2 > 0.1:
        if depth1_in_front and contour1_in_front:
            print(f"Overlap detected: {class1} is likely in front of {class2}")
        else:
            print(f"Overlap detected: {class2} is likely in front of {class1}")

# You can add more code here if needed
