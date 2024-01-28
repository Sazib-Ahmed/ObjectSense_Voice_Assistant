from ultralytics import YOLO
import cv2
import torch
from pathlib import Path
from itertools import combinations

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
            cv2.imwrite(f'./test_output/{seg_class}s.jpg', obj_mask.cpu().numpy())

# Check for overlapping masks and determine which object is in front
for (class1, mask1), (class2, mask2) in combinations(class_masks.items(), 2):
    overlap = mask1 & mask2
    if torch.any(overlap):
        area1 = mask_areas[class1]
        area2 = mask_areas[class2]
        print(f"area1: {area1}")
        print(f"area2: {area2}")

        if area2 > area1:
            print(f"Overlap detected: {class1} is in front of {class2}")
        else:
            print(f"Overlap detected: {class2} is in front of {class1}")

# You can add more code here if needed