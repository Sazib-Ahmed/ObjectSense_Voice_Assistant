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

# Function to process overlaps
def process_overlaps(class_masks, mask_areas):
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

            if area2 > area1:
                print(f"Overlap detected: {class1} is in front of {class2}")
            else:
                print(f"Overlap detected: {class2} is in front of {class1}")

process_overlaps(class_masks, mask_areas)
