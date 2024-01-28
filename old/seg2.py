from ultralytics import YOLO
import cv2
import torch
from pathlib import Path

# Load a pretrained YOLOv8n-seg Segment model
model = YOLO('yolov8s-seg.pt')

# Run inference on an image
results = model('Photo.jpg')  # results list

# Get the first result, assuming a single image
result = results[0]

# Print class names
#print(result.names)

# Create a directory for output
Path("./test_output/").mkdir(parents=True, exist_ok=True)

# Save the original image
cv2.imwrite("./test_output/original_image.jpg", result.orig_img)

# Get the class names and class indices
seg_classes = list(result.names.values())
seg_class_indices = set(result.names.keys())
print("seg_classes:", seg_classes)
print("seg_class_indices:", seg_class_indices)



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

    # Mask of all instances of each detected class
    for i in seg_class_indices:
        if i in clss:
            seg_class = seg_classes[i]
            obj_indices = torch.where(clss == i)
            obj_masks = masks[obj_indices]
            obj_mask = torch.any(obj_masks, dim=0).int() * 255
            cv2.imwrite(f'./test_output/{seg_class}s.jpg', obj_mask.cpu().numpy())

            # Mask for each instance of a class
            for j, obj_index in enumerate(obj_indices[0].numpy()):
                obj_masks = masks[torch.tensor([obj_index])]
                obj_mask = torch.any(obj_masks, dim=0).int() * 255
                cv2.imwrite(f'./test_output/{seg_class}_{j}.jpg', obj_mask.cpu().numpy())
