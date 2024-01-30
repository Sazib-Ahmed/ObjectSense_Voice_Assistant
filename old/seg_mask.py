from ultralytics import YOLO
import cv2
import torch
from pathlib import Path

# Load a pretrained YOLOv8n-seg Segment model
model = YOLO('yolov8n-seg.pt')

# Run inference on an image
results = model('bus.jpg')  # results list

result = results[0]

print(result.names)
# print(result.boxes.xyxy)
# print(result.boxes.conf)
# print(result.boxes.cls)
# print(result.masks.data)

Path("./test_output/").mkdir(parents=True, exist_ok=True)

cv2.imwrite(f"./test_output/original_image.jpg", result.orig_img)

seg_classes = list(result.names.values())
# seg_classes = ["door", "insulator", "wall", "window"]

for result in results:

    masks = result.masks.data
    boxes = result.boxes.data

    clss = boxes[:, 5]
    print("clss")
    print(clss)

    #EXTRACT A SINGLE MASK WITH ALL THE CLASSES
    obj_indices = torch.where(clss != -1)
    obj_masks = masks[obj_indices]
    obj_mask = torch.any(obj_masks, dim=0).int() * 255
    cv2.imwrite(str(f'./test_output/all-masks.jpg'), obj_mask.cpu().numpy())

    #MASK OF ALL INSTANCES OF A CLASS
    for i, seg_class in enumerate(seg_classes):

        obj_indices = torch.where(clss == i)
        print("obj_indices")
        print(obj_indices)
        obj_masks = masks[obj_indices]
        obj_mask = torch.any(obj_masks, dim=0).int() * 255

        cv2.imwrite(str(f'./test_output/{seg_class}s.jpg'), obj_mask.cpu().numpy())

        #MASK FOR EACH INSTANCE OF A CLASS
        for i, obj_index in enumerate(obj_indices[0].numpy()):
            obj_masks = masks[torch.tensor([obj_index])]
            obj_mask = torch.any(obj_masks, dim=0).int() * 255
            cv2.imwrite(str(f'./test_output/{seg_class}_{i}.jpg'), obj_mask.cpu().numpy())