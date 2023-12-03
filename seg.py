import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO

# Function to check if two masks overlap
def masks_overlap(mask1, mask2):
    # Create blank images to draw masks
    img1 = np.zeros((1080, 1920), dtype=np.uint8)
    img2 = np.zeros((1080, 1920), dtype=np.uint8)

    # Draw masks on blank images
    cv2.fillPoly(img1, [mask1], 255)
    cv2.fillPoly(img2, [mask2], 255)

    # Check for overlap
    overlap = np.logical_and(img1, img2)
    return np.any(overlap)

# Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')
#model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

# Loop through the video frames
while cap.isOpened():
    # Capture the time before processing the frame
    new_frame_time = time.time()

    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        #results = model.track(frame, conf=0.20, imgsz=640, iou=0.45, half=False)
        # Run YOLOv8 inference on the frame with ByteTrack tracker
        results = model.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.50, imgsz=640, iou=0.50) 
        #results = model.track(source='video.mp4', conf=0.30, imgsz=800, iou=0.50, half=True, tracker='deepsort', device='cuda', visualize=True, save_results=True, save_path='results.avi', verbose=True)
        #results = model.track(source="path/to/video.mp4", show=True, tracker='botsort.yaml', conf=0.25, iou=0.45, max_det=1000, max_age=30, n_init=3, device='cuda')


        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Check if there are any detections and extract mask segments
        if results[0].masks is not None:
            all_masks = []

            for mask in results[0].masks.xy:  # Assuming mask segments are stored in the 'xy' attribute
                # Convert segments to numpy arrays
                np_mask = np.array(mask, dtype=np.int32)
                all_masks.append(np_mask)

            # Check for overlapping masks
            for i, mask1 in enumerate(all_masks):
                for j, mask2 in enumerate(all_masks):
                    if i != j and masks_overlap(mask1, mask2):
                        print(mask1)
                        print(mask2)
                        print(f"Objects {i} and {j} are overlapping.")

        # Calculate the FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {fps:.2f}"

        # Display FPS on the frame
        cv2.putText(annotated_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
