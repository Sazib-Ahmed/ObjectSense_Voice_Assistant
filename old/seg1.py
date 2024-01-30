import cv2
import time
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')

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
        results = model(frame)

        # Get the masks of segmented objects
        masks = results.pred[0][:, 6:7]

        # Display the masks on the frame
        for mask in masks:
            mask = mask.squeeze().cpu().numpy() * 255
            mask = mask.astype(np.uint8)
            cv2.imshow("Segmented Object Mask", mask)

        # Calculate the FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {fps:.2f}"

        # Display FPS on the frame
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # Display the original frame
        cv2.imshow("YOLOv8 Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
