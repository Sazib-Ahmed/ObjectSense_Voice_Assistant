from collections import defaultdict
import cv2
import numpy as np
import time
from ultralytics import YOLO
from mysql.connector import connect, Error
from datetime import datetime

# Initialize the YOLOv8 model
model = YOLO('yolov8n.pt')

# Set the video source: 0 for webcam, or the path to a video file
video_path = 0
cap = cv2.VideoCapture(video_path)

# Create a defaultdict to store the tracking history
track_history = defaultdict(lambda: [])

# Variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0

# Define the list of moving object IDs
moving_object_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79
]
# Define the list of stationary object IDs
stationary_object_ids = [13, 56, 57, 58, 59, 60, 61, 62, 68, 69, 70, 71, 72]


# Dictionary to hold the bounding boxes of stationary objects
stationary_objects_bboxes = {}

# Try to establish a connection to the MySQL database
try:
    db = connect(
        host="localhost",
        user="root",
        password="",
        database="yolo"
    )
    cursor = db.cursor()
except Error as e:
    print(f"Error connecting to MySQL: {e}")
    # Depending on the nature of your script, you may want to exit if the database connection is not successful
    exit(1)

# Define a function to determine the spatial relationship based on bounding boxes
def determine_spatial_relationship(moving_obj_box, stationary_obj_box):
    # Implement the spatial relationship logic as discussed earlier
    # Placeholder for the actual spatial relationship determination
    mx, my, mw, mh = moving_obj_box  # Moving object bounding box
    sx, sy, sw, sh = stationary_obj_box  # Stationary object bounding box

    # Check if the bounding boxes intersect
    intersect = not (mx + mw < sx or mx > sx + sw or my + mh < sy or my > sy + sh)

    if intersect:
        # Define a simple rule for "on" - if the bottom of the moving object is near the top of the stationary
        if my + mh - sy < 10:  # 10 is a threshold value that you can adjust
            return "on"
    return "not near"

# Process the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if not success:
        print("Failed to grab frame")
        break

    new_frame_time = time.time()

    try:
       # Run YOLOv8 detection on the frame
        detections = model(frame)

        # Print the type and structure of the detections object
        detections = model(frame)
        print(type(detections), detections)



        # Attempt to detect stationary objects and get their bounding boxes
        for det in detections.pred[0]:
            class_id = int(det[5])  # Index 5 should be the class ID
            if class_id in stationary_object_ids:
                bbox = det[:4].cpu().numpy()  # Get the bounding box
                stationary_objects_bboxes[class_id] = bbox  # Store the bbox with class_id as key

    except Exception as e:
        print(f"An error occurred during object detection: {e}")

    try:
        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True)

        # Check if detections and track IDs are present
        if results.pred is not None and len(results.pred):
            boxes = results.pred[0][:, :4].cpu().numpy()  # Bounding boxes
            class_ids = results.pred[0][:, 5].cpu().numpy().astype(int)  # Class IDs
            track_ids = results.pred[0][:, 6].cpu().numpy().astype(int)  # Track IDs

            # Visualize the results on the frame
            annotated_frame = np.copy(frame)

            # Update the tracking history and draw tracking lines
            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                if class_id in moving_object_ids:
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((x + w / 2, y + h / 2))  # Center point (x, y)
                    if len(track) > 30:  # Limit track history to 30 points
                        track.pop(0)

                    # Draw the tracking lines on the frame
                    points = np.array(track).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                    # Check spatial relationship with each detected stationary object
                    for stationary_id, stationary_bbox in stationary_objects_bboxes.items():
                        location_description = determine_spatial_relationship(box, stationary_bbox)
                        cursor.execute("SELECT tracker_id FROM detections WHERE tracker_id = %s", (track_id,))
                        existing_record = cursor.fetchone()

                        if existing_record:
                            cursor.execute(
                                "UPDATE detections SET class_id = %s, location_description = %s, updated_at = %s WHERE tracker_id = %s",
                                (class_id, location_description, datetime.now(), track_id)
                            )
                        else:
                            cursor.execute(
                                "INSERT INTO detections (tracker_id, class_id, location_description, created_at, updated_at) VALUES (%s, %s, %s, %s, %s)",
                                (track_id, class_id, location_description, datetime.now(), datetime.now())
                            )
                        db.commit()

            # Calculate and display the FPS
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

            # Show the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("No detections or track IDs in this frame.")

    except Exception as e:
        print(f"An error occurred during tracking: {e}")

# Release resources
cap.release()
cv2.destroyAllWindows()
cursor.close()
db.close()
