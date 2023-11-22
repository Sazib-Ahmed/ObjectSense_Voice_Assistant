from collections import defaultdict
import cv2
import numpy as np
import time
from ultralytics import YOLO
from mysql.connector import connect, Error
from datetime import datetime
import threading
from tkinter import Tk, Button, Label
import PIL.Image
import PIL.ImageTk


# Global variables
running = False
frame_to_display = None
video_frame_label = None

# class names present in the model
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# Define moving object IDs
moving_object_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79
]

# Define the list of stationary object IDs
stationary_object_ids = [13, 56, 57, 58, 59, 60, 61, 62, 68, 69, 70, 71, 72]



# Function to process video
def process_video():
    global running, frame_to_display

    # Load the YOLOv8 model
    model = YOLO('yolov8l.pt')

    # Open the video file or webcam
    video_path = 0
    cap = cv2.VideoCapture(video_path)

    # Store the track history and for FPS calculation
    track_history = defaultdict(lambda: [])
    prev_frame_time = 0

    # Connect to the database
    try:
        db = connect(
            host="localhost",
            user="root",
            password="",
            database="assistant"
        )
        cursor = db.cursor()
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return

    while running:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        new_frame_time = time.time()

        # Run YOLOv8 tracking on the frame
        results = model.track(frame, persist=True)

        # Initialize annotated_frame with the original frame
        annotated_frame = frame.copy()

        # Check if detections and track IDs are present
        if results[0].boxes is not None and getattr(results[0].boxes, 'id', None) is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            # Process each detection and update annotated_frame
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x, y, w, h = box
                if class_id in moving_object_ids:
                    x_value = float(x)
                    y_value = float(y)
                    w_value = float(w)
                    h_value = float(h)
                    class_name = class_names[class_id]

                    try:
                        # Check if the tracker_id already exists in the database
                        cursor.execute("SELECT * FROM detections WHERE tracker_id = %s", (track_id,))
                        existing_record = cursor.fetchone()

                        if existing_record:
                            # Update the existing record
                            cursor.execute(
                                "UPDATE detections SET class_id = %s, class_name = %s, x = %s, y = %s, width = %s, height = %s, timestamp = %s WHERE tracker_id = %s",
                                (class_id, class_name, x_value, y_value, w_value, h_value, datetime.now(), track_id)
                            )
                        else:
                            # Insert a new record
                            cursor.execute(
                                "INSERT INTO detections (tracker_id, class_id, class_name, x, y, width, height, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                                (track_id, class_id, class_name, x_value, y_value, w_value, h_value, datetime.now())
                            )
                        db.commit()
                    except Error as e:
                        print(f"Error interacting with database: {e}")

            # # Calculate and display the FPS
            # fps = 1 / (new_frame_time - prev_frame_time)
            # prev_frame_time = new_frame_time
            # cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

            # # Show the annotated frame
            # cv2.imshow("YOLOv8 Tracking", annotated_frame)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Update the frame to display
        frame_to_display = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Calculate and display the FPS
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

    cap.release()
    if cursor and db:
        cursor.close()
        db.close()

# Function to update video frame in the UI
def update_video_frame():
    global frame_to_display, video_frame_label
    if frame_to_display is not None:
        # Convert to a format Tkinter can use
        img = PIL.Image.fromarray(frame_to_display)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        video_frame_label.imgtk = imgtk
        video_frame_label.configure(image=imgtk)
    root.after(10, update_video_frame)  # Update the frame every 10 ms

# Function to handle start/stop button
def toggle_video():
    global running
    running = not running
    if running:
        threading.Thread(target=process_video, daemon=True).start()
        start_stop_btn.config(text='Stop')
    else:
        start_stop_btn.config(text='Start')

# Create UI
root = Tk()
root.title("Video Tracking Control")
video_frame_label = Label(root)
video_frame_label.pack()
start_stop_btn = Button(root, text="Start", command=toggle_video)
start_stop_btn.pack(pady=20)
root.after(10, update_video_frame)  # Start the update loop for the video frame

# Start the UI loop
root.mainloop()