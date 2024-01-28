from queue import Queue
import threading
import cv2
from ultralytics import YOLO

def run_tracker_in_thread(filename, model, file_count, frame_queue):
    video = cv2.VideoCapture(filename)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        frame_queue.put((file_count, res_plotted))
    
    video.release()

def display_frames(frame_queue):
    while True:
        if not frame_queue.empty():
            file_count, frame = frame_queue.get()
            if frame is None:
                break
            cv2.imshow(f"Tracking_Stream_{file_count}", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()

# Main thread setup
frame_queue = Queue()
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n-seg.pt')
video_file1 = 0  # For example
video_file2 = 0  # For example

# Thread setup
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 1, frame_queue), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2, frame_queue), daemon=True)

# Start threads
tracker_thread1.start()
tracker_thread2.start()

# Display frames in main thread
display_frames(frame_queue)

# Wait for threads to finish
tracker_thread1.join()
tracker_thread2.join()
