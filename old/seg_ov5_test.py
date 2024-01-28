from queue import Queue
import threading
import cv2
from ultralytics import YOLO

def run_tracker_in_thread(cam_id, model, stream_id, frame_queue, stop_event):
    video = cv2.VideoCapture(cam_id)
    frame_skip = 5  # Skip every 5 frames to reduce processing
    frame_count = 0

    while not stop_event.is_set():
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        frame_queue.put((stream_id, res_plotted))
    
    video.release()

def display_frames(frame_queue, stop_event):
    while not stop_event.is_set():
        if not frame_queue.empty():
            stream_id, frame = frame_queue.get()
            if frame is None:
                break
            if stream_id == 2:  # Display only stream 2
                cv2.imshow(f"Tracking_Stream_{stream_id}", frame)
                if cv2.waitKey(1) == ord('q'):
                    stop_event.set()

    cv2.destroyAllWindows()

# Main thread setup
frame_queue = Queue()
stop_event = threading.Event()
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n-seg.pt')

# Thread setup
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(0, model1, 1, frame_queue, stop_event), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(1, model2, 2, frame_queue, stop_event), daemon=True)

# Start threads
tracker_thread1.start()
tracker_thread2.start()

# Display frames in main thread
display_frames(frame_queue, stop_event)

# Wait for threads to finish
tracker_thread1.join()
tracker_thread2.join()
