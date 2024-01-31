from PySide6.QtCore import QThread, Signal
from core.video_processing import process_video

class VideoProcessingThread(QThread):
    frame_processed = Signal(object)  # Signal to send a frame to the GUI
    finished = Signal()

    def __init__(self, widget_instance):
        super().__init__()
        self.widget_instance = widget_instance
        self.stopped = False

    def run(self):
        # Start the video processing and send frames to the GUI as they are processed
        print("selected_video_source: ",self.widget_instance.selected_video_source,type(self.widget_instance.selected_video_source))
        print("selected_video_file: ",self.widget_instance.selected_video_file,type(self.widget_instance.selected_video_file))
        print("selected_live_video_input: ",self.widget_instance.selected_live_video_input,type(self.widget_instance.selected_live_video_input))
        print("selected_detection_model: ",self.widget_instance.selected_detection_model,type(self.widget_instance.selected_detection_model))
        print("selected_pixel_size: ",self.widget_instance.selected_pixel_size,type(self.widget_instance.selected_pixel_size))
        print("selected_tracker: ",self.widget_instance.selected_tracker,type(self.widget_instance.selected_tracker))
        print("selected_confidence: ",self.widget_instance.selected_confidence,type(self.widget_instance.selected_confidence))
        print("selected_iou: ",self.widget_instance.selected_iou,type(self.widget_instance.selected_iou))

        process_video(self.widget_instance, frame_callback=self.on_frame_processed)
        
        self.finished.emit()  # Emit the finished signal when the thread completes

    def stop(self):
        # Signal the thread to stop gracefully
        self.stopped = True

    def on_frame_processed(self, frame):
        # Emit the frame_processed signal if the thread is not stopped
        if not self.stopped:
            self.frame_processed.emit(frame)