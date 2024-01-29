from PySide6.QtCore import QThread, Signal
from core.video_processing import process_video

class VideoProcessingThread(QThread):
    frame_processed = Signal(object)  # Signal to send a frame to the GUI
    finished = Signal()

    def __init__(self, widget_instance, video_path):
        super().__init__()
        self.widget_instance = widget_instance
        self.video_path = video_path

    def run(self):
        # Start the video processing and send frames to the GUI as they are processed
        process_video(self.video_path, self.widget_instance, frame_callback=self.frame_processed.emit)
        
        self.finished.emit()  # Emit the finished signal when the thread completes
