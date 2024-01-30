
from PySide6.QtCore import QThread, Signal, QTimer, QLocale, QObject, Slot, QDateTime, Qt
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QTextBrowser, QPushButton, QMainWindow, QLabel
from core.assistant import start_assistant
class AssistantWorkerThread(QObject):
    text_signal = Signal(str, bool)  # Add a boolean flag to indicate if it's a stop message

    def __init__(self, parent=None):
        super(AssistantWorkerThread, self).__init__(parent)
        self.is_running = True
        self.show_date = True

    @Slot()
    def generate_and_emit_text(self):
        import random
        import string
        timestamp_format = "h:mm:ss AP"
        # if self.is_running:
        start_assistant(self,self.is_running)
        # while self.is_running:
        #     # text = "".join(random.choices(string.ascii_letters, k=10))
        #     if self.show_date:
        #         timestamp_format1 = "MMMM dd, yyyy :"
        #         timestamp = QDateTime.currentDateTime().toString(timestamp_format1)
        #         self.text_signal.emit(f"{timestamp}\n-------------------", False)
        #     timestamp = QDateTime.currentDateTime().toString(timestamp_format)
        #     # message = f"{timestamp}: {text}"
        #     message = f"{timestamp}:"
        #     self.text_signal.emit(message, False)
        #     QThread.msleep(1000)  # Sleep for 1 second
        #     self.show_date = False  # Set to False after the first message
        # else:
        #     self.text_signal.emit("",True) 
        #     self.text_signal.emit("=====================", True) 
        #     self.text_signal.emit("Assistant Stopped.", True)  # True indicates it's a stop message
        #     self.text_signal.emit("=====================",True) 
        #     self.text_signal.emit("", True) 


class WorkerThread(QThread):
    def __init__(self, worker):
        super(WorkerThread, self).__init__()
        self.worker = worker

    def run(self):
        self.worker.generate_and_emit_text()