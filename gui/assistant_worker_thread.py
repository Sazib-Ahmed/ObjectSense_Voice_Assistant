
from PySide6.QtCore import QThread, Signal, QTimer, QLocale, QObject, Slot, QDateTime, Qt
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QTextBrowser, QPushButton, QMainWindow, QLabel
from core.assistant import start_assistant
class Worker(QObject):
    text_signal = Signal(str, bool)  # Add a boolean flag to indicate if it's a stop message

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.is_running = True
        self.show_date = True

    @Slot()
    def generate_and_emit_text(self):
        start_assistant(self,self.is_running)

class WorkerThread(QThread):
    def __init__(self, worker):
        super(WorkerThread, self).__init__()
        self.worker = worker

    def run(self):
        self.worker.generate_and_emit_text()