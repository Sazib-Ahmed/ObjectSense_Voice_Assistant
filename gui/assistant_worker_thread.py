
from PySide6.QtCore import QThread, Signal, QTimer, QLocale, QObject, Slot, QDateTime, Qt
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QTextBrowser, QPushButton, QMainWindow, QLabel
from core.text_generator import TextGenerator


class AssistantWorkerThread(QObject):
    text_signal = Signal(str, bool)  # Add a boolean flag to indicate if it's a stop message

    def __init__(self, parent=None):
        super(AssistantWorkerThread, self).__init__(parent)
        self.is_running = True
        self.show_date = True

class WorkerThread(QThread):
    def __init__(self, worker):
        super(WorkerThread, self).__init__()
        self.worker = worker

    def run(self):
        # Create an instance of TextGenerator
        text_generator = TextGenerator()

        # Connect the TextGenerator signal to the AssistantWorkerThread slot
        text_generator.text_signal.connect(self.worker.text_signal.emit)

        # Start the text generation
        text_generator.generate_and_emit_text()