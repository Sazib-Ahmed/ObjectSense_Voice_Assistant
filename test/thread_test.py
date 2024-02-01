from PySide6.QtCore import QThread, Signal, QObject, Slot
from PySide6.QtWidgets import QApplication, QMainWindow

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

class TextGenerator(QObject):
    text_signal = Signal(str, bool)

    def __init__(self):
        super(TextGenerator, self).__init__()
        self.is_running = True
        self.show_date = True

    @Slot()
    def generate_and_emit_text(self):
        import random
        import string
        from PySide6.QtCore import QDateTime, QThread

        timestamp_format = "h:mm:ss AP"
        while self.is_running:
            text = "".join(random.choices(string.ascii_letters, k=10))
            if self.show_date:
                timestamp_format1 = "MMMM dd, yyyy :"
                timestamp = QDateTime.currentDateTime().toString(timestamp_format1)
                self.text_signal.emit(f"{timestamp}\n-------------------", False)
            timestamp = QDateTime.currentDateTime().toString(timestamp_format)
            message = f"{timestamp}: {text}"
            self.text_signal.emit(message, False)
            QThread.msleep(1000)  # Sleep for 1 second
            self.show_date = False  # Set to False after the first message
        else:
            self.text_signal.emit("", True)
            self.text_signal.emit("=====================", True)
            self.text_signal.emit("Text generation stopped.", True)
            self.text_signal.emit("=====================", True)
            self.text_signal.emit("", True)

class MyMainWindow(QMainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()

        # Create an instance of AssistantWorkerThread
        self.assistant_worker = AssistantWorkerThread()

        # Create an instance of WorkerThread and start it
        self.worker_thread = WorkerThread(self.assistant_worker)
        self.worker_thread.start()

        # Connect the AssistantWorkerThread signal to a slot in this class
        self.assistant_worker.text_signal.connect(self.handle_text_signal)

    @Slot(str, bool)
    def handle_text_signal(self, text, is_stopped):
        # Do something with the emitted text signal
        print(text)
        if is_stopped:
            # Optionally handle the stop signal
            print("Text generation stopped")

if __name__ == "__main__":
    app = QApplication([])
    main_window = MyMainWindow()
    main_window.show()
    app.exec()
