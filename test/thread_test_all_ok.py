from PySide6.QtCore import QThread, Signal, QTimer, QLocale, QObject, Slot, QDateTime, Qt
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QTextBrowser, QPushButton, QMainWindow, QLabel

class Worker(QObject):
    text_signal = Signal(str, bool)  # Add a boolean flag to indicate if it's a stop message

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.is_running = True
        self.show_date = True

    @Slot()
    def generate_and_emit_text(self):
        import random
        import string
        timestamp_format = "h:mm:ss AP"
        while self.is_running:
            text = "".join(random.choices(string.ascii_letters, k=10))
            if self.show_date:
                timestamp_format1 = "MMMM dd, yyyy :"
                timestamp = QDateTime.currentDateTime().toString(timestamp_format1)
                self.text_signal.emit(f"{timestamp}\n\n-------------------", False)
            timestamp = QDateTime.currentDateTime().toString(timestamp_format)
            message = f"{timestamp}: {text}"
            self.text_signal.emit(message, False)
            QThread.msleep(1000)  # Sleep for 1 second
            self.show_date = False  # Set to False after the first message
        else:
            self.text_signal.emit("\n=====================", True) 
            self.text_signal.emit("Text generation stopped.", True)  # True indicates it's a stop message
            self.text_signal.emit("=====================\n",True)


class WorkerThread(QThread):
    def __init__(self, worker):
        super(WorkerThread, self).__init__()
        self.worker = worker

    def run(self):
        self.worker.generate_and_emit_text()

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.text_browser = QTextBrowser(self)
        self.text_browser.setAlignment(Qt.AlignBottom)

        self.status_label = QLabel(self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

        self.toggle_button = QPushButton("Start", self)
        self.toggle_button.clicked.connect(self.toggle_thread)

        layout = QVBoxLayout()
        layout.addWidget(self.text_browser)
        layout.addWidget(self.status_label)
        layout.addWidget(self.toggle_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.worker = Worker()
        self.worker_thread = WorkerThread(self.worker)
        self.worker.text_signal.connect(self.update_label)

    def update_label(self, message, is_stop_message):
        current_text = self.text_browser.toPlainText()
        new_text = f"{current_text}\n{message}"
        self.text_browser.setPlainText(new_text)
        # Scroll to the bottom
        scrollbar = self.text_browser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        if is_stop_message:
            self.status_label.setText("Text generation stopped.")
        else:
            self.status_label.clear()

    def toggle_thread(self):
        if self.worker_thread.isRunning():
            self.worker.is_running = False
            self.worker_thread.wait()
            self.toggle_button.setText("Start")
        else:
            self.worker.is_running = True
            self.worker.show_date = True  # Reset to True when starting
            self.worker_thread.start()
            self.toggle_button.setText("Quit")

app = QApplication([])
window = MainWindow()
window.show()
app.exec()