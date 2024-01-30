from PySide6.QtCore import QThread, Signal, QTimer, Qt, QObject, Slot
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextBrowser

class Worker(QObject):
    text_signal = Signal(str)

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.is_running = True

    @Slot()
    def generate_and_emit_text(self):
        import random
        import string
        while self.is_running:
            text = "".join(random.choices(string.ascii_letters, k=10))
            self.text_signal.emit(text)
            QThread.msleep(1000)  # Sleep for 1 second

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

        self.quit_button = QPushButton("Quit", self)
        self.quit_button.clicked.connect(self.stop_thread)

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_thread)

        layout = QVBoxLayout()
        layout.addWidget(self.text_browser)
        layout.addWidget(self.quit_button)
        layout.addWidget(self.start_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.worker = Worker()
        self.worker_thread = WorkerThread(self.worker)
        self.worker.text_signal.connect(self.update_label)

    def update_label(self, text):
        current_text = self.text_browser.toPlainText()
        new_text = f"{current_text}\n{text}"
        self.text_browser.setPlainText(new_text)
        # Scroll to the bottom
        scrollbar = self.text_browser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def stop_thread(self):
        self.worker.is_running = False

    def start_thread(self):
        if not self.worker_thread.isRunning():
            self.worker.is_running = True
            self.worker_thread.start()

app = QApplication([])
window = MainWindow()
window.show()
app.exec()
