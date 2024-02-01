# text_generator.py

from PySide6.QtCore import QDateTime, Signal, QObject, Slot, QThread

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