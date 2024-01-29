
from core.assistant import start_assistant
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

class AssistantThread(QThread):
    respond_processed = Signal(object)
    finished = Signal()

    def __init__(self, widget_instance):
        super().__init__()
        self.widget_instance = widget_instance
        self.stopped = False

    def run(self):
        start_assistant(self.widget_instance, frame_callback=self.on_respond_processed)
        self.finished.emit()

    def stop(self):
        self.stopped = True

    def on_respond_processed(self, frame):
        if not self.stopped:
            # Emit the signal to update the GUI in the main thread
            self.respond_processed.emit(frame)






# from PySide6.QtCore import QObject, Signal, Qt, QMetaObject, Q_ARG

# # ... (your existing code)

# class AssistantThread(QObject):
#     update_signal = Signal(str)

#     def update_assistant_conversation(self, text):
#         # Use invokeMethod to update the GUI from the main thread
#         QMetaObject.invokeMethod(self, "update_assistant_conversation_gui", Qt.QueuedConnection, Q_ARG(str, text))

#     def update_assistant_conversation_gui(self, text):
#         self.update_signal.emit(text)

# # Inside Widget class
# def __init__(self):
#     # ... (existing code)

#     # Create an instance of AssistantUpdater
#     self.assistant_updater = AssistantThread()

#     # Connect the signal to the slot for updating the GUI
#     self.assistant_updater.update_signal.connect(self.update_assistant_conversation)
