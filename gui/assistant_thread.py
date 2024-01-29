

from PySide6.QtCore import QObject, Signal, Qt, QMetaObject, Q_ARG

# ... (your existing code)

class AssistantThread(QObject):
    update_signal = Signal(str)

    def update_assistant_conversation(self, text):
        # Use invokeMethod to update the GUI from the main thread
        QMetaObject.invokeMethod(self, "update_assistant_conversation_gui", Qt.QueuedConnection, Q_ARG(str, text))

    def update_assistant_conversation_gui(self, text):
        self.update_signal.emit(text)

# Inside Widget class
def __init__(self):
    # ... (existing code)

    # Create an instance of AssistantUpdater
    self.assistant_updater = AssistantThread()

    # Connect the signal to the slot for updating the GUI
    self.assistant_updater.update_signal.connect(self.update_assistant_conversation)
