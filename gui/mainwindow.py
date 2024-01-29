# gui/mainwindow.py

from PySide6.QtWidgets import QMainWindow, QApplication

from .widget import Widget

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Set up the main window
        self.setWindowTitle("Your Assistant GUI")
        self.setGeometry(100, 100, 800, 700)

        # Create the central widget and set the layout
        central_widget = Widget()
        self.setCentralWidget(central_widget)
        self.statusBar().showMessage("Running",5000)

    def on_button_click(self):
        self.label.setText("Button Clicked!")