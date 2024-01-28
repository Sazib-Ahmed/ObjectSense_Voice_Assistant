import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Set up the main window
        self.setWindowTitle("Your Assistant GUI")
        self.setGeometry(100, 100, 800, 600)

        # Create the central widget and set the layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Add widgets to the layout
        self.label = QLabel("Welcome to Your Assistant GUI", self)
        layout.addWidget(self.label)

        self.button = QPushButton("Click Me", self)
        self.button.clicked.connect(self.on_button_click)
        layout.addWidget(self.button)

    def on_button_click(self):
        self.label.setText("Button Clicked!")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
