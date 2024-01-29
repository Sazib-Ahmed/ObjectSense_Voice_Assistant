# gui/widget.py
import cv2 
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox,QComboBox,QRadioButton, QFileDialog, QFrame, QWidget, QLabel, QVBoxLayout, QPushButton, QTabWidget, QLineEdit, QHBoxLayout, QSizePolicy, QGridLayout

class Widget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ObjectSense Voice Assistant")

        tab_widget = QTabWidget(self)

        # Application
        application_widget = QWidget()

        # Detection Grid
        detection_grid_layout = QGridLayout()
        detection_label = QLabel("Object Detection System")
        detection_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        detection_label.setAlignment(Qt.AlignCenter)
        detection_grid_layout.addWidget(detection_label, 0, 0, 1, 3)  # Set column span to 3

        detection_video_label = QLabel("Video")
        detection_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        detection_video_label.setAlignment(Qt.AlignCenter)
        detection_grid_layout.addWidget(detection_video_label, 1, 0, 4, 4) 

        # Radio Buttons for Video Source
        self.video_file_radio = QRadioButton("Video File:")
        self.live_video_radio = QRadioButton("Live Video:")
        # Set "Live Video" as default selected
        self.live_video_radio.setChecked(True)

        # Connect signals to slots
        self.video_file_radio.toggled.connect(self.toggle_video_source)
        self.live_video_radio.toggled.connect(self.toggle_video_source)

        # File Input for Video File
        self.file_input = QLineEdit()
        file_browse_button = QPushButton("Browse")
        file_browse_button.clicked.connect(self.browse_file)
        self.file_browse_button = file_browse_button

        # Available Video Inputs for Live Video
        self.video_inputs = self.get_available_video_inputs()
        self.video_input_combobox = QComboBox()
        self.video_input_combobox.addItems(self.video_inputs)

        # Set up layouts for input options
        file_input_layout = QHBoxLayout()
        file_input_layout.addWidget(self.file_input)
        file_input_layout.addWidget(file_browse_button)

        # Set the first option as default
        if self.video_inputs:
            self.video_input_combobox.setCurrentIndex(0)


        detection_model_label = QLabel("YOLOv8 Model Size:")
        detection_model_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        detection_model_label.setAlignment(Qt.AlignCenter)
        self.detection_model_combobox = QComboBox()
        self.detection_model_combobox.addItems(["Nano", "Small", "Medium", "Large", "Extra Large"])

        # Set the default value to "Medium"
        default_index = self.detection_model_combobox.findText("Medium")
        self.detection_model_combobox.setCurrentIndex(default_index)

        detection_video_res_label = QLabel("Pixel Size:")
        detection_video_res_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        detection_video_res_label.setAlignment(Qt.AlignCenter)
        original_resolution = (1920, 1080)  # Replace this with the actual resolution of your video

        self.detection_video_res_combobox = QComboBox()

        # Calculate multiples of 32 for the height only
        resolutions = [str(j) for j in range(32, original_resolution[1] + 1, 32)]

        self.detection_video_res_combobox.addItems(resolutions)
        # Set the default value to 640
        default_res = resolutions.index("640")
        self.detection_video_res_combobox.setCurrentIndex(default_res)

        detection_tracker_label = QLabel("Tracker:")
        detection_tracker_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        detection_tracker_label.setAlignment(Qt.AlignCenter)

        #Radio buttons : answers
        answers = QGroupBox("Choose Answer")
        answer_a = QRadioButton("A")
        answer_b = QRadioButton("B")
        answer_c = QRadioButton("C")
        answer_a.setChecked(True)

        answers_layout = QHBoxLayout()
        answers_layout.addWidget(answer_a)
        answers_layout.addWidget(answer_b)
        answers_layout.addWidget(answer_c)
        answers.setLayout(answers_layout)






        

        # Add layouts to detection grid
        detection_grid_layout.addWidget(self.video_file_radio, 5, 0)
        detection_grid_layout.addLayout(file_input_layout, 5, 1, 1, 3)
        detection_grid_layout.addWidget(self.live_video_radio, 6, 0)
        detection_grid_layout.addWidget(self.video_input_combobox, 6, 1, 1, 3)
        detection_grid_layout.addWidget(detection_model_label, 7, 0)
        detection_grid_layout.addWidget(self.detection_model_combobox, 7, 1, 1, 3)
        detection_grid_layout.addWidget(detection_video_res_label, 8, 0)
        detection_grid_layout.addWidget(self.detection_video_res_combobox, 8, 1, 1, 3)
        detection_grid_layout.addWidget(detection_tracker_label, 9, 0)
        detection_grid_layout.addWidget(answers, 9, 1, 1, 3)









        # Vertical Separator Line
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.VLine)
        separator_line.setFrameShadow(QFrame.Sunken)
        separator_line.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Assistant Grid
        assistant_grid_layout = QGridLayout()
        assistant_label = QLabel("Assistant")
        assistant_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        assistant_label.setAlignment(Qt.AlignCenter)
        assistant_grid_layout.addWidget(assistant_label, 0, 0, 1, 3)  # Set column span to 3

        assistant_conversation_label = QLabel("Conversation")
        assistant_conversation_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        assistant_conversation_label.setAlignment(Qt.AlignCenter)
        assistant_grid_layout.addWidget(assistant_conversation_label, 1, 0, 4, 4)

        # Application Layout with Separator
        application_layout = QHBoxLayout()
        application_layout.addLayout(detection_grid_layout)
        application_layout.addWidget(separator_line)
        application_layout.addLayout(assistant_grid_layout)
        application_widget.setLayout(application_layout)

        # Add tabs to widget
        tab_widget.addTab(application_widget, "Application")

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)

    def toggle_video_source(self):
        is_file_selected = self.sender().text() == "Video File"
        self.file_input.setEnabled(is_file_selected)
        self.file_browse_button.setEnabled(is_file_selected)  # Disable the Browse button when Video File is selected
        self.video_input_combobox.setEnabled(not is_file_selected)

        # Disable the other input option
        if is_file_selected:
            self.live_video_radio.setChecked(False)
        else:
            self.video_file_radio.setChecked(False)



    def browse_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            self.file_input.setText(file_path)

    def get_available_video_inputs(self):
        # Use OpenCV to get available video inputs (webcams)
        available_inputs = []
        for i in range(10):  # Check up to 10 video inputs
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_inputs.append(f"Camera {i}")
                cap.release()
        return available_inputs