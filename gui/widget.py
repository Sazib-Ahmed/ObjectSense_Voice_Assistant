# gui/widget.py
import cv2
from threading import Thread
import queue
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider,QGroupBox,QComboBox,QRadioButton, QFileDialog, QFrame, QWidget, QLabel, QVBoxLayout, QPushButton, QTabWidget, QLineEdit, QHBoxLayout, QSizePolicy, QGridLayout
from core.video_processing import process_video
from .video_processing_thread import VideoProcessingThread
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

        self.detection_video_display = QLabel("Video")
        self.detection_video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.detection_video_display.setAlignment(Qt.AlignCenter)
        detection_grid_layout.addWidget(self.detection_video_display, 1, 0, 4, 4) 

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
        # detection_model_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
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
        # answers = QGroupBox("Choose Answer")
        detection_tracker_bot = QRadioButton("BoT-SORT")
        detection_tracker_byte = QRadioButton("ByteTrack ")
        detection_tracker_bot.setChecked(True)

        detection_tracker_layout = QHBoxLayout()
        detection_tracker_layout.addWidget(detection_tracker_bot)
        detection_tracker_layout.addWidget(detection_tracker_byte)
        # answers.setLayout(answers_layout)

        # Slider for Confidence Threshold
        confidence_label = QLabel("Confidence Threshold:")
        confidence_slider = QSlider(Qt.Horizontal)
        confidence_slider.setMinimum(10)  # minimum value * 10
        confidence_slider.setMaximum(100)  # maximum value * 10
        confidence_slider.setValue(50)  # default value, 0.5 as percentage
        confidence_slider.setTickPosition(QSlider.TicksBelow)
        confidence_slider.setTickInterval(10)

        confidence_value_label = QLabel(str(confidence_slider.value() / 100.0))  # Initial value as string
        confidence_value_label.setFixedWidth(40)  # Set a fixed width to avoid length change
        confidence_slider.valueChanged.connect(lambda value: confidence_value_label.setText(str(value / 100.0)))

        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(confidence_slider)
        confidence_layout.addWidget(confidence_value_label)

        # Slider for IoU Threshold
        iou_label = QLabel("IoU Threshold:")
        iou_slider = QSlider(Qt.Horizontal)
        iou_slider.setMinimum(10)  # minimum value * 10
        iou_slider.setMaximum(100)  # maximum value * 10
        iou_slider.setValue(50)  # default value, 0.5 as percentage
        iou_slider.setTickPosition(QSlider.TicksBelow)
        iou_slider.setTickInterval(10)

        iou_value_label = QLabel(str(iou_slider.value() / 100.0))  # Initial value as string
        iou_value_label.setFixedWidth(40)  # Set a fixed width to avoid length change
        iou_slider.valueChanged.connect(lambda value: iou_value_label.setText(str(value / 100.0)))

        iou_layout = QHBoxLayout()
        iou_layout.addWidget(iou_slider)
        iou_layout.addWidget(iou_value_label)

        # Start/Stop Button
        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.clicked.connect(self.start_stop_process)
        self.process_running = False  # Variable to track process state

        # Set a professional-looking style sheet for the button
        self.start_stop_button.setStyleSheet("""
            QPushButton {
                color: white;
                border: 1px solid #4CAF50;
                border-radius: 4px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #45a049; /* Darker Green */
            }
            QPushButton[stopped="false"] {
                background-color: #FF0000; /* Red */
                border: 1px solid #FF0000;
            }
            QPushButton[stopped="false"]:hover {
                background-color: #D70000; /* Darker Red */
            }
            QPushButton[stopped="true"] {
                background-color: #4CAF50; /* Green */
                border: 1px solid #4CAF50;
            }
            QPushButton[stopped="true"]:hover {
                background-color: #45a049; /* Darker Green */
            }
        """)
        

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
        detection_grid_layout.addLayout(detection_tracker_layout, 9, 1, 1, 3)
        detection_grid_layout.addWidget(confidence_label, 10, 0)
        detection_grid_layout.addLayout(confidence_layout, 10, 1, 1, 3)
        detection_grid_layout.addWidget(iou_label, 11, 0)
        detection_grid_layout.addLayout(iou_layout, 11, 1, 1, 3)
        detection_grid_layout.addWidget(self.start_stop_button, 12, 0, 1, 4)  # Add start/stop button

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
        is_file_selected = self.sender().text() == "Video File:"
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
    

    # Inside Widget class
    def start_stop_process(self):
        if self.process_running:
            # If the process is running, stop it
            self.process_running = False
            self.start_stop_button.setProperty("stopped", "true")
            self.start_stop_button.setText("Start")
            print("Process Stopped")
            # Add logic to stop the process (replace print statement with your logic)
        else:
            # If the process is stopped, start it
            self.process_running = True
            self.start_stop_button.setProperty("stopped", "false")
            self.start_stop_button.setText("Stop")
            print("Process Started")
            self.video_thread = VideoProcessingThread(self, "./Demo04.mp4")
            self.video_thread.finished.connect(self.on_video_processing_finished)
            self.video_thread.start()
            
            # Add logic to start the process (replace print statement with your logic)

        # Update style to apply changes
        self.start_stop_button.style().polish(self.start_stop_button)

    def on_video_processing_finished(self):
        # This method is called when the video processing thread finishes
        print("Video Processing Finished")
        self.video_thread.wait()  # Wait for the thread to finish before allowing it to be destroyed
        self.video_thread.deleteLater()  # Delete the thread


    def display_video_frame(self, frame):
        """
        Display a video frame in the QLabel.
        Convert the OpenCV frame to a QPixmap and set it as the label's pixmap.
        """
        # Get the size of the QLabel
        label_size = self.detection_video_display.size()
        
        # Resize the frame to match the QLabel size
        frame = cv2.resize(frame, (label_size.width(), label_size.height()))
        
        # Convert the OpenCV frame to a QPixmap
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Set the QPixmap as the label's pixmap
        self.detection_video_display.setPixmap(pixmap)

