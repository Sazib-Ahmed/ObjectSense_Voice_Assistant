# gui/widget.py
import cv2
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QSlider, QGroupBox, QComboBox, QRadioButton, QFileDialog, QFrame,
    QWidget, QLabel, QVBoxLayout, QPushButton, QTabWidget, QLineEdit,
    QHBoxLayout, QSizePolicy, QGridLayout,QTextBrowser,QButtonGroup
)
from core.video_processing import process_video
from .video_processing_thread import VideoProcessingThread
from .assistant_worker_thread import Worker, WorkerThread

# from core.assistant import *
# from threading import Thread


class Widget(QWidget):
    def __init__(self):
        super().__init__()

        # Instance variables to store selected options
        self.selected_video_source = "live"  # To store the selected video source (file or live)
        self.selected_video_file = None  # To store the selected video file path
        self.selected_live_video_input = 0  # To store the selected live video input
        self.selected_detection_model = "../yolov8s-seg.pt"  # To store the selected YOLOv8 model size
        self.selected_pixel_size = 640  # To store the selected pixel size
        self.selected_tracker = "botsort.yaml"   # botsort.yaml/bytetrack.yaml To store the selected tracker (BoT-SORT or ByteTrack)
        self.selected_confidence = 0.25
        self.selected_iou = 0.7



        self.setWindowTitle("ObjectSense Voice Assistant")

        self.tab_widget = QTabWidget(self)

        # Application
        self.application_widget = QWidget()

        # Detection Grid
        self.detection_grid_layout = QGridLayout()
        self.detection_label = QLabel("Object Detection System")
        self.detection_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.detection_label.setAlignment(Qt.AlignCenter)
        self.detection_grid_layout.addWidget(self.detection_label, 0, 0, 1, 4)  # Set column span to 3

        self.detection_video_display = QLabel("Video")
            # Load the initial image using OpenCV
        self.initial_image = cv2.imread("assets/video_label.JPG")

        # Check if the image is loaded successfully
        if self.initial_image is not None:
            #Convert the OpenCV image to QPixmap
            height, width, channel = self.initial_image.shape
            # height = 480
            # width = 640

            bytes_per_line = 3 * width
            q_image = QImage(self.initial_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # Set the QPixmap as the label's pixmap and resize the label
            self.detection_video_display.setPixmap(pixmap)
            self.detection_video_display.setFixedSize(width, height)

            # Adjust the size policy
            self.detection_video_display.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        else:
            # If the image fails to load, print an error message
            print("Failed to load the initial image.")
        self.detection_video_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.detection_video_display.setAlignment(Qt.AlignCenter)
        self.detection_grid_layout.addWidget(self.detection_video_display, 1, 0, 4, 4)

        # Radio Buttons for Video Source
        self.video_file_radio = QRadioButton("Video File:")
        self.live_video_radio = QRadioButton("Live Video:")
        # Set "Live Video" as the default selected
        self.live_video_radio.setChecked(True)

        # Create button groups for video source and tracker options
        self.video_source_button_group = QButtonGroup(self)
        self.video_source_button_group.addButton(self.video_file_radio)
        self.video_source_button_group.addButton(self.live_video_radio)


        # Connect signals to slots
        self.video_file_radio.toggled.connect(self.toggle_video_source)
        self.live_video_radio.toggled.connect(self.toggle_video_source)

        # File Input for Video File
        self.file_input = QLineEdit()
        self.file_browse_button = QPushButton("Browse")
        self.file_browse_button.clicked.connect(self.browse_file)

        # Available Video Inputs for Live Video
        self.video_inputs = self.get_available_video_inputs()
        self.video_input_combobox = QComboBox()
        self.video_input_combobox.addItems(self.video_inputs)

        # Set up layouts for input options
        self.file_input_layout = QHBoxLayout()
        self.file_input_layout.addWidget(self.file_input)
        self.file_input_layout.addWidget(self.file_browse_button)

        # Set the first option as default
        if self.video_inputs:
            self.video_input_combobox.setCurrentIndex(0)

        self.detection_model_label = QLabel("YOLOv8 Model Size:")
        self.detection_model_label.setAlignment(Qt.AlignCenter)
        self.detection_model_combobox = QComboBox()
        self.detection_model_combobox.addItems(["Nano", "Small", "Medium", "Large", "Extra Large"])

        # Set the default value to "Medium"
        self.default_model_size= self.detection_model_combobox.findText("Small")
        self.detection_model_combobox.setCurrentIndex(self.default_model_size)

        self.detection_video_res_label = QLabel("Pixel Size:")
        self.detection_video_res_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.detection_video_res_label.setAlignment(Qt.AlignCenter)
        self.original_resolution = (1920, 1080)  # Replace this with the actual resolution of your video

        self.detection_video_res_combobox = QComboBox()

        # Calculate multiples of 32 for the height only
        self.resolutions = [str(j) for j in range(32, self.original_resolution[1] + 1, 32)]

        self.detection_video_res_combobox.addItems(self.resolutions)
        # Set the default value to 640
        self.default_res = self.resolutions.index("640")
        self.detection_video_res_combobox.setCurrentIndex(self.default_res)

        self.detection_tracker_label = QLabel("Tracker:")
        self.detection_tracker_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.detection_tracker_label.setAlignment(Qt.AlignCenter)

        # Radio buttons : answers
        self.detection_tracker_bot = QRadioButton("BoT-SORT")
        self.detection_tracker_byte = QRadioButton("ByteTrack ")
        self.detection_tracker_bot.setChecked(True)

        self.tracker_button_group = QButtonGroup(self)
        self.tracker_button_group.addButton(self.detection_tracker_bot)
        self.tracker_button_group.addButton(self.detection_tracker_byte)

        # Connect signals to slots for radio buttons
        self.video_file_radio.toggled.connect(self.toggle_video_source)
        self.live_video_radio.toggled.connect(self.toggle_video_source)

        self.detection_tracker_bot.toggled.connect(self.toggle_tracker)
        self.detection_tracker_byte.toggled.connect(self.toggle_tracker)


        self.detection_tracker_layout = QHBoxLayout()
        self.detection_tracker_layout.addWidget(self.detection_tracker_bot)
        self.detection_tracker_layout.addWidget(self.detection_tracker_byte)

        # Slider for Confidence Threshold
        self.confidence_label = QLabel("Confidence Threshold:")
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(10)  # minimum value * 10
        self.confidence_slider.setMaximum(100)  # maximum value * 10
        self.confidence_slider.setValue(25)  # default value, 0.5 as a percentage
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)

        self.confidence_value_label = QLabel(str(self.confidence_slider.value() / 100.0))  # Initial value as a string
        self.confidence_value_label.setFixedWidth(40)  # Set a fixed width to avoid length change
        self.confidence_slider.valueChanged.connect(lambda value: self.confidence_value_label.setText(str(value / 100.0)))

        self.confidence_layout = QHBoxLayout()
        self.confidence_layout.addWidget(self.confidence_slider)
        self.confidence_layout.addWidget(self.confidence_value_label)

        # Slider for IoU Threshold
        self.iou_label = QLabel("IoU Threshold:")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(10)  # minimum value * 10
        self.iou_slider.setMaximum(100)  # maximum value * 10
        self.iou_slider.setValue(70)  # default value, 0.5 as a percentage
        self.iou_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_slider.setTickInterval(10)

        self.iou_value_label = QLabel(str(self.iou_slider.value() / 100.0))  # Initial value as a string
        self.iou_value_label.setFixedWidth(40)  # Set a fixed width to avoid length change
        self.iou_slider.valueChanged.connect(lambda value: self.iou_value_label.setText(str(value / 100.0)))

        self.iou_layout = QHBoxLayout()
        self.iou_layout.addWidget(self.iou_slider)
        self.iou_layout.addWidget(self.iou_value_label)

        # Start/Stop Button
        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.clicked.connect(self.start_stop_process)
        self.process_running = False  # Variable to track process state

        # Set a professional-looking style sheet for the button
        self.start_stop_button.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #4CAF50; /* Green */
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

        # Add layouts to the detection grid
        self.detection_grid_layout.addWidget(self.video_file_radio, 5, 0)
        self.detection_grid_layout.addLayout(self.file_input_layout, 5, 1, 1, 3)
        self.detection_grid_layout.addWidget(self.live_video_radio, 6, 0)
        self.detection_grid_layout.addWidget(self.video_input_combobox, 6, 1, 1, 3)
        self.detection_grid_layout.addWidget(self.detection_model_label, 7, 0)
        self.detection_grid_layout.addWidget(self.detection_model_combobox, 7, 1, 1, 3)
        self.detection_grid_layout.addWidget(self.detection_video_res_label, 8, 0)
        self.detection_grid_layout.addWidget(self.detection_video_res_combobox, 8, 1, 1, 3)
        self.detection_grid_layout.addWidget(self.detection_tracker_label, 9, 0)
        self.detection_grid_layout.addLayout(self.detection_tracker_layout, 9, 1, 1, 3)
        self.detection_grid_layout.addWidget(self.confidence_label, 10, 0)
        self.detection_grid_layout.addLayout(self.confidence_layout, 10, 1, 1, 3)
        self.detection_grid_layout.addWidget(self.iou_label, 11, 0)
        self.detection_grid_layout.addLayout(self.iou_layout, 11, 1, 1, 3)
        self.detection_grid_layout.addWidget(self.start_stop_button, 12, 0, 1, 4)  # Add start/stop button

        # Vertical Separator Line
        self.separator_line = QFrame()
        self.separator_line.setFrameShape(QFrame.VLine)
        self.separator_line.setFrameShadow(QFrame.Sunken)
        self.separator_line.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Assistant Grid
        self.assistant_grid_layout = QGridLayout()
        self.assistant_label = QLabel("Assistant")
        self.assistant_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.assistant_label.setAlignment(Qt.AlignCenter)
        self.assistant_grid_layout.addWidget(self.assistant_label, 0, 0, 1, 4)  # Set column span to 3

        self.text_browser = QTextBrowser(self)
        self.text_browser.setAlignment(Qt.AlignBottom)

        self.status_label = QLabel(self)
        self.status_label.setText("")
        self.status_label.setAlignment(Qt.AlignCenter)
        # self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        self.assistant_grid_layout.addWidget(self.text_browser, 1, 0, 10, 4)
        self.assistant_grid_layout.addWidget(self.status_label, 11, 0, 1, 4)

        self.worker = Worker()
        self.worker_thread = WorkerThread(self.worker)
        self.worker.text_signal.connect(self.update_label)

        # Start/Stop Button
        self.assistant_start_stop_button = QPushButton("Start")
        self.assistant_start_stop_button.clicked.connect(self.assistant_start_stop_process)
        self.assistant_process_running = False  # Variable to track process state

        # Set a professional-looking style sheet for the button
        self.assistant_start_stop_button.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: #4CAF50; /* Green */
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

        self.assistant_grid_layout.addWidget(self.assistant_start_stop_button, 12, 0, 1, 4)  # Add start/stop button

        # Application Layout with Separator
        self.application_layout = QHBoxLayout()
        self.application_layout.addLayout(self.detection_grid_layout)
        self.application_layout.addWidget(self.separator_line)
        self.application_layout.addLayout(self.assistant_grid_layout)
        self.application_widget.setLayout(self.application_layout)

        # Add tabs to the widget
        self.tab_widget.addTab(self.application_widget, "ObjectSense Voice Assistant")

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tab_widget)

        self.setLayout(self.layout)

    def toggle_video_source(self):
        is_file_selected = self.video_file_radio.isChecked()
        self.file_input.setEnabled(is_file_selected)
        self.file_browse_button.setEnabled(is_file_selected)
        self.video_input_combobox.setEnabled(not is_file_selected)

        if is_file_selected:
            self.selected_video_source = "file"
        else:
            self.selected_video_source = "live"

    def toggle_tracker(self):
        if self.detection_tracker_bot.isChecked():
            self.selected_tracker = "botsort.yaml"
        else:
            self.selected_tracker = "bytetrack.yaml"

    # def browse_file(self):
    #     file_dialog = QFileDialog()
    #     file_path, _ = file_dialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.mov *.avi)")
    #     if file_path:
    #         self.file_input.setText(file_path)
    #         self.selected_video_file = file_path
    def browse_file(self):
        video_formats = [
            "Video Files (*.asf *.avi *.gif *.m4v *.mkv *.mov *.mp4 *.mpeg *.mpg *.ts *.wmv *.webm)",
            "All Files (*)"
        ]
        
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Video File", "", ";;".join(video_formats)
        )
        
        if file_path:
            self.file_input.setText(file_path)
            self.selected_video_file = file_path
   

    def get_available_video_inputs(self):
        # Use OpenCV to get available video inputs (webcams)
        available_inputs = []
        for i in range(5):  # Check up to 5 video inputs
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
            self.start_stop_button.style().polish(self.start_stop_button)
            self.stop_video_thread()
            print("Process Stopped")
            # Add logic to stop the process (replace print statement with your logic)
        else:
            # If the process is stopped, start it
            # Set the selected options when starting the process
            detection_model = self.detection_model_combobox.currentText()
            if detection_model=="Nano":
                self.selected_detection_model = "../yolov8n-seg.pt"
            elif detection_model=="Small":
                self.selected_detection_model = "../yolov8s-seg.pt"
            elif detection_model=="Medium":
                self.selected_detection_model = "../yolov8m-seg.pt"
            elif detection_model=="Large":
                self.selected_detection_model = "../yolov8l-seg.pt"
            elif detection_model=="Extra Large":
                self.selected_detection_model = "../yolov8x-seg.pt"
            else:
                self.selected_detection_model = "../yolov8s-seg.pt"
                
            self.selected_pixel_size = int(self.detection_video_res_combobox.currentText())
            self.selected_live_video_input = self.video_input_combobox.currentIndex()
            # self.selected_tracker = self.tracker_button_group.checkedButton().text()
            self.selected_confidence = self.confidence_slider.value() / 100.0
            self.selected_iou = self.iou_slider.value() / 100.0


            self.process_running = True
            self.start_stop_button.setProperty("stopped", "false")
            self.start_stop_button.setText("Stop")
            print("Process Started")
            self.video_thread = VideoProcessingThread(self)
            self.video_thread.finished.connect(self.on_video_processing_finished)
            self.video_thread.start()
            
            # Add logic to start the process (replace print statement with your logic)

        # Update style to apply changes
        self.start_stop_button.style().polish(self.start_stop_button)

    def stop_video_thread(self):
        if hasattr(self, 'video_thread') and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()



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

        # Convert the OpenCV frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the RGB frame to a QImage
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert the QImage to a QPixmap
        pixmap = QPixmap.fromImage(q_image)

        # Set the QPixmap as the label's pixmap
        self.detection_video_display.setPixmap(pixmap)


    # def press_hold_assistant(self):
    #     if self.press_hold_button.isChecked():
    #         print("Press and Hold Assistant button is pressed and held")
    #         # Add logic for press and hold assistant action (e.g., activate voice recognition)
    #     else:
    #         print("Press and Hold Assistant button is released")
    #         # Add logic for releasing press and hold assistant action (e.g., deactivate voice recognition)

    # def start_stop_assistant(self):
    #     if self.start_stop_assistant_button.isChecked():
    #         print("Start Assistant button pressed")
    #         # Add logic for starting the assistant (e.g., start processing voice commands)
    #     else:
    #         print("Stop Assistant button pressed")
    #         # Add logic for stopping the assistant (e.g., stop processing voice commands)



    # Inside Widget class
    def assistant_start_stop_process(self):
        if self.assistant_process_running: #self.worker_thread.isRunning():
            self.worker.is_running = False
            self.worker_thread.wait()
            self.assistant_process_running = False
            self.assistant_start_stop_button.setProperty("stopped", "true")
            self.assistant_start_stop_button.setText("Start")
            self.status_label.setText("Assistant Offline.")
            self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

        else:
            self.worker.is_running = True
            self.worker.show_date = True  # Reset to True when starting
            self.worker_thread.start()
            # If the process is stopped, start it
            self.assistant_process_running = True
            self.assistant_start_stop_button.setProperty("stopped", "false")
            self.assistant_start_stop_button.setText("Stop")
            self.status_label.setText("Assistant Online.")
            self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")

        # Update style to apply changes
        self.assistant_start_stop_button.style().polish(self.assistant_start_stop_button)


    # def stop_assistant_thread(self):
    #     if hasattr(self, 'assistant_thread') and self.assistant_thread.isRunning():
    #         self.assistant_thread.stop()
    #         self.assistant_thread.wait()

    # def on_assistant_finished(self):
    #     # This method is called when the video processing thread finishes
    #     print("Assistant Finished")
    #     self.assistant_thread.wait()  # Wait for the thread to finish before allowing it to be destroyed
    #     self.assistant_thread.deleteLater()  # Delete the thread

    def update_label(self, message, is_stop_message):
        current_text = self.text_browser.toPlainText()
        new_text = f"{current_text}\n{message}"
        self.text_browser.setPlainText(new_text)
        # Scroll to the bottom
        scrollbar = self.text_browser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        if message=="Exited":
            print("-------------stoped-------")
            self.worker_thread.wait()
            self.status_label.setText("Assistant Offline.")
            self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            self.assistant_process_running = False
            self.assistant_start_stop_button.setProperty("stopped", "true")
            self.assistant_start_stop_button.setText("Start")
        # if is_stop_message:
        #     self.status_label.setText("Assistant Online.")
        #     self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        # else:
        #     self.status_label.setText("Assistant Offline.")
        #     self.status_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

        self.assistant_start_stop_button.style().polish(self.assistant_start_stop_button)

