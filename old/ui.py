import tkinter as tk
from tkinter import ttk
import threading
import cv2
from PIL import Image, ImageTk
from your_yolo_script import YOLO_Processing  # Assume this is your YOLO processing class

class YOLO_UI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("YOLO Model Output Display")
        self.video_frame = ttk.LabelFrame(self.window, text="Video Output")
        self.video_frame.grid(row=0, column=0, padx=10, pady=10)
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(padx=10, pady=10)
        self.button_frame = ttk.Frame(self.window)
        self.button_frame.grid(row=0, column=1, padx=10, pady=10)
        self.on_button = ttk.Button(self.button_frame, text="On", command=self.on_button_click)
        self.on_button.pack(fill='x', padx=5, pady=5)
        self.off_button = ttk.Button(self.button_frame, text="Off", command=self.off_button_click)
        self.off_button.pack(fill='x', padx=5, pady=5)
        self.yolo_processing = YOLO_Processing()
        self.update_video()

    def on_button_click(self):
        threading.Thread(target=self.yolo_processing.start_processing, args=(self.update_frame,)).start()

    def off_button_click(self):
        self.yolo_processing.stop_processing()

    def update_frame(self, frame):
        self.current_frame = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video_label.config(image=self.current_frame)

    def update_video(self):
        if self.yolo_processing.is_running():
            self.video_label.after(10, self.update_video)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = YOLO_UI()
    app.run()
