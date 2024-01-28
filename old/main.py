from ultralytics import YOLO
import cv2
import numpy as np
import mysql.connector
import speech_recognition as sr
from gtts import gTTS
import subprocess
#from spatial_mapping import SpatialMapper



# Initialize YOLO model
model = YOLO("data/yolov8n.pt")

# Database connection setup
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="",
    database="yolo"
)
cursor = db.cursor()

def detect_and_track_objects(frame):
    # Detect objects in the frame
    detections = model(frame)

    # Placeholder for tracking logic
    # Update this part with your tracking algorithm
    tracked_objects = track_objects(detections)

    # Update database with the latest locations
    for obj in tracked_objects:
        # Example: obj might have attributes like obj.id, obj.x, obj.y
        cursor.execute("UPDATE object_table SET x=%s, y=%s WHERE id=%s", (obj.x, obj.y, obj.id))
    db.commit()



def spatial_analysis(objects):
    # Placeholder for spatial analysis logic
    # Use SpatialMapper to analyze the spatial relationship between objects
    # Update the database with relative positions
    for obj in objects:
        # Example: Update relative position in the database
        cursor.execute("UPDATE object_table SET relative_position=%s WHERE id=%s", (obj.relative_position, obj.id))
    db.commit()



def listen_for_commands():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        return command.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError:
        print("Error with the Google Speech Recognition service")
        return None
    
# def respond(text):
#     print("Showing and telling:", text)
#     tts = gTTS(text=text, lang='en')
#     tts.save("response.mp3")
#     subprocess.run(["afplay", "response.mp3"])  # Use afplay for audio playback on macOS

def respond_to_commands(command):
    if "where is" in command:
        # Extract object name from command
        object_name = command.split("where is")[-1].strip()
        
        # Query the database for the object's location
        cursor.execute("SELECT location FROM object_table WHERE name=%s", (object_name,))
        location = cursor.fetchone()

        if location:
            response = f"The {object_name} is at {location[0]}"
        else:
            response = f"I don't know where the {object_name} is."

        tts = gTTS(text=response, lang='en')
        tts.save("response.mp3")
        subprocess.run(["afplay", "response.mp3"])  # Change 'afplay' to suitable player for your OS

def main():
    cap = cv2.VideoCapture(0)  # or the path to your video file

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_and_track_objects(frame)
        # Perform spatial analysis here if necessary

        command = listen_for_commands()
        if command:
            respond_to_commands(command)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
