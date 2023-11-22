import sounddevice as sd
import numpy as np
import speech_recognition as sr
import mysql.connector
from datetime import datetime

def recognize_speech():
    recognizer = sr.Recognizer()
    audio = sd.rec(int(5 * 44100), samplerate=44100, channels=2, dtype=np.int16)
    sd.wait()
    audio = audio.tobytes()

    with sr.AudioData(audio, sample_rate=44100, sample_width=2) as source:
        print("Listening for your request...")
        try:
            request = recognizer.recognize_google(source)
            print("You said: " + request)
            return request
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand your request.")
            return None
        except sr.RequestError as e:
            print("Sorry, an error occurred: {0}".format(e))
            return None

def check_tracker_id(tracker_id):
    try:
        # Database Connection
        db = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="",
            database="yolo"
        )
        cursor = db.cursor()

        # Check if the tracker_id exists in the database
        cursor.execute("SELECT tracker_id FROM detections WHERE tracker_id = %s", (tracker_id,))
        result = cursor.fetchone()

        if result:
            print(f"Tracker ID {tracker_id} has been seen.")
        else:
            print(f"Tracker ID {tracker_id} has not been seen.")

        db.close()
    except mysql.connector.Error as err:
        print("Database error: {}".format(err))

if __name__ == "__main__":
    while True:
        request = recognize_speech()

        if request:
            if "seen tracker ID" in request:
                words = request.split()
                try:
                    tracker_id = int(words[-1])
                    check_tracker_id(tracker_id)
                except ValueError:
                    print("Invalid tracker ID format. Please provide a valid integer tracker ID.")
