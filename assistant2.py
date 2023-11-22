import numpy as np
import speech_recognition as sr
import mysql.connector
import pyaudio

# Set the correct input device name
input_device = 1  # Replace with your input device name

def list_input_devices():
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    print("List of available audio input devices:")
    for i in range(0, numdevices):
        if "Microphone" in p.get_device_info_by_host_api_device_index(0, i)['name']:
            print(f"Device {i}: {p.get_device_info_by_host_api_device_index(0, i)['name']}")

def recognize_speech():
    recognizer = sr.Recognizer()
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    input_device_index=input_device,
                    frames_per_buffer=1024)

    print("Listening for your request...")

    audio_frames = []
    try:
        while True:
            audio_chunk = stream.read(1024)
            audio_frames.append(audio_chunk)
    except KeyboardInterrupt:
        pass

    audio = b''.join(audio_frames)

    with sr.AudioData(audio, sample_rate=44100, sample_width=2) as source:
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
            host="YourHost",
            user="YourUser",
            password="YourPassword",
            database="YourDatabase"
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
    list_input_devices()
    
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
