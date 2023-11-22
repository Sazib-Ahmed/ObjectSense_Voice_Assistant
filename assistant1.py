import speech_recognition as sr

def list_input_devices():
    # Initialize the recognizer
    r = sr.Recognizer()

    # List available input devices
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"Device {index}: {name}")

if __name__ == "__main__":
    print("List of available audio input devices:")
    list_input_devices()
