import speech_recognition as sr
from gtts import gTTS
import subprocess
import pyautogui
import webbrowser
import mysql.connector

def sep(mes="---"):
    print("==================================="+mes+"===================================")

class_names = (
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)

def listen_for_command():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for commands...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio)
        print("You said:", command)
        #respond(command)  # Call the function to display and tell what was recognized
        return command.lower()
    except sr.UnknownValueError:
        print("Could not understand audio. Please try again.")
        return None
    except sr.RequestError:
        print("Unable to access the Google Speech Recognition API.")
        return None

def respond(text):
    print("Assistant Said:", text)
    sep()
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    subprocess.run(["afplay", "response.mp3"])  # Use afplay for audio playback on macOS


# Helper function to respond with location information
def respond_location_results(results, object_type, object_identifier=None):
    if results:
        num_objects = len(results)
        object_descriptions = []

        for result in results:
            location = result[7]
            if location is not None:
                object_descriptions.append(f"{location} the {result[6]}")

        if num_objects == 1:
            if object_type=="tracker_id":
                respond(f"I have seen the tracker ID {object_identifier}. I can see that it's a {result[3]} and it's {object_descriptions[0]}.")
            elif object_identifier != None:
                respond(f"I have seen {object_type} {object_identifier} {object_descriptions[0]}.")
            else:
                respond(f"I have seen {object_type} {object_descriptions[0]}.")
        elif num_objects > 1:
            respond(f"I have seen {num_objects} {object_type}s. ")
            for i in range(num_objects):
                respond(f"One is {object_descriptions[i]}.")
        else:
            respond(f"I haven't seen any {object_type}.")
    else:
        respond(f"I haven't seen any {object_type}.")

def check_location(tracker_id=None, obj_class=None, type=None):
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="",
            database="assistant"
        )
        cursor = connection.cursor()
        if tracker_id is not None and obj_class is None and type == "id":
            cursor.execute("SELECT * FROM detections WHERE mobile_object_tracker_id = %s", (tracker_id,))
            results = cursor.fetchall()  # Fetch all rows
        elif tracker_id is None and obj_class is not None and type == "class":
            cursor.execute("""
                SELECT *
                FROM detections
                WHERE mobile_object_class_name = %s
                GROUP BY stationary_object_class_id
                ORDER BY stationary_object_class_id, MAX(timestamp) DESC
                """, (obj_class,))

            results = cursor.fetchall()  # Fetch all rows
        elif tracker_id is None and obj_class is None and type == "all":
            cursor.execute("SELECT * FROM detections")
            results = cursor.fetchall()  # Fetch all rows
        else:
            return None

        return results

    except mysql.connector.Error as error:
        print("Error:", error)
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def clear_database():
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="",
            database="assistant"
        )
        cursor = connection.cursor()

        cursor.execute("DELETE FROM detections")
        connection.commit()

        respond("Cleared all data from the database!")

    except mysql.connector.Error as error:
        print("Error:", error)
        respond("An error occurred while clearing the database.")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()



def is_number(x):
    if type(x) == str:
        x = x.replace(',', '')
    try:
        float(x)
    except:
        return False
    return True

def text2int(textnum, numwords={}):
    units = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,'for': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
        'seventeen': 17, 'eighteen': 18, 'nineteen': 19
    }
    tens = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
    scales = {'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000, 'trillion': 1000000000000}

    if not numwords:
        numwords.update(units)
        numwords.update(tens)
        numwords.update(scales)

    current = result = 0
    onnumber = False
    lastunit = False
    lastscale = False

    def is_numword(x):
        return x.replace('-', '').lower() in numwords

    def from_numword(x):
        return numwords[x.replace('-', '').lower()]

    for word in textnum.replace('-', ' ').split():
        if word in scales:
            lastscale = True
            if onnumber:
                current = max(1, current)
                result += current
            current = 0
        elif is_numword(word):
            onnumber = True
            increment = from_numword(word)
            current = current * 10 + increment
            lastunit = True
        elif word == 'and' and not lastscale:
            lastscale = True
        elif lastunit:
            onnumber = False
            lastunit = False

    if onnumber:
        current = max(1, current)
        result += current

    return result


def main():
    global tasks
    global listeningToTask

    while True:
        command = listen_for_command()

        triggerKeywords = ["assistant", "tracker", "seen", "id", "have you"]
        #print("Received command:", command)

        if command and any(keyword in command for keyword in triggerKeywords):
            if "show the database" in command:
                respond("Opening browser.")
                webbrowser.open("http://localhost/phpmyadmin/index.php?route=/sql&pos=0&db=assistant&table=detections")
            elif "tracker" in command and "id" in command:
                parts = command.split()
                index_tracker_id = parts.index("tracker") + 2  # Adjusted index to get the part after "tracker"

                if len(parts) > index_tracker_id:
                    raw_tracker_id = parts[index_tracker_id].lower()  # Convert to lowercase for case-insensitive matching
                    try:
                        tracker_id=int(raw_tracker_id)
                    except ValueError:
                        tracker_id = text2int(raw_tracker_id)

                    # Use the text2int function to convert words to numbers
                    # tracker_id = text2int(raw_tracker_id)

                    if tracker_id is not None:
                        # Use the provided query to get the object locations for the given tracker ID
                        results = check_location(tracker_id,None,"id")
                        respond_location_results(results, "tracker_id", raw_tracker_id)
                    else:
                        respond("I'm sorry, I couldn't convert the tracker ID to a number.")
                else:
                    respond("I'm not sure how to handle that command.")

            elif any(keyword in command for keyword in ["seen", "saw", "know", "where"]):
                for class_name in class_names:
                    if class_name.lower() in command.lower():
                        # Use the provided query to get the object locations for the given class name
                        results = check_location(None, class_name, "class")
                        respond_location_results(results, class_name)
                    # else:
                    #     respond("I'm sorry, I did not get the object name.")
                    
            elif "clear the database" in command:
                clear_database()
            elif "exit" in command:
                respond("Goodbye!")
                break
            else:
                respond("Sorry, I'm not sure how to handle that command.")

if __name__ == "__main__":
    respond("Assistant Online")
    main()
    
    # id_result = check_location(3,None,"id")
    # print(id_result)
    # print(check_location(None,"bottle","class"))
    # print(check_location(None,"book","class"))

    # print(check_location(None,None,"all"))

