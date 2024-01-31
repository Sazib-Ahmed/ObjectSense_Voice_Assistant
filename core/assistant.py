import speech_recognition as sr
from gtts import gTTS
import subprocess
import pyautogui
import webbrowser
import mysql.connector
from datetime import datetime

timestamp_format = "%I:%M:%S %p"
def ass_message(worker):
    worker.text_signal.emit(">>>>>>>>>>>> Hi, I am assistan.t <<<<<<<<<<<", True) 


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

def listen_for_command(assistant_worker_thread, timeout=5):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        try:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source)
            print("Listening for commands...")
            timestamp = datetime.now().strftime(timestamp_format)
            message = f"{timestamp}: Listening for commands..."
            assistant_worker_thread.text_signal.emit(message, False)

            audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)

            command = recognizer.recognize_google(audio)
            print("You said:", command)
            mes = "You: " + command
            timestamp = datetime.now().strftime(timestamp_format)
            assistant_worker_thread.text_signal.emit(f"\n-------------------\n{timestamp}\n{mes}\n-------------------", False)

            return command.lower()

        except sr.UnknownValueError:
            print("Could not understand audio. Please try again.")
            return None

        except sr.RequestError:
            print("Unable to access the Google Speech Recognition API.")
            return None

        except sr.WaitTimeoutError:
            print("Listening timeout. No command detected.")
            return None

def respond(text,assistant_worker_thread):
    print("Assistant Said:", text)
    mes = "Assistant: "+text
    sep()
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    timestamp = datetime.now().strftime(timestamp_format)
    assistant_worker_thread.text_signal.emit(f"\n-------------------\n{timestamp}\n{mes}\n-------------------", False)

    subprocess.run(["afplay", "response.mp3"])  # Use afplay for audio playback on macOS
    


# Helper function to respond with location information
def respond_location_results(results, object_type, assistant_worker_thread, object_identifier=None):
    response_list = ""

    if results:
        num_objects = len(results)
        object_descriptions = []

        for result in results:
            location = result[7]
            if location is not None:
                object_descriptions.append(f"{location} the {result[6]}")

        if num_objects == 1:
            if object_type == "tracker_id":
                response_list += f"I have seen the tracker ID {object_identifier}. I can see that it's a {result[3]} and it's {object_descriptions[0]}."
            elif object_identifier is not None:
                response_list += f"I have seen {object_type} {object_identifier}. It is {object_descriptions[0]}."
            else:
                response_list += f"I have seen {object_type}. It is {object_descriptions[0]}."
        elif num_objects > 1:
            response_list += f"I have seen {num_objects} {object_type}s. "
            for i in range(num_objects):
                response_list += f"One is {object_descriptions[i]}."
        else:
            response_list += f"I haven't seen any {object_type}."

    else:
        response_list += f"I haven't seen any {object_type}."

    # Send the response to the respond method
    respond(response_list, assistant_worker_thread)


def respond(message, assistant_worker_thread):
    timestamp = datetime.now().strftime(timestamp_format)
    formatted_message = f"\n-------------------\n{timestamp}\n{message}\n-------------------"
    assistant_worker_thread.text_signal.emit(formatted_message, False)


def check_location(assistant_worker_thread,tracker_id=None, obj_class=None, type=None):
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
        respond("Unable to connect to the database.",assistant_worker_thread)
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def clear_database(assistant_worker_thread):
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

        respond("Cleared all data from the database!",assistant_worker_thread)

    except mysql.connector.Error as error:
        print("Error:", error)
        respond("An error occurred while clearing the database.",assistant_worker_thread)

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


def start_assistant(assistant_worker_thread,is_running):
    timestamp_format1 = "%B %d, %Y  \n  Time: %I:%M:%S %p          "
    assistant_worker_thread.text_signal.emit("=====================\n||       New Chat Started       ||\n=====================",True) 
    timestamp = datetime.now().strftime(timestamp_format1)
    assistant_worker_thread.text_signal.emit(f"  Date: {timestamp}\n=====================", False)


    respond("Assistant Online",assistant_worker_thread)
    
    while assistant_worker_thread.is_running:
        command = listen_for_command(assistant_worker_thread)

        triggerKeywords = ["assistant", "tracker", "seen", "id", "have you"]
        #print("Received command:", command)

        if command and any(keyword in command for keyword in triggerKeywords):
            if "show the database" in command:
                respond("Opening browser.",assistant_worker_thread)
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
                        results = check_location(assistant_worker_thread,tracker_id,None,"id")
                        respond_location_results(results, "tracker_id",assistant_worker_thread, raw_tracker_id)
                    else:
                        respond("I'm sorry, I couldn't convert the tracker ID to a number.",assistant_worker_thread)
                else:
                    respond("I'm not sure how to handle that command.",assistant_worker_thread)

            elif any(keyword in command for keyword in ["seen", "saw", "know", "where"]):
                for class_name in class_names:
                    if class_name.lower() in command.lower():
                        # Use the provided query to get the object locations for the given class name
                        results = check_location(assistant_worker_thread,None, class_name, "class")
                        if results:
                            respond_location_results(results, class_name,assistant_worker_thread)
                        else:
                            respond(f"No data found in the database about {class_name}",assistant_worker_thread)
                    # else:
                    #     respond("I'm sorry, I did not get the object name.")
                    
            elif "clear the database" in command:
                clear_database(assistant_worker_thread)
            elif "exit" in command:
                respond("Goodbye!",assistant_worker_thread)
                assistant_worker_thread.text_signal.emit("Exited",True) 
                break
            else:
                respond("Sorry, I'm not sure how to handle that command.",assistant_worker_thread)
            
    else:
        assistant_worker_thread.text_signal.emit("\n\n=====================\n||      Assistant Stopped.      ||\n=====================\n\n\n",True) 

# if __name__ == "__main__":
#     respond("Assistant Online")
#     main()
    
#     # id_result = check_location(3,None,"id")
#     # print(id_result)
#     # print(check_location(None,"bottle","class"))
#     # print(check_location(None,"book","class"))

#     # print(check_location(None,None,"all"))

