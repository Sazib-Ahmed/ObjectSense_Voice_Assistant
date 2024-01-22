import speech_recognition as sr
from gtts import gTTS
import subprocess
import pyautogui
import webbrowser
import mysql.connector

def sep(mes="---"):
    print("==================================="+mes+"===================================")

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
    print("Showing and telling:", text)
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    subprocess.run(["afplay", "response.mp3"])  # Use afplay for audio playback on macOS

def check_tracker_id(tracker_id=None,obj_class=None,type=None):
    try:
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="",
            database="assistant"
        )
        cursor = connection.cursor()
        sep()
        if tracker_id is not None and obj_class is None and type == "id":
            cursor.execute("SELECT * FROM detections WHERE mobile_object_tracker_id = %s", (tracker_id,))
            results = cursor.fetchone()  # Fetch one rows
        elif tracker_id is None and obj_class is not None and type == "class":
            #cursor.execute("SELECT * FROM detections WHERE mobile_object_class_name = %s", (obj_class,))
            cursor.execute("""
                SELECT *
                FROM detections
                WHERE mobile_object_class_name = %s
                GROUP BY stationary_object_class_id
                ORDER BY stationary_object_class_id, MAX(timestamp) DESC
                """, (obj_class,))


            results = cursor.fetchall()  # Fetch one rows
        elif tracker_id is None and obj_class is None and type == "all":
            cursor.execute("SELECT * FROM detections")
            results = cursor.fetchall()  # Fetch all rows
        else:
            return False

        if results:
            # for result in results:
            #     print(result)
            # sep()
            return results  # Return the fetched data as a list of tuples
        else:
            return None  # Return None if no data is found

    except mysql.connector.Error as error:
        print("Error:", error)
        return None
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

# Rest of your code...

tasks = []
listeningToTask = False

def main():
    global tasks
    global listeningToTask

    while True:
        command = listen_for_command()

        triggerKeyword = "assistant"
        print("Received command:", command)

        if command and triggerKeyword in command:
            print("in triggerKeyword")
            if listeningToTask:
                tasks.append(command)
                listeningToTask = False
                respond("Added")
            elif "add a task" in command:
                listeningToTask = True
                respond("Sure, what is the task?")
            elif "list tasks" in command:
                respond("Sure. Your tasks are:")
                for task in tasks:
                    respond(task)
            elif "take a screenshot" in command:
                screenshot = pyautogui.screenshot()
                screenshot.save("screenshot.png")
                respond("I took a screenshot for you.")
            elif "open chrome" in command:
                respond("Opening Chrome.")
                webbrowser.open("http://www.youtube.com/@JakeEh")
            elif "tracker" in command and "id" in command:
                print("Checking tracker ID command")
                parts = command.split()
                index_tracker_id = parts.index("tracker") + 2  # Adjusted index to get the part after "tracker"
                print("Tracker ID index:", index_tracker_id)

                if len(parts) > index_tracker_id:
                    raw_tracker_id = parts[index_tracker_id].lower()  # Convert to lowercase for case-insensitive matching
                    print("Raw tracker ID:", raw_tracker_id)
                    print(type(raw_tracker_id))
                    try:
                        tracker_id=int(raw_tracker_id)
                    except ValueError:
                        tracker_id = text2int(raw_tracker_id)

                    # Use the text2int function to convert words to numbers
                    # tracker_id = text2int(raw_tracker_id)
                    print(tracker_id)

                    if tracker_id is not None:
                        if check_tracker_id(tracker_id):
                            respond("Yes, I have seen tracker ID " + str(raw_tracker_id))
                        else:
                            respond("No, I have not seen tracker ID " + str(raw_tracker_id))
                    else:
                        respond("I'm sorry, I couldn't convert the tracker ID to a number.")
                else:
                    respond("I'm not sure how to handle that command.")

            elif "exit" in command:
                respond("Goodbye!")
                break
            else:
                respond("Sorry, I'm not sure how to handle that command.")

if __name__ == "__main__":
    # respond("Online")
    #main()
    
    id_result = check_tracker_id(3,None,"id")
    print(id_result)
    print(check_tracker_id(None,"bottle","class"))
    print(check_tracker_id(None,"book","class"))

    print(check_tracker_id(None,None,"all"))

