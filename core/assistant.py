import speech_recognition as sr
from gtts import gTTS
import subprocess
# import pyautogui
import webbrowser
import mysql.connector
from datetime import datetime
# import pyttsx3

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
    # Initialize a recognizer object
    recognizer = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        try:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            # Indicate that the system is listening for commands
            print("Listening for commands...")
            # Get the current timestamp
            timestamp = datetime.now().strftime(timestamp_format)
            # Emit a signal to update the user interface with the listening status
            message = f"{timestamp}: Listening for commands for 3 seconds..."
            assistant_worker_thread.text_signal.emit(message, False)
            
            # Listen for audio input with a timeout
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
            # Attempt to recognize speech using Google's speech recognition service
            command = recognizer.recognize_google(audio)

            # Log the recognized command
            print("You said:", command)
            # Prepare the command for display in the user interface
            mes = "You: " + command
            # Get the current timestamp
            timestamp = datetime.now().strftime(timestamp_format)
            # Emit a signal to update the user interface with the recognized command
            assistant_worker_thread.text_signal.emit(f"\n-------------------\n{timestamp}\n{mes}\n-------------------", False)

            # Return the recognized command in lowercase
            return command.lower()

        except sr.UnknownValueError:
            # Handle cases where speech cannot be understood
            print("Could not understand audio. Please try again.")
            return None

        except sr.RequestError:
            # Handle cases where access to the Google Speech Recognition API fails
            print("Unable to access the Google Speech Recognition API.")
            return None

        except sr.WaitTimeoutError:
            # Handle cases where no command is detected within the specified timeout duration
            print("Listening timeout. No command detected.")
            return None


def respond(text, assistant_worker_thread):
    # Log the assistant's response
    print("Assistant Said:", text)
    # Prepare the assistant's response for display in the user interface
    mes = "Assistant: " + text
    # Call a function to print separator lines for better readability
    sep()  
    # Convert the text response into speech using Google's Text-to-Speech service
    tts = gTTS(text=text, lang='en')
    # Save the speech as an audio file
    tts.save("response.mp3")
    # Get the current timestamp
    timestamp = datetime.now().strftime(timestamp_format)
    # Emit a signal to update the user interface with the assistant's response
    assistant_worker_thread.text_signal.emit(f"\n-------------------\n{timestamp}\n{mes}\n-------------------", False)
    # Play the generated audio response using the system's audio player (afplay for macOS)
    subprocess.run(["afplay", "response.mp3"])

    
# def respond(text, assistant_worker_thread):
#     try:
#         print("Assistant Said:", text)
#         mes = "Assistant: "+text
#         sep()
#         timestamp = datetime.now().strftime(timestamp_format)
#         assistant_worker_thread.text_signal.emit(f"\n-------------------\n{timestamp}\n{mes}\n-------------------", False)

#         # Initialize the text-to-speech engine
#         engine = pyttsx3.init()

#         # Adjust the rate and volume if needed

#         engine.say(text)
#         engine.runAndWait()

#     except Exception as e:
#         print("Error during text-to-speech:", str(e))
#         # Handle the error as needed



def respond_location_results(results, object_type, assistant_worker_thread, object_identifier=None):
    # Initialize an empty string to store the response
    response_list = ""

    # Check if there are any results from the database query
    if results:
        # Get the number of objects detected
        num_objects = len(results)
        # Initialize an empty list to store object descriptions
        object_descriptions = []

        # Iterate through each result to extract location information
        for result in results:
            location = result[7]  # Assuming location information is stored at index 7
            # Check if location information is available
            if location is not None:
                # Construct a descriptive string for each object
                object_descriptions.append(f"{location} the {result[6]}")  # Assuming object class is at index 6

        # Generate the response based on the number of objects detected
        if num_objects == 1:
            # Handle singular case
            if object_type == "tracker_id":
                response_list += f"I have seen the tracker ID {object_identifier}. I can see that it's a {result[3]} and it's {object_descriptions[0]}. "
            elif object_identifier is not None:
                response_list += f"I have seen {object_type} {object_identifier}. It is {object_descriptions[0]}. "
            else:
                response_list += f"I have seen {object_type}. It is {object_descriptions[0]}. "
        elif num_objects > 1:
            # Handle plural case
            response_list += f"I have seen {num_objects} {object_type}s. "
            for i in range(num_objects):
                response_list += f"One is {object_descriptions[i]}. "
        else:
            # Handle case when no objects are detected
            response_list += f"I haven't seen any {object_type}. "
    else:
        # Handle case when no results are returned from the database query
        response_list += f"I haven't seen any {object_type}. "

    # Send the generated response to the respond method for further processing
    respond(response_list, assistant_worker_thread)


def check_location(assistant_worker_thread, tracker_id=None, obj_class=None, type=None):
    try:
        # Establish a connection to the MySQL database
        connection = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="",
            database="assistant"
        )
        cursor = connection.cursor()
        # Check if the function is called to retrieve data by tracker ID
        if tracker_id is not None and obj_class is None and type == "id":
            # Execute SQL query to fetch all detections with the specified tracker ID
            cursor.execute("SELECT * FROM detections WHERE mobile_object_tracker_id = %s", (tracker_id,))
            results = cursor.fetchall()  # Fetch all rows
        # Check if the function is called to retrieve data by object class
        elif tracker_id is None and obj_class is not None and type == "class":
            # Execute SQL query to fetch all detections with the specified object class, grouping by stationary object class ID
            cursor.execute("""
                SELECT *
                FROM detections
                WHERE mobile_object_class_name = %s
                GROUP BY stationary_object_class_id
                ORDER BY stationary_object_class_id, MAX(timestamp) DESC
                """, (obj_class,))
            results = cursor.fetchall()  # Fetch all rows
        # Check if the function is called to retrieve all detection records
        elif tracker_id is None and obj_class is None and type == "all":
            # Execute SQL query to fetch all detections
            cursor.execute("SELECT * FROM detections")
            results = cursor.fetchall()  # Fetch all rows
        else:
            return None

        return results

    except mysql.connector.Error as error:
        print("Error:", error)
        # If there's an error connecting to the database, notify the user via the assistant worker thread
        respond("Unable to connect to the database.", assistant_worker_thread)
        return None
    finally:
        # Close the cursor and database connection when finished
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

def convert_text_to_integer(textnum, word_to_num={}):
    # Define dictionaries to map words to their corresponding numerical values
    units = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'for': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
        'seventeen': 17, 'eighteen': 18, 'nineteen': 19
    }
    tens = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
    scales = {'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000, 'trillion': 1000000000000}

    # Update the default word_to_num dictionary with the defined units, tens, and scales dictionaries
    if not word_to_num:
        word_to_num.update(units)
        word_to_num.update(tens)
        word_to_num.update(scales)

    # Initialize variables for tracking the current and total numerical values
    current = result = 0
    onnumber = False
    lastunit = False
    lastscale = False

    # Define helper functions to check if a word is a number and retrieve its numerical value
    def is_numword(x):
        return x.replace('-', '').lower() in word_to_num

    def from_numword(x):
        return word_to_num[x.replace('-', '').lower()]

    # Iterate through each word in the input text
    for word in textnum.replace('-', ' ').split():
        # Check if the word represents a scale (e.g., hundred, thousand, million)
        if word in scales:
            lastscale = True
            # If a number was encountered before the scale, update the result with the current value
            if onnumber:
                current = max(1, current)
                result += current
            # Reset the current value for the next number
            current = 0
        # Check if the word represents a numerical value
        elif is_numword(word):
            onnumber = True
            # Calculate the increment based on the numerical value of the word
            increment = from_numword(word)
            # Update the current value by multiplying it by 10 and adding the increment
            current = current * 10 + increment
            lastunit = True
        # Handle the case when 'and' is used in a number (e.g., one hundred and ten)
        elif word == 'and' and not lastscale:
            lastscale = True
        # Reset the onnumber flag when a unit (e.g., hundred, thousand) is encountered
        elif lastunit:
            onnumber = False
            lastunit = False

    # Add the current value to the result if a number was the last word in the text
    if onnumber:
        current = max(1, current)
        result += current

    # Return the total numerical value derived from the input text
    return result






def start_assistant(assistant_worker_thread,is_running):
    # Define the timestamp format for logging
    timestamp_format1 = "%B %d, %Y  \n  Time: %I:%M:%S %p          "
    
    # Emit a signal to indicate the start of a new chat session
    assistant_worker_thread.text_signal.emit("=====================\n||       New Chat Started       ||\n=====================",True) 
    
    # Log the current date and time
    timestamp = datetime.now().strftime(timestamp_format1)
    assistant_worker_thread.text_signal.emit(f"  Date: {timestamp}\n=====================", False)

    # Notify that the assistant is online
    respond("Assistant Online",assistant_worker_thread)
    
    # Start a loop to continuously listen for commands while the assistant is running
    while assistant_worker_thread.is_running:
        # Listen for a command from the user
        command = listen_for_command(assistant_worker_thread)

        # Define trigger keywords to identify the user's intent
        triggerKeywords = ["assistant", "tracker", "seen", "id", "have you"]

        # Process the command if it exists and contains relevant trigger keywords
        if command and any(keyword in command for keyword in triggerKeywords):
            if "show the database" in command:
                # Open the database in a web browser
                respond("Opening browser.",assistant_worker_thread)
                webbrowser.open("http://localhost/phpmyadmin/index.php?route=/sql&pos=0&db=assistant&table=detections")
            elif "tracker" in command and "id" in command:
                # Process commands related to tracking IDs
                parts = command.split()
                index_tracker_id = parts.index("tracker") + 2  # Adjusted index to get the part after "tracker"

                if len(parts) > index_tracker_id:
                    # Extract and convert the tracker ID from the command
                    raw_tracker_id = parts[index_tracker_id].lower()  # Convert to lowercase for case-insensitive matching
                    try:
                        tracker_id=int(raw_tracker_id)
                    except ValueError:
                        tracker_id = text2int(raw_tracker_id)

                    # Fetch and respond with object locations based on the tracker ID
                    if tracker_id is not None:
                        results = check_location(assistant_worker_thread,tracker_id,None,"id")
                        respond_location_results(results, "tracker_id",assistant_worker_thread, raw_tracker_id)
                    else:
                        respond("I'm sorry, I couldn't convert the tracker ID to a number.",assistant_worker_thread)
                else:
                    respond("I'm not sure how to handle that command.",assistant_worker_thread)

            elif any(keyword in command for keyword in ["seen", "saw", "know", "where"]):
                # Process commands related to object detection
                for class_name in class_names:
                    if class_name.lower() in command.lower():
                        # Fetch and respond with object locations based on the object class
                        results = check_location(assistant_worker_thread,None, class_name, "class")
                        if results:
                            respond_location_results(results, class_name,assistant_worker_thread)
                        else:
                            respond(f"No data found in the database about {class_name}",assistant_worker_thread)
                            
            elif "clear the database" in command:
                # Clear the database upon request
                clear_database(assistant_worker_thread)
            elif "exit" in command:
                # Exit the assistant loop and end the chat session
                respond("Goodbye!",assistant_worker_thread)
                assistant_worker_thread.text_signal.emit("Exited",True) 
                break
            else:
                # Respond to unrecognized commands
                respond("Sorry, I'm not sure how to handle that command.",assistant_worker_thread)
    else:
        # Signal the end of the chat session when the assistant stops
        assistant_worker_thread.text_signal.emit("\n\n=====================\n||      Assistant Stopped.      ||\n=====================\n\n\n",True) 


