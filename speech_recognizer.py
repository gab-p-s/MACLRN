import speech_recognition as sr

recognizer = sr.Recognizer()
filename = input("Input filename: ")

with sr.AudioFile(filename) as source:
    print("Reading audio...")
    audio_data = recognizer.record(source)

try:
    print("\nRecognized Text:")
    text = recognizer.recognize_google(audio_data)
    print(text)

except sr.UnknownValueError:
    print("Sorry, could not understand the audio.")
except sr.RequestError:
    print("Could not connect to Google API.")