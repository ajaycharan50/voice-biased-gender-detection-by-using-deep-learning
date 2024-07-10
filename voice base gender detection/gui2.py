import tkinter as tk
from tkinter import filedialog, messagebox
import os
from audio_processing import record_to_file, extract_feature, load_model
from PIL import Image, ImageDraw, ImageFont
import playsound
import threading

# Create a Tkinter root window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Display a messagebox to choose between selecting a file or speaking
response = messagebox.askquestion("File Selection", "Do you want to select a WAV file?")

if response == "yes":
    # Open a file dialog to select the WAV file
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    # Check if a file was selected
    if file_path:
        print("Selected file:", file_path)
    else:
        print("No file selected.")
        exit()

    if not os.path.isfile(file_path):
        print("Invalid file:", file_path)
        print("Please talk")
        file = "test.wav"
        record_to_file(file)
    else:
        file = file_path

else:
    print("Please talk")
    file = "test.wav"
    record_to_file(file)

# Confirm with the user to play the audio file
confirmation = messagebox.askquestion("Play Audio", "Do you want to listen?")

def play_audio():
    # Play the selected audio file
    playsound.playsound(file)

if confirmation == "yes":
    # Show a messagebox while playing the audio file
    messagebox.showinfo("Listening", "You are listening to the selected audio file and the gender detection analysis is running in the background press ok.")

    # Start a new thread to play the audio file
    audio_thread = threading.Thread(target=play_audio)
    audio_thread.start()            

# Load the trained weights
model = load_model(r"C:\Users\Dell\Desktop\voice base gender detection\results\model.h5")  # Assuming this loads your model, replace with your actual loading logic

# Rest of your code
# extract features and reshape it
features = extract_feature(file, mel=True).reshape(1, -1)

# predict the gender!
male_prob = model.predict(features)[0][0]
female_prob = 1 - male_prob
gender = "male" if male_prob > female_prob else "female"

# Display the result image
if gender == "male":
    image_path = r"C:\Users\Dell\Desktop\voice base gender detection\male.png"  # Path to the male image file
else:
    image_path = r"C:\Users\Dell\Desktop\voice base gender detection\female.png"  # Path to the female image file

result_image = Image.open(image_path)

# Draw the text on the image
draw = ImageDraw.Draw(result_image)
title_text = "Voice-based Gender Detection Mini Project By Me"
gender_text = f"Gender: {gender}\nProbabilities: Male: {male_prob*100:.2f}% Female: {female_prob*100:.2f}%"
font = ImageFont.truetype("arial.ttf", size=20)

title_text_bbox = draw.textbbox((0, 0), title_text, font=font)
gender_text_bbox = draw.textbbox((0, 0), gender_text, font=font)

title_text_position = (10, 10)
gender_text_position = (10, title_text_bbox[3] + 10)

draw.rectangle(title_text_bbox, fill='white')
draw.rectangle(gender_text_bbox, fill='white')

draw.text(title_text_position, title_text, fill='black', font=font)
draw.text(gender_text_position, gender_text, fill='black', font=font)

# Show the result image
result_image.show()

# Optionally, you can save the result image
result_image.save("result_image.png")
