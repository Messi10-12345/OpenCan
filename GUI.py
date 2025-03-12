import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from pt2 import AIFOROUTE

model = tf.keras.models.load_model('model.keras')
IMG_SIZE = (224, 224)

def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE, color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    print("Raw prediction value:", prediction)  
    return prediction > 0.5

def open_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img = ImageTk.PhotoImage(img)
            panel.config(image=img)
            panel.image = img

            has_tumor = predict_image(file_path)
            if has_tumor:
                result_label.config(text="MENINGIOMA DETECTED", fg="red")
                show_form()
            else:
                result_label.config(text="NO TUMOR", fg="green")
                hide_form()
        except Exception as e:
            result_label.config(text=f"Error: {e}", fg="red")

def show_form():
    form_frame.pack(pady=10)

def hide_form():
    form_frame.pack_forget()

def submit_form():
    global gender, age, allergies, area  
    gender = gender_entry.get()
    age = age_entry.get()
    allergies = allergy_entry.get()
    area = location_entry.get()

    print(f"Stored Data - Gender: {gender}, Age: {age}, Allergies: {allergies}, Tumor Location: {area}")

    plan = AIFOROUTE(gender, age, allergies, area)

    text_widget.delete("1.0", tk.END)
    text_widget.insert("1.0", plan)

    result_label.config(text="Form Submitted!", fg="blue")
    hide_form()

root = tk.Tk()
root.title("Cancer Detection AI")
root.geometry("600x600")
root.config(bg="#f5f5f5")

title_label = tk.Label(root, text="OpenCancer", font=("Helvetica", 24, "bold"), bg="#f5f5f5", fg="#333333")
title_label.pack(pady=20)

frame = tk.Frame(root, bg="#f5f5f5")
frame.pack(pady=10)

btn_open = tk.Button(frame, text="Upload CT-Scan", command=open_image, font=("Helvetica", 16), bg="#4CAF50", fg="white", width=20, height=2, bd=0, relief="flat")
btn_open.pack(pady=10)

panel = tk.Label(frame, bd=2, relief="solid", bg="light gray", width=320, height=250)
panel.pack(padx=10, pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 18, "bold"), bg="#f5f5f5")
result_label.pack(pady=10)

form_frame = tk.Frame(root, bg="#f5f5f5")
form_frame.pack_forget()

tk.Label(form_frame, text="Age:", bg="#f5f5f5").grid(row=0, column=0, sticky="w")
age_entry = tk.Entry(form_frame)
age_entry.grid(row=0, column=1)

tk.Label(form_frame, text="Gender:", bg="#f5f5f5").grid(row=1, column=0, sticky="w")
gender_entry = tk.Entry(form_frame)
gender_entry.grid(row=1, column=1)

tk.Label(form_frame, text="Allergies:", bg="#f5f5f5").grid(row=2, column=0, sticky="w")
allergy_entry = tk.Entry(form_frame)
allergy_entry.grid(row=2, column=1)

tk.Label(form_frame, text="Tumor Location:", bg="#f5f5f5").grid(row=3, column=0, sticky="w")
location_entry = tk.Entry(form_frame)
location_entry.grid(row=3, column=1)

submit_btn = tk.Button(form_frame, text="Submit", command=submit_form, font=("Helvetica", 12), bg="#2196F3", fg="white", width=10)
submit_btn.grid(row=4, columnspan=2, pady=10)

text_widget = tk.Text(root, height=5, width=40)
text_widget.pack(pady=10)

root.mainloop()
