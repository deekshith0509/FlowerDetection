import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import tensorflow as tf
import numpy as np

# Load the pre-trained flower recognition model
model = tf.keras.models.load_model('Flower_Recog_Model.keras')

# Define flower names corresponding to the model's output classes
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def classify_image(image_path):
    try:
        # Load and preprocess the input image
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)

        # Make predictions on the input image
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])

        # Determine the predicted flower class and confidence score
        predicted_class_index = np.argmax(result)
        predicted_flower = flower_names[predicted_class_index]
        confidence_score = np.max(result) * 100

        return predicted_flower, confidence_score
    except Exception as e:
        print(f"Error classifying image: {e}")
        return None, None

def classify_images_in_directory(directory_path):
    predictions = []
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        flower_class, confidence = classify_image(image_path)
        if flower_class is not None and confidence is not None:
            predictions.append((image_file, flower_class, confidence))
    
    return predictions

def select_directory():
    directory_path = filedialog.askdirectory()
    if directory_path:
        predictions = classify_images_in_directory(directory_path)
        if predictions:
            display_predictions(directory_path, predictions)
        else:
            messagebox.showinfo("Info", "No valid images found for classification.")

def display_predictions(directory_path, predictions):
    new_window = tk.Toplevel(root)
    new_window.title("Image Classifications")

    canvas = tk.Canvas(new_window)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(new_window, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Configure canvas to use scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create a frame inside the canvas for the images
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)

    # Display images and classifications in the frame
    for i, (image_file, flower_class, confidence) in enumerate(predictions):
        img = Image.open(os.path.join(directory_path, image_file))
        img = img.resize((200, 200))  # Resize for display
        photo = ImageTk.PhotoImage(img)

        label_text = f"{image_file} - {flower_class} (Confidence: {confidence:.2f}%)"
        label = tk.Label(frame, text=label_text)
        label.grid(row=i, column=0, padx=10, pady=10, sticky=tk.W)

        img_label = tk.Label(frame, image=photo)
        img_label.image = photo  # Keep a reference to avoid garbage collection
        img_label.grid(row=i, column=1, padx=10, pady=10)

    # Update the canvas scroll region
    frame.update_idletasks()  # Ensure all widgets are updated and visible
    canvas.configure(scrollregion=canvas.bbox(tk.ALL))

# Create main application window
root = tk.Tk()
root.title("Image Classifier")

# Create a button to select directory
select_button = tk.Button(root, text="Select Directory", command=select_directory)
select_button.pack(pady=20)

# Run the main event loop
root.mainloop()
