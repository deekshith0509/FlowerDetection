import os
import streamlit as st
import tensorflow as tf
import numpy as np

# Load the pre-trained flower recognition model
model = tf.keras.models.load_model('Flower_Recog_Model.keras')

# Define flower names corresponding to the model's output classes
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Function to classify images using the loaded model
def classify_image(image_path):
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

    # Generate the outcome message
    outcome = f"The image belongs to {predicted_flower} with a confidence score of {confidence_score:.2f}%"
    return outcome

# Streamlit app layout and functionality
st.header('Flower Classification CNN Model')

# File uploader for image upload
uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Create 'uploads' directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Save the uploaded file to the 'uploads' directory
    image_path = os.path.join('uploads', uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Classify the uploaded image using the model
    outcome = classify_image(image_path)
    st.markdown(outcome)
