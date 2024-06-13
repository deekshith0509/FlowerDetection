import os
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import random
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# Initialize nltk
nltk.download('punkt')
nltk.download('wordnet')

# Load flower descriptions from a JSON file
flower_descriptions = {}

try:
    with open('flowers.txt', 'r') as file:
        data = file.read()
        if data:
            flower_descriptions = json.loads(data)
        else:
            st.error("The 'flowers.txt' file is empty.")
except FileNotFoundError:
    st.error("The 'flowers.txt' file was not found.")
except json.decoder.JSONDecodeError as err:
    st.error(f"Error decoding JSON data: {err}")

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
    return predicted_flower, confidence_score, outcome

# Preprocessing for chatbot
lemmer = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching for chatbot
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "hello", "howdy", "hi there", "hey there", "greetings", "good day", "good morning", "good afternoon", "good evening", "salutations", "yo", "what's up", "how's it going", "nice to see you", "pleased to meet you", "hiya", "hey friend", "hello there", "hi friend", "hey buddy", "good to see you", "how's everything", "hey you", "hi stranger", "well met", "hello friend", "how are you doing", "what's happening", "how's life", "how are things", "long time no see", "how have you been", "it's great to see you", "what's new", "what's happening", "how's your day", "how's your day going", "how's your day been", "how's your day been going"]


def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Function to get flower attribute response
def get_flower_attribute_response(flower_name, attribute):
    if flower_name in flower_descriptions and attribute in flower_descriptions[flower_name]:
        response = f"**{flower_name.capitalize()}: {attribute.capitalize()}**\n\n"
        response += f"{flower_descriptions[flower_name][attribute]}"
        return response
    else:
        return f"Sorry, I don't have information on {attribute} for {flower_name}."

# Streamlit app layout and functionality
st.title('Flower Chatbot & Recognition')

# Functionality for Image Recognition
st.header('Image Recognition')
uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    image_path = os.path.join('uploads', uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    flower_class, confidence, outcome = classify_image(image_path)
    st.session_state['flower_class'] = flower_class
    st.session_state['confidence'] = confidence

    st.markdown(outcome)

    if flower_class.lower() in flower_descriptions:
        st.session_state['description'] = flower_descriptions[flower_class.lower()]["description"]
        st.subheader('Description')
        st.write(st.session_state['description'])

# Functionality for Chatbot
st.header('FlowerBot: Chat with Me!')
user_input = st.text_input("You:", key='user_input')

if 'flower_name' not in st.session_state:
    st.session_state['flower_name'] = None
if 'attribute_input' not in st.session_state:
    st.session_state['attribute_input'] = ''



attribute_keywords = {
    "color": ["color", "colour", "hue", "tone", "shade", "pigment", "tint", "chroma", "saturation", "undertone", "palette", "spectrum", "rainbow", "vibrant", "vivid", "muted", "pastel", "primary", "secondary", "tertiary", "complementary", "monochromatic", "multicolored", "variegated"],
    "evolution": ["evolution", "evolutionary", "development", "history", "adaptation", "genetic variation", "natural selection", "speciation", "divergence", "convergence", "phylogeny", "phylogenetics", "evolutionary biology", "process", "journey"],
    "medicinal": ["medicinal", "medical", "therapeutic", "healing", "curative", "pharmaceutical", "pharmacological", "herbal", "traditional medicine", "herbal remedy", "alternative medicine", "health benefits", "wellness", "herbalism", "phytotherapy", "ethnobotany", "folk medicine", "natural medicine", "elixir", "panacea"],
    "characteristic": ["characteristic", "traits", "feature", "attribute", "quality", "aspect", "property", "marker", "identifier", "trait", "appearance", "morphology", "behavior", "habit", "habitual", "peculiarity", "distinctive", "typical", "charm", "signature", "hallmark", "personality", "quintessence"],
    "use cases": ["use cases", "usages", "uses", "applications", "utilization", "purposes", "functions", "roles", "benefits", "practical applications", "utilitarian uses", "pragmatic purposes", "applications", "utility", "versatility", "advantages", "scenarios", "possibilities", "potentials", "capabilities", "effectiveness"],
    "general appearance": ["general appearance", "appearance", "outward appearance", "overall look", "visual aspect", "physical appearance", "external features", "outward characteristics", "external appearance", "overall appearance", "overall impression", "aesthetic", "panorama", "landscape", "scenery", "visage", "semblance", "vista", "countenance", "panoply", "regalia"],
    "rare appearances": ["rare appearances", "rare", "uncommon", "unusual", "scarce", "infrequent", "atypical", "exotic", "unique", "uncommon occurrences", "unusual sightings", "exceptional", "extraordinary", "seldom seen", "curiosities", "rarities", "marvels", "wonders", "phenomena", "oddities", "eccentricities", "peculiarities", "anomalies"],
    "availability": ["availability", "available", "presence", "existence", "accessibility", "presence", "occurrence", "availability", "abundance", "availability", "accessibility", "readiness", "availability", "convenience", "proliferation", "ubiquity", "prevalence", "circulation", "dispersion", "distribution", "readily accessible", "readily obtainable"],
    "description": ["describe", "description", "detail", "elaborate", "explain", "depict", "portray", "interpret", "characterize", "define", "narrate", "represent", "illustrate", "outline", "summarize", "enumerate", "specify", "delineate", "depiction", "depicting", "depicted", "rendering", "rendered", "depicts", "illustrative", "descriptive", "know", "tell"]
}


# Define function to extract attributes from user query using string tokenization
def extract_attributes(query):
    tokens = nltk.word_tokenize(query.lower())
    attributes = set()
    for token in tokens:
        for attr, keywords in attribute_keywords.items():
            for keyword in keywords:
                if keyword in token:
                    attributes.add(attr)
                    break
    return attributes

# Modify the button click functionality to handle different combinations of the asterisk and flower name
if st.button('Submit'):
    user_response = user_input.lower()
    if user_response != 'bye':
        greeting_response = greeting(user_response)
        if greeting_response is not None:
            st.write("FlowerBot:", greeting_response)
        else:
            # Check if the input text contains a flower name and attribute
            flower_names_in_query = []

            for name in flower_names:
                if name.lower() in user_response:
                    flower_names_in_query.append(name.lower())
            
            if len(flower_names_in_query) > 1:
                st.write("FlowerBot:", "Sorry, only one flower's information can be shown at a time.")
            elif len(flower_names_in_query) == 1:
                flower_name = flower_names_in_query[0]
                st.session_state['flower_name'] = flower_name
                # Check if the user query includes the flower name and asterisk in different positions
                if f"*{flower_name}" in user_response or f"{flower_name}*" in user_response or f"{flower_name} *" in user_response:
                    # Retrieve all attributes for the specified flower name
                    attributes = flower_descriptions.get(flower_name, {}).keys()
                    if attributes:
                        for attribute in attributes:
                            response_text = get_flower_attribute_response(flower_name, attribute)
                            st.write("FlowerBot:", response_text)
                    else:
                        st.write("FlowerBot:", "No attributes found for this flower.")
                else:
                    attributes = extract_attributes(user_response)
                    if attributes:
                        for attribute in attributes:
                            response_text = get_flower_attribute_response(flower_name, attribute)
                            st.write("FlowerBot:", response_text)
                    else:
                        st.write("FlowerBot:", "I didn't understand that. Please mention a valid attribute or enter the flower name followed by an asterisk (*) in any position to display all attributes.")
            else:
                st.write("FlowerBot:", "I didn't understand that. Please mention a flower name.")
    else:
        st.write("FlowerBot:", "Bye! Take care.")



if st.session_state['attribute_input']:
    attribute = st.session_state['attribute_input'].lower()
    flower_name = st.session_state['flower_name']
    if flower_name:
        response_text = get_flower_attribute_response(flower_name, attribute)
        st.write("FlowerBot:", response_text)
        st.session_state['attribute_input'] = ''
        st.session_state['flower_name'] = None
    else:
        st.write("FlowerBot:", "Please mention a flower name first.")
