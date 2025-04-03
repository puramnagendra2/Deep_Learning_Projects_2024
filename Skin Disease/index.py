import streamlit as st
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# https://www.kaggle.com/datasets/pritpal2873/multiple-skin-disease-detection-and-classification/data


# Load the model
model = load_model('skin_disease_model.keras')

def preprocess_image(img_path):
    img = Image.open(img_path)  # Open image from Streamlit uploader
    img = img.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Convert to NumPy array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

disease_labels = ['Acitinic Keratosis', 'Basal Cell Carcinoma', 'Dermatofibroma',
                  'Melanoma', 'Nevus', 'Pigmented Benign Keratosis',
                  'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion']

def predict_disease(img_path):
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    return disease_labels[predicted_class[0]]


st.set_page_config(page_title="Skin Disease Classification", layout="wide")

# Title
st.header("A CNN Framework for Precision Skin Lesion Analysis and Cancer Identification")

st.divider()

emp1, button, emp2 = st.columns([0.5, 2, 0.5], gap="large")

with button:
    image_path = st.file_uploader(label="Upload Image", type=["png", "jpg", "jpeg"])
    if image_path is not None:
        prediction_value = predict_disease(image_path)
        st.write(f"Prediction: {prediction_value}")

