import streamlit as st
import numpy as np
import os
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.compat.v1.reset_default_graph()

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


trained_model = load_model('saved_model.h5')
print('@@ Model loaded')


def predict_disease(model, img):
    img = img.resize((150, 150))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_labels = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

    return class_labels[class_index]

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

st.set_page_config(page_title="Maize Project", page_icon=":female-student:", layout="centered")

st.header("Maize Crop Foliage Disease Classification")

st.divider()

image_path = st.file_uploader(label="Choose an image...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if image_path is not None:
    image = Image.open(image_path).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True, width=50, )

    result = predict_disease(trained_model, image)

    if result == 'Blight':
        st.header("Blight")
    elif result == 'Common_Rust':
        st.header("Common Rust")
    elif result == 'Gray_Leaf_Spot':
        st.header("Gray Leaf Spot")
    else:
        st.header("Healthy")