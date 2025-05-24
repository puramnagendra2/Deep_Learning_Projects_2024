import streamlit as st
from streamlit_option_menu import option_menu
import os
import joblib
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Fake News Detection", layout="wide")

model = load_model("detection_model.keras")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

st.title("Fabricated Information Recognition with Embedding Techniques")

col1, col2, col3 = st.columns([0.5, 3, 0.5])

with col2:
    menu = option_menu(
        menu_title=None,
        options=['Prediction', 'Visualization'],
        icons=['code-slash', 'pie-chart'],
        orientation="horizontal"
    )

    if menu == 'Prediction':
        news = st.text_area(label="News Article", placeholder="Enter or Paste News")
        if st.button("Predict"):
            if news:
                user_sequences = tokenizer.texts_to_sequences([news])
                user_padded = pad_sequences(user_sequences, maxlen=500)

                if user_padded.shape[0] > 0:
                    predictions = model.predict(user_padded)
                    threshold = 0.5
                    classification = "Fake News" if predictions[0][0] > threshold else "Genuine"
                    st.header(f"Classification: {classification}")
                else:
                    st.warning("Unable to process the input. Please enter valid text.")

    elif menu == 'Visualization':
        images_dir = "./images"
        image_list = [d for d in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, d))]
        for i in image_list:
            st.image(f"./images/{i}")
            st.divider()

