import streamlit as st
import torch
from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer
model_name = "model"  # Path to your trained model folder
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

st.set_page_config(page_title="English to French Translator", page_icon="🇫🇷")

# Streamlit UI
st.title("English to French Translation")
st.write("Enter an English sentence below to translate it into French.")

# Input text
input_text = st.text_area("English Text", placeholder="Type your sentence here...", label_visibility="collapsed")

# Translate button
if st.button("Translate"):
    if input_text.strip():
        # Tokenize and translate
        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display the result
        st.subheader("Translated French Text")
        st.write(translated_text)
    else:
        st.error("Please enter a valid English sentence.")