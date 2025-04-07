import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
# Load the saved model and tokenizer
model = load_model('text_generation_model.keras')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_len = 20

# Define the Streamlit UI
st.title("Next Word Prediction")
st.write("Enter a sentence, and the model will predict the next word.")

# Input from the user
input_sentence = st.text_input("Enter a sentence:")

# Predict the next word
if st.button("Predict"):
    if input_sentence.strip():
        # Convert input sentence to tokens
        token_list = tokenizer.texts_to_sequences([input_sentence])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        # Predict the next word
        predicted = np.argmax(model.predict(token_list), axis=-1)

        # Find the word corresponding to the predicted token
        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                next_word = word
                break

        st.write(f"Predicted next word: **{next_word}**")
    else:
        st.write("Please enter a valid sentence.")