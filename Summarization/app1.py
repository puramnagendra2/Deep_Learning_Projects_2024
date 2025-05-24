import streamlit as st
from PyPDF2 import PdfReader
from transformers import BartTokenizer, BartForConditionalGeneration
import spacy
import time
from evaluate import load
import torch

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load ROUGE evaluator
rouge = load("rouge")

# Streamlit UI config
st.set_page_config(page_title="PDF Summariser", page_icon=":closed_book:", layout="wide")

# Caching model and tokenizer
@st.cache_resource
def load_model(distilled=False):
    if distilled:
        model_name = 'sshleifer/distilbart-cnn-12-6'
    else:
        model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks using SpaCy
def split_text_into_chunks(text, tokenizer, max_token_length=1024):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(tokenizer.encode(chunk + " " + sentence)) < max_token_length:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Function to summarize a chunk
def summarize_chunk(text, tokenizer, model, max_words=100):
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=max_words * 2,
        min_length=max_words,
        num_beams=2,
        length_penalty=1.0,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to evaluate performance using ROUGE
@st.cache_data
def evaluate_summary(generated, reference):
    return rouge.compute(predictions=[generated], references=[reference])

# Function to summarize full PDF
def summarize_pdf(pdf_path, tokenizer, model, max_words_per_chunk=100):
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(raw_text, tokenizer)

    st.write(f"Total chunks to summarize: {len(chunks)}")

    summaries = []
    start_time = time.time()
    for i, chunk in enumerate(chunks):
        st.write(f"Summarizing chunk {i + 1}...")
        summary = summarize_chunk(chunk, tokenizer, model, max_words=max_words_per_chunk)
        summaries.append(summary)
    total_time = round(time.time() - start_time, 2)

    final_summary = "\n\n".join(summaries)
    return final_summary, raw_text, total_time

# UI Layout
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("AI Powered Text Summariser from PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    model_choice = st.selectbox("Select summarization mode:", ["Accurate (BART Large)", "Fast (Distilled BART)"])
    distilled = model_choice == "Fast (Distilled BART)"

    max_words_per_chunk = st.slider("Max words per chunk", 50, 300, 100)

    if st.button("Summarise File Data"):
        if uploaded_file is not None:
            tokenizer, model = load_model(distilled)
            summary, original_text, infer_time = summarize_pdf(uploaded_file, tokenizer, model, max_words_per_chunk)
            
            # Display Results
            original, summary_col = st.columns(2)
            with original:
                st.header("Original PDF Content")
                st.text_area("Content", original_text, height=400)
            with summary_col:
                st.header("Summarised Content")
                st.text_area("Summary", summary, height=400)

            # Evaluation (ROUGE against original if it's short)
            if len(original_text.split()) < 2048:
                scores = evaluate_summary(summary, original_text)
                st.subheader("Evaluation Metrics")
                st.write({k: round(v, 4) for k, v in scores.items()})

            st.success(f"Inference time: {infer_time} seconds")
        else:
            st.warning("Please upload a PDF file.")
