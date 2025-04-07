import streamlit as st
from PyPDF2 import PdfReader
from transformers import BartTokenizer, BartForConditionalGeneration

# Libraries for summarizing text
from transformers import BartTokenizer, BartForConditionalGeneration
import spacy

nlp = spacy.load("en_core_web_sm")


st.set_page_config(page_title="PDF Summariser",page_icon=":closed_book:", layout="wide")

# Streamlit UI

# All Functions

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks using SpaCy
def split_text_into_chunks(text, max_token_length=1024):
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

# Loading the BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Function to summarize chunks of text
def summarize_chunk(text, max_words=100):
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    
    summary_ids = model.generate(
        inputs,
        max_length=max_words * 2,  # BART counts tokens, not words. Words ~ half tokens
        min_length=max_words,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to summarize the entire text
def summarize_pdf(pdf_path, max_words_per_chunk=100):
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(raw_text)

    print(f"Total chunks to summarize: {len(chunks)}\n")

    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}...")
        summary = summarize_chunk(chunk, max_words=max_words_per_chunk)
        summaries.append(summary)

    final_summary = "\n\n".join(summaries)
    return final_summary


col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("AI Powered Text Summariser from PDF")
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Button to display file data
    if st.button("Summarise File Data"):
        if uploaded_file is not None:
            # Read the PDF file
            pdf_reader = PdfReader(uploaded_file)
            # max_words_per_chunk = st.slider("Max words per chunk", 50, 500, 100)
            max_words_per_chunk = 300
            summary = summarize_pdf(uploaded_file, max_words_per_chunk)
            
            # Display the content of the PDF
            # st.text_area("PDF Content", pdf_text, height=400)
            original, summary_col = st.columns(2)
            with original:
                st.header("Original PDF Content")
                pdf_text = extract_text_from_pdf(uploaded_file)
                st.text_area("Content", pdf_text, height=400)
            with summary_col:
                st.header("Summarised Content")
                st.text_area("Summary", summary, height=400)
        else:
            st.warning("Please upload a PDF file.")