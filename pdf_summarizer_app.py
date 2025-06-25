import streamlit as st
import pdfplumber
from transformers import pipeline

# Load the summarization pipeline
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_model()

# PDF text extraction function
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Streamlit UI
st.title("📄 PDF Summarizer using Hugging Face Transformers")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    st.subheader("Extracted Text Preview:")
    st.text_area("Full Text", pdf_text[:3000] + ("..." if len(pdf_text) > 3000 else ""), height=300)

    if st.button("Generate Summary"):
        if pdf_text.strip():
            with st.spinner("Generating summary..."):
                chunks = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 1000)]
                summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]["summary_text"] for chunk in chunks]
                full_summary = " ".join(summaries)
            st.subheader("📝 Summary:")
            st.write(full_summary)
        else:
            st.warning("No text found in the PDF.")
