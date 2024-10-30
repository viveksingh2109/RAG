import PyMuPDF  # PyMuPDF
import re
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer
from langdetect import detect
from groq import Groq
import pytesseract
from PIL import Image
import io
import time
from functools import lru_cache
from deep_translator import GoogleTranslator
import numpy as np
import streamlit as st
import speech_recognition as sr
import json
from datetime import datetime
import requests
import sqlite3
from streamlit_lottie import st_lottie

# Load sentence transformer model
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_sentence_transformer()

# Perform OCR on images
def perform_ocr(image):
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        raise Exception(f"Error performing OCR: {str(e)}")

# Extract text from PDF (including OCR on images)
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")
            # Extract images and perform OCR
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                image_text = perform_ocr(image)
                if image_text:
                    text += f"\n{image_text}\n"
        doc.close()
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

# Clean extracted text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[^\w\s,.]', '', text)
    return text.strip()

# Chunk text for semantic search
def chunk_text(text, chunk_size=300, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = ' '.join(words[start:end])
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Store chunks in SQLite
def store_chunks_in_sqlite(chunks):
    try:
        conn = sqlite3.connect('document_chunks.db')
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS pdf_chunks')
        c.execute('CREATE TABLE pdf_chunks (id INTEGER PRIMARY KEY, chunk TEXT NOT NULL)')
        c.executemany('INSERT INTO pdf_chunks (chunk) VALUES (?)', [(chunk,) for chunk in chunks if chunk.strip()])
        conn.commit()
        conn.close()
    except Exception as e:
        raise Exception(f"Database error: {str(e)}")

# Generate embeddings for chunks
def generate_embeddings(chunks):
    try:
        embeddings = model.encode(chunks, convert_to_tensor=True)
        return embeddings.cpu().detach().numpy()
    except Exception as e:
        raise Exception(f"Error generating embeddings: {str(e)}")

# Create FAISS index for embeddings
def create_faiss_index(embeddings):
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, 'document_index.faiss')
        return index
    except Exception as e:
        raise Exception(f"Error creating FAISS index: {str(e)}")

# Load FAISS index
def load_faiss_index():
    try:
        return faiss.read_index('document_index.faiss')
    except Exception as e:
        raise Exception(f"Error loading FAISS index: {str(e)}")

# Retrieve semantic chunks for a query
def retrieve_semantic_chunks(query, index, chunks, top_n=5):
    try:
        query_embedding = model.encode([query], convert_to_tensor=True).cpu().detach().numpy()
        distances, indices = index.search(query_embedding, min(top_n, len(chunks)))
        return [chunks[i] for i in indices[0] if i < len(chunks)]
    except Exception as e:
        raise Exception(f"Error in semantic search: {str(e)}")

# Generate response based on relevant chunks
def generate_response(query, relevant_chunks):
    if not query or not relevant_chunks:
        return "I apologize, but I don't have enough context to provide a meaningful response."
    
    relevant_document = ' '.join(relevant_chunks)
    prompt = f"""
    You are an AI assistant. Your task is to answer questions based solely on the provided information.
    Relevant information from the document:
    {relevant_document}
    User query: {query}
    Response:
    """
    
    try:
        key = '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'
        client = Groq(api_key=key)
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model="llama3-groq-70b-8192-tool-use-preview",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1024,
                    top_p=0.65,
                    stream=True
                )
                
                full_response = []
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        full_response.append(chunk.choices[0].delta.content)
                
                return ''.join(full_response)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e
                    
    except Exception as e:
        raise Exception(f"Error generating response: {str(e)}")

# Translation functions
@lru_cache(maxsize=1000)
def cached_translation(text, source_lang, target_lang):
    try:
        if source_lang == target_lang:
            return text
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        return translator.translate(text)
    except Exception as e:
        raise Exception(f"Translation error: {str(e)}")

def translate_to_english(text):
    try:
        detected_lang = detect(text)
        if detected_lang != 'en':
            translator = GoogleTranslator(source=detected_lang, target='en')
            text = translator.translate(text)
        return text, detected_lang
    except Exception as e:
        raise Exception(f"Translation error: {str(e)}")

def translate_response(text, target_lang):
    try:
        return cached_translation(text, 'en', target_lang)
    except Exception as e:
        raise Exception(f"Translation error: {str(e)}")

# Get speech input and translate to English if necessary
def get_speech_input():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening... Speak now!")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            try:
                text = recognizer.recognize_google(audio)
                detected_lang = detect(text)
                if detected_lang != 'en':
                    translator = GoogleTranslator(source=detected_lang, target='en')
                    text = translator.translate(text)
                return text, detected_lang
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None, None
            except sr.RequestError as e:
                print(f"Could not request results: {str(e)}")
                return None, None
    except Exception as e:
        print(f"Error accessing microphone: {str(e)}")
        return None, None

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        st.error(f"Error loading animation: {str(e)}")
        return None
    
# Lottie animations
lottie_upload = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_h4th9ofg.json")
lottie_chatbot = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_uhatvnlx.json")

# Set up Streamlit UI
st.markdown("<h1 style='text-align: center; color: white;'>Multilingual AI Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>Ask questions in any preferred language!</h4>", unsafe_allow_html=True)

# Language selector
languages = {
    'en': 'English', 'hi': 'Hindi', 'te': 'Telugu', 'ta': 'Tamil', 'ml': 'Malayalam',
    'kn': 'Kannada', 'bn': 'Bengali', 'gu': 'Gujarati', 'mr': 'Marathi', 'pa': 'Punjabi', 'ur': 'Urdu'
}
selected_language = st.selectbox("Select your preferred language", options=list(languages.values()), index=0)
selected_lang_code = [k for k, v in languages.items() if v == selected_language][0]

# Upload PDF file
if lottie_upload:
    st_lottie(lottie_upload, height=200, key="upload")

pdf_file = st.file_uploader("Choose a PDF file of the document", type="pdf")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Speech input and text input container
input_container = st.container()
with input_container:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🎤 Speak", key="speak_button"):
            with st.spinner("Listening..."):
                speech_input, detected_lang = get_speech_input()
                if speech_input:
                    st.session_state.speech_input = speech_input
                    st.session_state.detected_lang = detected_lang
                    st.rerun()

    with col2:
        text_input = st.text_input("Type your question:", key="text_input")

# Process input (either speech or text)
if hasattr(st.session_state, 'speech_input'):
    prompt = st.session_state.speech_input
    detected_lang = st.session_state.detected_lang
    del st.session_state.speech_input
    del st.session_state.detected_lang
elif text_input:
    prompt, detected_lang = translate_to_english(text_input)
else:
    prompt, detected_lang = None, None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process PDF and user prompt if available
if pdf_file and prompt:
    if "document_processed" not in st.session_state:
        with st.spinner("Processing PDF with OCR..."):
            try:
                raw_text = extract_text_from_pdf(pdf_file)
                cleaned_text = clean_text(raw_text)
                chunks = chunk_text(cleaned_text)
                store_chunks_in_sqlite(chunks)
                chunk_embeddings = generate_embeddings(chunks)
                faiss_index = create_faiss_index(chunk_embeddings)
                st.session_state.document_processed = True
                st.session_state.chunks = chunks
                st.success("Document processed and indexed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing the document: {e}")
                st.session_state.document_processed = False

    # If document is processed, generate response
    if st.session_state.get("document_processed", False):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing response..."):
                try:
                    faiss_index = load_faiss_index()
                    relevant_chunks = retrieve_semantic_chunks(prompt, faiss_index, st.session_state.chunks)
                    english_response = generate_response(prompt, relevant_chunks)
                    
                    # Translate response to user's language
                    final_response = translate_response(english_response, selected_lang_code)
                    
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})

                    # View relevant document context
                    with st.expander("View Source Context"):
                        st.info("These are the relevant sections from the document that were used:")
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.write(f"Section {i}:")
                            st.write(chunk)


                except Exception as e:
                    st.error(f"Error generating response: {e}")

else:
    if not pdf_file:
        st.info("Please upload a PDF document to start chatting.")
    elif not st.session_state.get("document_processed", False):
        st.warning("Please wait while the document is being processed...")

# Export conversation
if st.session_state.messages:
    if st.button("Export Conversation"):
        conversation_export = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="Download Conversation",
            data=conversation_export,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
