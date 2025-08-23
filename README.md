# RAG: Multilingual PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot deployed on Streamlit that allows for PDF-based question-and-answer sessions in over 10 Indian languages.

## ✨ Features

* **PDF-Based Q&A:** Upload a PDF and ask questions about its content.
* **Multilingual Support:** Supports on-the-fly translation for both queries and responses in 10+ Indian languages.
* **Intelligent Text Extraction:** Extracts text from both regular PDF pages and images using OCR.
* **Fast Response Generation:** Integrates the Groq Llama-3 API for high-speed, accurate answers.
* **Efficient Semantic Search:** Uses FAISS for lightning-fast retrieval of relevant document segments.

## ⚙️ How It Works

1.  **Document Ingestion:** A user uploads a PDF. **PyMuPDF** extracts text, and **pytesseract** performs OCR on images to get text from them.
2.  **Indexing:** The extracted text is chunked into smaller segments. **Sentence-transformers** create vector embeddings for each chunk, which are stored in a **FAISS** index for efficient search.
3.  **User Query:** A user asks a question in their preferred language. **Langdetect** identifies the language, and **deep-translator** handles any necessary translation.
4.  **Retrieval:** The query is embedded, and **FAISS** performs a semantic search to find the most relevant text chunks from the PDF.
5.  **Generation:** The retrieved chunks and the user's original query are sent to the **Groq Llama-3 API**. The model uses this context to generate a precise and coherent answer, which is then translated back to the user's language.

## 🛠️ Tech Stack

* **Python**
* **Streamlit** (for the UI)
* **PyMuPDF** (for PDF text extraction)
* **pytesseract** (for OCR)
* **sentence-transformers** (for embeddings)
* **FAISS** (for vector search)
* **Groq API** (for LLM inference)
* **deep-translator** & **langdetect** (for multilingual support)

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/viveksingh2109/RAG.git](https://github.com/viveksingh2109/RAG.git)
    cd RAG
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up your Groq API Key:**
    * Sign up on [Groq Cloud](https://console.groq.com/keys) to get your API key.
    * Create a `.env` file in the project root and add your key:
        ```
        GROQ_API_KEY="your_api_key_here"
        ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

The application will open in your web browser.

## 💡 Usage

1.  Upload a PDF document.
2.  Wait for the processing to complete.
3.  Type your question in the text box.
4.  The chatbot will provide an answer based on the PDF's content.
