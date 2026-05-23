# PDF Q&A System with RAG

This is a simple Document Question Answering (Q&A) system built using Retrieval-Augmented Generation (RAG). You can upload any PDF document and ask questions about its content. The app will search for the most relevant parts of the document and generate an response based only on the text it found.

## Features

- PDF text extraction page by page
- Overlapping text chunking implemented in pure Python
- Vector indexing using FAISS and sentence embeddings
- Similarity search to find relevant information
- Clean chat interface with memory to ask follow-up questions
- Fallback logic to avoid hallucination (tells you if the answer isn't in the PDF)

## Tech Stack

- Streamlit (Frontend interface)
- PyPDF2 (PDF text extraction)
- sentence-transformers (all-MiniLM-L6-v2 for generating vector embeddings)
- FAISS (vector store for similarity search)
- OpenRouter API (Mistral-7B-Instruct / free models for answering questions)
- requests (making REST API calls)
- python-dotenv (managing environment variables)

## Project Structure

```text
rag-doc-qa/
├── app.py
├── rag_engine.py
├── requirements.txt
├── .env (needs to be created locally)
└── .gitignore
```

## Setup Instructions

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/Sangamacharya3108/rag-doc-qa.git
   cd rag-doc-qa
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your OpenRouter API key:
   ```text
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

## How to Run the App

Launch the Streamlit app with the following command:
```bash
streamlit run app.py
```

Open the link shown in your terminal (usually `http://localhost:8501`) in your web browser.

## Example Questions to Ask

Once you upload a PDF, you can try asking questions like:
- What is this document about?
- Summarize the main points of this document.
- What are the key takeaways?
- Tell me more about [specific keyword or topic]?

## Future Improvements

- Add support for scanned PDFs using OCR
- Highlight the source text in the PDF itself when an answer is generated
- Support multiple PDF uploads at the same time
- Improve search by using cross-encoders for reranking
