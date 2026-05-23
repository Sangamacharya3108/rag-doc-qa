import os
import requests
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(uploaded_file):
    # Extracts all text from the pages of a PDF file.
    reader = PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text is not None:
            text_parts.append(page_text)
    return "".join(text_parts)


def chunk_text(text):
    # Splits the input text into overlapping chunks of 500 characters.
    chunks = []
    if not text:
        return chunks
    chunk_size = 500
    chunk_overlap = 100
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def build_vector_store(chunks):
    # Encodes text chunks into FAISS index.
    if not chunks:
        return None, []
    embeddings = embedder.encode(chunks)
    embeddings_array = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    return index, chunks


def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    # Finds the most similar chunks from the index for the query.
    if index is None or not chunks:
        return []
    
    query_lower = query.lower()
    general_keywords = ["summarize", "summary", "overview", "what is this document about", "what is this pdf about", "main topics", "main points", "key points"]
    if any(keyword in query_lower for keyword in general_keywords):
        return chunks
        
    query_vector = embedder.encode([query])
    query_array = np.array(query_vector, dtype=np.float32)
    _, indices = index.search(query_array, min(top_k, len(chunks)))
    
    relevant = []
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            relevant.append(chunks[idx])
    return relevant


def generate_answer(query, relevant_chunks, chat_history):
    # Generates a grounded response to the query using OpenRouter API.
    context = "\n\n".join(relevant_chunks)
    
    system_prompt = (
        "You are a document assistant. You have been given context extracted from a PDF document.\n\n"
        "Answer the user's question based on the context below.\n\n"
        "If the user asks for a summary, overview, or what the document is about, provide a concise summary using the context.\n\n"
        "If the user asks a specific question and the answer is not in the context, say exactly: I could not find that in the document.\n\n"
        "Do not make up any information. Only use what is in the context.\n\n"
        f"CONTEXT:\n{context}"
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    for role, content in chat_history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": messages
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            payload["model"] = "openrouter/free"
            response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()
        return answer
    except (requests.exceptions.RequestException, KeyError, IndexError):
        return "Error: Unable to retrieve answer from the model. Please check your API key or network connection."
