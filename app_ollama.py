import io
import numpy as np
import streamlit as st
import requests
from PyPDF2 import PdfReader


OLLAMA_HOST = "http://localhost:11434"
CHAT_MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"


def extract_text_from_pdf(file_bytes):
    text = ""
    reader = PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def extract_text_from_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore")

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed_texts(texts):
    vectors = []
    for t in texts:
        r = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=120
        )
        r.raise_for_status()
        vectors.append(np.array(r.json()["embedding"], dtype=np.float32))
    return np.vstack(vectors)

def embed_query(query):
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": query},
        timeout=120
    )
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype=np.float32)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return a @ b


def ask_llm(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "Answer ONLY using the context below. "
        "If the answer is not present, say: "
        "'I couldn't find this in the document.'\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,          # ✅ IMPORTANT FIX
            "options": {"temperature": 0.1}
        },
        timeout=300
    )

    r.raise_for_status()
    return r.json()["response"].strip()


st.set_page_config(page_title="Document Q&A Bot (Ollama)", page_icon="📄")
st.title("📄 Document Q&A Bot (Local - Ollama)")

uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.embeddings = None

if uploaded_file:
    with st.spinner("Reading and indexing document..."):
        data = uploaded_file.read()
        text = extract_text_from_pdf(data) if uploaded_file.name.endswith(".pdf") else extract_text_from_txt(data)

        if not text.strip():
            st.error("Could not extract text from this document.")
        else:
            st.session_state.chunks = chunk_text(text)
            st.session_state.embeddings = embed_texts(st.session_state.chunks)
            st.success("Document indexed successfully ✅")

question = st.chat_input("Ask a question about the document")

if question:
    if not st.session_state.chunks:
        st.error("Please upload a document first.")
    else:
        q_emb = embed_query(question)
        sims = cosine_sim(st.session_state.embeddings, q_emb)
        top_idx = int(np.argmax(sims))
        answer = ask_llm(question, [st.session_state.chunks[top_idx]])
        st.chat_message("assistant").markdown(answer)
