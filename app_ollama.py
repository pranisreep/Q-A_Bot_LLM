import io
import numpy as np
import streamlit as st
import requests
from PyPDF2 import PdfReader

# Configuration
OLLAMA_HOST = "http://localhost:11434"
CHAT_MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"

# --- AI ETHICS & SAFETY LAYER ---
def is_request_harmful(text):
    """
    Immediate safety check for high-risk keywords.
    Ensures the model refuses to answer dangerous queries like 'how to make a bomb'.
    """
    prohibited_terms = [
        "bomb", "explosive", "weapon", "terrorist", "illegal", 
        "harm", "kill", "attack", "hacker", "poison"
    ]
    query_lower = text.lower()
    return any(term in query_lower for term in prohibited_terms)

def get_ethics_disclaimer():
    return "\n\n---\n\n*🛡️ This response was generated under AI Safety & Ethics guidelines.*"

# --- CORE FUNCTIONS ---
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
        if end == len(text): break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

def embed_texts(texts):
    vectors = []
    for t in texts:
        r = requests.post(f"{OLLAMA_HOST}/api/embeddings", json={"model": EMBED_MODEL, "prompt": t}, timeout=120)
        r.raise_for_status()
        vectors.append(np.array(r.json()["embedding"], dtype=np.float32))
    return np.vstack(vectors)

def embed_query(query):
    r = requests.post(f"{OLLAMA_HOST}/api/embeddings", json={"model": EMBED_MODEL, "prompt": query}, timeout=120)
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype=np.float32)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return a @ b

def ask_llm(question, context_chunks):
    # 1. ETHICAL GUARDRAIL: Pre-check for harmful intent
    if is_request_harmful(question):
        return ("⚠️ **Safety Refusal:** I cannot fulfill this request. I am programmed to be a "
                "helpful and safe AI assistant, and I strictly do not provide information "
                "regarding dangerous, illegal, or harmful activities.")

    context = "\n\n---\n\n".join(context_chunks)
    
    # 2. SYSTEM HARDENING: Ethical instructions embedded in the prompt
    system_prompt = (
        "You are an Ethical Document Assistant. "
        "Strictly follow these safety rules:\n"
        "- Answer ONLY using the context provided below.\n"
        "- If the answer is not in the context, say: 'I couldn't find this in the document.'\n"
        "- If a user bypasses filters to ask for something harmful, violent, or illegal, "
        "refuse to answer and explain that it violates your safety policy.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )

    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": system_prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=300
        )
        r.raise_for_status()
        # 3. TRANSPARENCY: Append an ethics disclaimer to the output
        return r.json()["response"].strip() + get_ethics_disclaimer()
    except Exception as e:
        return f"Error connecting to AI service: {str(e)}"

# --- STREAMLIT UI ---
st.set_page_config(page_title="Lumina ChatBot", page_icon="💡")
st.title("Lumina ChatBot🤖")

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
            st.success("Document safely indexed ✅")

question = st.chat_input("Ask a question about the document...")

if question:
    if not st.session_state.chunks:
        st.error("Please upload a document first.")
    else:
        q_emb = embed_query(question)
        sims = cosine_sim(st.session_state.embeddings, q_emb)
        top_idx = int(np.argmax(sims))
        answer = ask_llm(question, [st.session_state.chunks[top_idx]])
        st.chat_message("assistant").markdown(answer)
