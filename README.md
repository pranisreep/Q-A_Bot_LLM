# Document Q&A Bot (Streamlit)

A simple GUI chatbot that answers questions about an uploaded **PDF** or **.txt** file using OpenAI.  
It extracts text, splits it into chunks, builds embeddings for retrieval, and asks the LLM to answer using only the most relevant chunk(s).  
Includes basic error handling for missing key, unreadable files, API failures, and “no answer found”.

#Features

- 📄 Upload PDF or TXT
- 🔍 Text extraction + character-based chunking
- 🧭 Simple retrieval using OpenAI embeddings + cosine similarity
- 🤖 Answer generation from relevant context only
- 🛡️ Basic error handling

# Requirements

- Python 3.10+ (Windows / macOS / Linux)
- An OpenAI API key

#Setup (Windows / PowerShell shown)

1. **Open the project folder in VS Code.**
2. (Optional but recommended) Create a virtual environment:
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```
