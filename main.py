import os
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq
import chromadb
import io

app = FastAPI()

# Allow Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # ensure your key is set in env vars
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("pdf_docs")

# Split text helper
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    reader = PdfReader(io.BytesIO(contents))
    text = "".join(page.extract_text() for page in reader.pages)
    chunks = split_text(text)
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return {"message": f"Uploaded! Split into {len(chunks)} chunks"}

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question")
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n".join(results["documents"][0])
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"Answer based on this context:\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return {"answer": response.choices[0].message.content}

@app.get("/")
def root():
    return {"status": "PDF Chat API is running"}