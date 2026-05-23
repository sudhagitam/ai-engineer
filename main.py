import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq
import chromadb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("pdf_docs")

def split_text(text, chunk_size=2000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

@app.get("/")
def root():
    return {"status": "PDF Chat API is running"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed.")
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")
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

@app.post("/ask-groq")
async def ask_groq_directly(payload: dict):
    question = payload.get("question")
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": question}
        ]
    )
    return {"answer": response.choices[0].message.content}