import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq
from tavily import TavilyClient
import httpx
from bs4 import BeautifulSoup
import chromadb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("pdf_docs")

def split_text(text, chunk_size=1000, overlap=100):
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
    history = payload.get("history", [])
    results = collection.query(query_texts=[question], n_results=3)
    context = "\n".join(results["documents"][0])
    messages = [
        {"role": "system", "content": f"Answer based on this PDF context:\n{context}"}
    ]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": question})
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return {"answer": response.choices[0].message.content}

@app.post("/ask-groq")
async def ask_groq_directly(payload: dict):
    question = payload.get("question")
    history = payload.get("history", [])
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": question})
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return {"answer": response.choices[0].message.content}

@app.post("/fetch-url")
async def fetch_url(payload: dict):
    url = payload.get("url")
    question = payload.get("question", "Summarize this page")
    history = payload.get("history", [])
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10,
                headers={"User-Agent": "Mozilla/5.0"})
            html = response.text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)[:4000]
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Here is the content from {url}:\n\n{text}"}
        ]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": question})
        ai_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return {
            "answer": ai_response.choices[0].message.content,
            "url": url,
            "sources": [url]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

@app.post("/search-web")
async def search_web(payload: dict):
    question = payload.get("question")
    history = payload.get("history", [])
    try:
        search_results = tavily_client.search(
            query=question,
            search_depth="basic",
            max_results=5
        )
        context = ""
        sources = []
        for result in search_results["results"]:
            context += f"\nSource: {result['url']}\n"
            context += f"{result['content']}\n"
            sources.append(result['url'])
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful AI assistant with access to 
                real-time web search results. Answer based on these 
                current web results:\n\n{context}\n\n
                Always mention which sources you used."""
            }
        ]
        for msg in history:
            messages.append(msg)
        messages.append({"role": "user", "content": question})
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return {
            "answer": response.choices[0].message.content,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Web search failed: {str(e)}")