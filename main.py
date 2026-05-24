import os
import io
import smtplib
import hashlib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq
from tavily import TavilyClient
import httpx
from bs4 import BeautifulSoup
import chromadb
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clients
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("pdf_docs")

# Track sent news hashes to avoid duplicates
sent_news_hashes = set()

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def split_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def send_email_alert(subject, body):
    """Send Gmail HTML alert"""
    try:
        sender   = os.environ.get("ALERT_EMAIL")
        password = os.environ.get("ALERT_EMAIL_PASSWORD")
        receiver = os.environ.get("RECEIVER_EMAIL")

        if not all([sender, password, receiver]):
            print("❌ Email env vars missing")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = receiver

        html = f"""
        <html>
        <body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
            <div style="background:#1a1a2e;color:white;padding:20px;border-radius:8px;">
                <h2 style="color:#4ade80;">🚨 USCIS News Alert</h2>
                <p style="color:#ccc;font-size:12px;">
                    {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
                </p>
            </div>
            <div style="padding:20px;background:#f9f9f9;border-radius:8px;margin-top:10px;">
                {body.replace(chr(10), '<br>')}
            </div>
            <div style="padding:10px;text-align:center;color:#888;font-size:11px;">
                <p>Powered by PDF Chat AI + Groq + Tavily</p>
                <a href="https://www.uscis.gov/newsroom"
                   style="color:#2563eb;">Visit USCIS Newsroom →</a>
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())

        print(f"✅ Email sent: {subject}")
        return True

    except Exception as e:
        print(f"❌ Email failed: {str(e)}")
        return False


def check_uscis_news():
    """Scheduled job — runs every 6 hours"""
    print(f"🔍 Checking USCIS news at {datetime.now()}")
    try:
        results = tavily_client.search(
            query="USCIS immigration news update 2026",
            search_depth="basic",
            max_results=5
        )

        new_articles = []
        for result in results["results"]:
            url_hash = hashlib.md5(result["url"].encode()).hexdigest()
            if url_hash not in sent_news_hashes:
                sent_news_hashes.add(url_hash)
                new_articles.append(result)

        if new_articles:
            context = "\n\n".join([
                f"Title: {r.get('title', 'N/A')}\nURL: {r['url']}\nContent: {r['content']}"
                for r in new_articles
            ])

            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an immigration news assistant.
                        Summarize the USCIS news in clear bullet points.
                        Focus on: policy changes, processing times,
                        new rules, H1B/EB updates, and action items
                        for applicants. Be concise and clear."""
                    },
                    {
                        "role": "user",
                        "content": f"Summarize these USCIS news articles:\n\n{context}"
                    }
                ]
            )

            summary = response.choices[0].message.content
            sources = "\n\n📰 Sources:\n" + "\n".join([
                f"• {r.get('title', r['url'])}: {r['url']}"
                for r in new_articles
            ])

            send_email_alert(
                subject=f"🚨 USCIS News Alert - {datetime.now().strftime('%B %d, %Y')}",
                body=summary + sources
            )
        else:
            print("ℹ️ No new USCIS articles found")

    except Exception as e:
        print(f"❌ USCIS check failed: {str(e)}")


# Start scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    check_uscis_news,
    "interval",
    hours=6,
    id="uscis_news_check"
)
scheduler.start()
print("✅ USCIS scheduler started — checking every 6 hours")


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "PDF Chat API is running"}


@app.get("/alert-status")
async def alert_status():
    return {
        "scheduler_running": scheduler.running,
        "news_tracked": len(sent_news_hashes),
        "next_check": str(scheduler.get_job("uscis_news_check").next_run_time)
    }


@app.post("/trigger-uscis-check")
async def trigger_uscis_check():
    check_uscis_news()
    return {"message": "USCIS news check triggered!"}


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
    history  = payload.get("history", [])
    results  = collection.query(query_texts=[question], n_results=3)
    context  = "\n".join(results["documents"][0])
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
    history  = payload.get("history", [])
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
    url      = payload.get("url")
    question = payload.get("question", "Summarize this page")
    history  = payload.get("history", [])
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, timeout=10,
                headers={"User-Agent": "Mozilla/5.0"}
            )
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
            "sources": [url]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")


@app.post("/search-web")
async def search_web(payload: dict):
    question = payload.get("question")
    history  = payload.get("history", [])
    try:
        search_results = tavily_client.search(
            query=question,
            search_depth="basic",
            max_results=5
        )
        context = ""
        sources = []
        for result in search_results["results"]:
            context += f"\nSource: {result['url']}\n{result['content']}\n"
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