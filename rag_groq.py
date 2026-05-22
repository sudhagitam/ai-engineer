import os
import chromadb
from pypdf import PdfReader
from groq import Groq

# 1. Load PDF
print("Loading PDF...")
reader = PdfReader("Software Testing with Generative AI.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# 2. Split into chunks manually
def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = split_text(text)
print(f"Split into {len(chunks)} chunks")

# 3. Store in ChromaDB
print("Storing in vector database...")
client = chromadb.Client()
collection = client.create_collection("pdf_docs")
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# 4. Groq LLM
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # your key here

# 5. Chat loop
print("\n=== PDF Chat Ready ===")
print("Type 'exit' to quit\n")

while True:
    question = input("Your question:How an AI works in Software Testing? ")
    if question.lower() == "exit":
        break

    # Find relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=3
    )
    context = "\n".join(results["documents"][0])

    # Ask Groq with context
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"Answer based on this context:\n{context}"
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    print(f"\nAnswer: {response.choices[0].message.content}\n")