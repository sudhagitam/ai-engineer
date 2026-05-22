"use client";
import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploaded, setUploaded] = useState(false);

  const API = "http://localhost:8000"; // change to Railway URL after deploy

  const uploadPDF = async () => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API}/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setUploaded(true);
    alert(data.message);
  };

  const askQuestion = async () => {
    setLoading(true);
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    setAnswer(data.answer);
    setLoading(false);
  };

  return (
    <main className="max-w-2xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-8">📄 PDF Chat</h1>

      {/* Upload Section */}
      <div className="mb-8 p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-4">Upload PDF</h2>
        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files[0])}
          className="mb-4 block"
        />
        <button
          onClick={uploadPDF}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Upload
        </button>
        {uploaded && <p className="text-green-500 mt-2">✅ PDF Ready!</p>}
      </div>

      {/* Question Section */}
      <div className="p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-4">Ask a Question</h2>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What is this document about?"
          className="w-full border p-2 rounded mb-4"
        />
        <button
          onClick={askQuestion}
          disabled={loading || !uploaded}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:opacity-50"
        >
          {loading ? "Thinking..." : "Ask"}
        </button>

        {answer && (
          <div className="mt-4 p-4 bg-gray-100 rounded">
            <p className="font-semibold">Answer:</p>
            <p>{answer}</p>
          </div>
        )}
      </div>
    </main>
  );
}