from groq import Groq

client = Groq()  # automatically picks up GROQ_API_KEY

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful banking assistant."},
        {"role": "user", "content": "Explain what a SWIFT payment is in simple terms."}
    ]
)

print(response.choices[0].message.content)