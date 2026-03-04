import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Ensure your API key is set in your environment variables
client = Groq(api_key=os.environ.get("GROQ_API_KEY_1"))

try:
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Ping"}],
        model="llama3-8b-8192",
    )
    print(chat_completion.choices[0].message.content)
    print("Server connection successful!")
except Exception as e:
    print(f"Error connecting to server: {e}")
