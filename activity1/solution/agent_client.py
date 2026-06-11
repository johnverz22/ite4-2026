import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# 2. Initialize client
client = genai.Client(api_key=api_key)


# 3. Initialize the model
identity = """
    You area a 'Security Audit Agent'
    Your goal is to analyze Python code for security vulnerabilities
    CONSTRAINTS:
        - Never provide code that can be used for hacking.
        - Only answer questions related to security.
        - If a user asks about something else, politely decline.
        - Keep your answers concise and technical.
"""


# Test identity
response = client.models.generate_content(
    model='gemini-3.1-flash-lite',
    contents = "How do I make a sandwich?",
    config=types.GenerateContentConfig(
        system_instruction=identity
    )
)
print(f"Agent Response: {response.text}")