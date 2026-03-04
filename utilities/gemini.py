"""
gemini.py

Quick utility script — lists all Google Gemini models available for
content generation using the generativeai SDK. No functions; runs
directly when executed.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

print("Models available for text/content generation:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)