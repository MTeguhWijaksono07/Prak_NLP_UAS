"""
config.py - Konfigurasi utama sistem NLP/LLM
Mengatur LangSmith tracing, model, dan parameter global
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LangSmith Configuration
# LangSmith digunakan untuk monitoring, tracing, dan evaluasi
# ============================================================
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT   = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGCHAIN_API_KEY    = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT    = os.getenv("LANGCHAIN_PROJECT", "UAS-NLP-LLM-System")

os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"]   = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_PROJECT"]    = LANGCHAIN_PROJECT
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# ============================================================
# Model Configuration (GROQ)
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "llama-3.1-8b-instant"  # Model GROQ aktif dan cepat
EMBED_MODEL  = "all-MiniLM-L6-v2"  # Embedding lokal dari sentence-transformers
TEMPERATURE  = 0.7
MAX_TOKENS   = 1024

# ============================================================
# System Prompt
# ============================================================
SYSTEM_PROMPT = """Kamu adalah asisten cerdas berbasis NLP/LLM yang dibangun
menggunakan LangChain, LangGraph, dan LangSmith. Jawablah pertanyaan dengan
jelas, akurat, dan dalam Bahasa Indonesia. Jika kamu tidak tahu jawabannya,
katakan dengan jujur."""