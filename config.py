from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("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")

DB_FAISS_PATH = "faiss_index"
PDF_PATH = "rag.pdf"
UPLOAD_FOLDER = "uploads"