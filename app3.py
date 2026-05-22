import os
import shutil
 
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from schemas import QueryRequest, QueryResponse
from rag import ask_question, reset_db
from utils import create_vector_db
from config import DB_FAISS_PATH, PDF_PATH, UPLOAD_FOLDER
from fastapi.responses import FileResponse
import uvicorn
 
 
app = FastAPI(
    title="FastAPI RAG FAISS Chatbot",
    version="1.0.0"
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================
# STARTUP EVENT
# =========================
 
@app.on_event("startup")
def startup_event():
 
    faiss_file = f"{DB_FAISS_PATH}/index.faiss"
 
    if not os.path.exists(faiss_file):
 
        print("⚡ Creating FAISS vector database...")
 
        create_vector_db(PDF_PATH)
 
        print("✅ Vector DB Ready")
 
 
# =========================
# HOME ROUTE
# =========================
 
# @app.get("/")
# def home():
 
#     return {
#         "message": "FastAPI RAG Chatbot Running"
#     }
 
 
# =========================
# CHAT ROUTE
# =========================
 
@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
 
    answer = ask_question(request.query)
 
    return QueryResponse(answer=answer)
 
 
# =========================
# UPLOAD PDF ROUTE
# =========================
 
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
 
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )
 
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
 
    # Create new vector DB
    create_vector_db(file_path)
 
    # Reset old DB from memory
    reset_db()
 
    return {
        "message": "PDF uploaded and vector DB created successfully"
    }
 
# from fastapi import FastAPI
 
 
# some_file_path = "large-video-file.mp4"
# app = FastAPI()
 
 
@app.get("/")
async def main():
    return FileResponse("index.html")
 
 
if __name__ == "__main__":
    uvicorn.run("app3:app", host="0.0.0.0", port=8000)