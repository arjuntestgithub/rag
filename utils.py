import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import DB_FAISS_PATH




# =========================
# EMBEDDING MODEL
# =========================

def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


# =========================
# CREATE VECTOR DATABASE
# =========================

def create_vector_db(pdf_path: str):

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"✅ Total Pages Loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = text_splitter.split_documents(documents)

    print(f"✅ Total Chunks Created: {len(docs)}")

    embeddings = get_embeddings()

    db = FAISS.from_documents(docs, embeddings)

    db.save_local(DB_FAISS_PATH)

    print("✅ FAISS vector database created successfully")


# =========================
# LOAD VECTOR DB
# =========================

def load_vector_db():

    embeddings = get_embeddings()
    print("-=-=-=-=-=-=",DB_FAISS_PATH)

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db