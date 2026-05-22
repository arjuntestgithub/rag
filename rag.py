import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, DB_FAISS_PATH
from utils import load_vector_db
GROQ_API_KEY = "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"


# =========================
# GLOBAL DB VARIABLE
# =========================

db = None


# =========================
# RESET DB
# =========================

def reset_db():

    global db

    db = None

    print("🔄 DB Reset Complete")


# =========================
# GET DB (LAZY LOADING)
# =========================

def get_db():

    global db

    if db is None:

        faiss_file = f"{DB_FAISS_PATH}/index.faiss"

        if not os.path.exists(faiss_file):
            raise Exception(
                "FAISS database not found. "
                "Please upload a PDF first."
            )

        print("⚡ Loading FAISS Database...")

        db = load_vector_db()

        print("✅ FAISS Database Loaded")

    return db


# =========================
# RETRIEVER FUNCTION
# =========================

def reload_rag_tool(user_question: str):

    database = get_db()

    docs = database.similarity_search(
        user_question,
        k=3
    )

    sentences = [
        doc.page_content
        for doc in docs
    ]

    print("✅ Retrieved Sentences")

    return sentences


# =========================
# RAG TOOL
# =========================

@tool
def rag_tool(user_query: str) -> str:
    """
    RAG based response
    """

    print("==========================================")
    print("RAG TOOL CALLED")
    print("==========================================")

    query = (
        user_query.get("user_query")
        if isinstance(user_query, dict)
        and "user_query" in user_query
        else user_query
    )

    docs = reload_rag_tool(query)

    if not docs:
        return "I do not have information about this query."

    final_outputs = "\n\n".join(docs)

    print("============= final output ===============")
    print(final_outputs)

    return final_outputs


# =========================
# LLM
# =========================

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)


# =========================
# SYSTEM PROMPT
# =========================

system_prompt = (
    "You are a helpful assistant. "
    "Always use the rag_tool to find relevant information. "
    "Use the tool EXACTLY ONCE to gather context, "
    "then formulate your final answer "
    "based ONLY on the retrieved context. "
    "Do not call the tool multiple times."
)


# =========================
# AGENT
# =========================

agent = create_agent(
    model=model,
    tools=[rag_tool],
    system_prompt=system_prompt
)


# =========================
# ASK QUESTION FUNCTION
# =========================

def ask_question(question: str):

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    return response["messages"][-1].content