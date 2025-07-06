import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import openai
from io import BytesIO
import re
import fitz  # PyMuPDF for accurate section-based parsing

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Load .env variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not openai_api_key or not tavily_api_key:
    st.error("❌ Please set both OPENAI_API_KEY and TAVILY_API_KEY in your .env file.")
    st.stop()

# Utilities
def clean_text(text: str) -> str:
    text = re.sub(r"(\w)\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def infer_format_from_query(query: str) -> str:
    q = query.lower()
    if "hypher" in q or "hierarchy" in q:
        return "hypher"
    if "map" in q or "mapping" in q:
        return "map"
    if "table" in q or "score" in q or "criteria" in q:
        return "table"
    return "summary"

# Web search fallback
def run_web_search(query: str, format_type="summary") -> str:
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        if not results:
            return "🌐 No results found."

        combined = "\n\n".join(
            f"{r['title']}\n{clean_text(r['content'][:800])}" for r in results if r.get("content")
        )
        history = "\n\n".join(
            f"User: {u}\nManna: {a}" for u, a in st.session_state.chat_history[-5:]
        )
        system = (
            "You are a VC analyst AI. Use the web results and prior chat history to answer.\n\n"
            f"History:\n{history}\n\n"
            f"Web Results:\n{combined}\n\n"
            f"Answer in {format_type} format for query: {query}"
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        resp = llm.invoke(system)
        return resp.content
    except Exception as e:
        return f"🌐 Web search failed: {str(e)}"

# Section-based text extractor
def extract_sections_from_pdf(file_bytes: bytes) -> dict:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    sections = {}
    headings = [
        "summary", "objective", "education", "experience", "work history", "projects", 
        "skills", "technical skills", "achievements", "team", "business model",
        "market", "revenue", "traction", "product", "financial", "go-to-market",
        "competition", "funding"
    ]
    current = "other"
    for line in text.splitlines():
        l = line.strip()
        if not l:
            continue
        if any(h in l.lower() for h in headings):
            current = l.strip().lower()
            sections[current] = []
        sections.setdefault(current, []).append(l)
    return {k: "\n".join(v) for k, v in sections.items()}

# Resume analyzer
def analyze_resume(file_bytes: bytes, format_type="table") -> str:
    sections = extract_sections_from_pdf(file_bytes)
    prompt = """
You are a resume reviewer AI. Analyze this resume with sections below.
For each: give a 0–10 score, notes, and 1-line improvement.
Format as markdown table: Section | Score | Notes | Improvement.

"""
    for sec, content in sections.items():
        prompt += f"\n## {sec.title()}\n{content}\n"
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    return llm.invoke(prompt).content

# Pitch deck analyzer
def analyze_deck(file_bytes: bytes, format_type="table") -> str:
    criteria = [
        "Market Opportunity",
        "Competitive Landscape",
        "Business Model & Revenue Potential",
        "Traction & Product Validation",
        "Go-To-Market Strategy",
        "Founding Team & Execution Capability",
        "Financial Viability & Funding Ask",
        "Revenue Model, Margins, and EBITDA"
    ]
    sections = extract_sections_from_pdf(file_bytes)
    prompt = """
You are a VC analyst AI. Analyze the following startup pitch sections:
Each section should be scored on a scale of 1–10 with concise feedback.
Return a markdown table: Criterion | Score | Notes | Improvement.

"""
    for crit in criteria:
        prompt += f"\n## {crit}\n{sections.get(crit.lower(), 'Not Found')}\n"
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    return llm.invoke(prompt).content

# General GPT chat with memory
from streamlit_chat import message

def answer_with_openai(query: str) -> str:
    history_context = "\n\n".join(
        f"User: {q}\nManna: {a}" for q, a in st.session_state.chat_history[-5:]
    )
    prompt = f"""
You are Manna, a helpful AI assistant. Use the chat history and current user question to answer thoughtfully.

Chat History:
{history_context}

Current Question:
{query}
"""
    chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    return chat.invoke(prompt).content

# Streamlit UI
st.set_page_config(page_title="Manna - AI VC & Resume Evaluator", page_icon="🤖")
st.title("🤖 Manna: Resume & Pitch Deck Scorer")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI widgets
resume_mode = st.checkbox("📄 Analyze as Resume (uncheck for Pitch Deck)")
uploaded_file = st.file_uploader("Upload a PDF (Resume or Deck)", type=["pdf"])
web_search_enabled = st.checkbox("🌐 Allow web search on request (prefix with 'search web:')")

file_bytes = None
if uploaded_file:
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    st.success("✅ File uploaded successfully")

# Chat input
user_input = st.chat_input("💬 Ask something about your file or just use GPT")

if user_input:
    st.session_state.chat_history.append((user_input, ""))
    fmt = infer_format_from_query(user_input)
    answer = None

    wants_web = web_search_enabled and user_input.lower().startswith("search web:")

    if wants_web:
        query = user_input[len("search web:"):].strip()
        answer = run_web_search(query, fmt)
    elif file_bytes:
        if resume_mode:
            answer = analyze_resume(file_bytes, fmt)
        else:
            answer = analyze_deck(file_bytes, fmt)
    else:
        answer = answer_with_openai(user_input)

    st.session_state.chat_history[-1] = (user_input, answer)

# Display chat
for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    if a:
        message(a, is_user=False, key=f"bot_{i}")
