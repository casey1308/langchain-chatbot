import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import openai
from io import BytesIO
import re
import PyPDF2  # ✅ Using PyPDF2 instead of fitz

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from streamlit_chat import message

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("❌ Please set both OPENAI_API_KEY and TAVILY_API_KEY in your .env file.")
    st.stop()

# Clean and format utilities
def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n+", "\n", text)
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

def extract_text_pypdf2(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    return full_text

def sectionize_text(raw_text: str) -> dict:
    headings = [
        "summary", "objective", "education", "experience", "work history", "projects", 
        "skills", "technical skills", "achievements", "team", "business model",
        "market", "revenue", "traction", "product", "financial", "go-to-market",
        "competition", "funding"
    ]
    sections = {}
    current = "other"
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if any(h in line.lower() for h in headings):
            current = line.lower()
            sections[current] = []
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items()}

def analyze_resume(file_bytes: bytes, fmt="table") -> str:
    raw = extract_text_pypdf2(file_bytes)
    sections = sectionize_text(raw)
    prompt = "You are a resume reviewer AI. Provide a markdown table: Section | Score | Notes | Improvement.\n\n"
    for sec, content in sections.items():
        prompt += f"\n## {sec.title()}\n{content}\n"
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    return llm.invoke(prompt).content

def analyze_pitch_deck(file_bytes: bytes, fmt="table") -> str:
    raw = extract_text_pypdf2(file_bytes)
    sections = sectionize_text(raw)
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
    prompt = "You are a VC analyst AI. Score the following sections (1–10) with notes. Return as markdown table: Criterion | Score | Notes | Improvement.\n"
    for crit in criteria:
        prompt += f"\n## {crit}\n{sections.get(crit.lower(), 'Not found')}\n"
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    return llm.invoke(prompt).content

def run_web_search(query: str, fmt="summary") -> str:
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        if not results:
            return "🌐 No results found."
        combined = "\n\n".join(f"{r['title']}\n{clean_text(r['content'][:800])}" for r in results if r.get("content"))
        chat_history = "\n\n".join(f"User: {u}\nManna: {a}" for u, a in st.session_state.chat_history[-5:])
        system_prompt = (
            "You are a VC analyst AI. Use the web results and history to answer.\n\n"
            f"Chat History:\n{chat_history}\n\nWeb Results:\n{combined}\n\n"
            f"Answer in {fmt} format for query: {query}"
        )
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        return llm.invoke(system_prompt).content
    except Exception as e:
        return f"🌐 Web search failed: {str(e)}"

def answer_with_gpt(query: str) -> str:
    history_context = "\n\n".join(f"User: {q}\nManna: {a}" for q, a in st.session_state.chat_history[-5:])
    prompt = f"You are Manna, a helpful assistant.\n\nChat History:\n{history_context}\n\nUser: {query}"
    chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    return chat.invoke(prompt).content

# === Streamlit App UI ===
st.set_page_config(page_title="Manna - AI Resume & Deck Analyzer", page_icon="🤖")
st.title("🤖 Manna - Resume & Pitch Deck Analyzer")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

resume_mode = st.checkbox("📄 Treat as Resume (uncheck for Pitch Deck)")
web_search_enabled = st.checkbox("🌐 Enable Web Search (prefix query with 'search web:')")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
file_bytes = uploaded_file.read() if uploaded_file else None
if uploaded_file:
    st.success("✅ File uploaded!")

user_input = st.chat_input("💬 Ask Manna something...")

if user_input:
    st.session_state.chat_history.append((user_input, ""))
    fmt = infer_format_from_query(user_input)
    answer = None

    if web_search_enabled and user_input.lower().startswith("search web:"):
        query = user_input[len("search web:"):].strip()
        answer = run_web_search(query, fmt)
    elif file_bytes:
        if resume_mode:
            answer = analyze_resume(file_bytes, fmt)
        else:
            answer = analyze_pitch_deck(file_bytes, fmt)
    else:
        answer = answer_with_gpt(user_input)

    st.session_state.chat_history[-1] = (user_input, answer)

# Show chat
for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    if a:
        message(a, is_user=False, key=f"bot_{i}")
