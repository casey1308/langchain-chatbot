import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import openai
from io import BytesIO
import re

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
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
            f"{r['title']}:\n{clean_text(r['content'][:800])}" for r in results if r.get("content")
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

# Resume analyzer
def analyze_resume(file_bytes: bytes, format_type="table") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        pdf_path = tmp.name
    pages = PyPDFLoader(pdf_path).load()
    text = "\n\n".join([p.page_content for p in pages])
    prompt = f"""
You are a resume expert. The resume text is below.

Break the resume into these sections:
1. Summary/Objective
2. Education
3. Professional Experience
4. Technical Skills
5. Projects
6. Formatting/Layout

For each section:
- List key content.
- Score 0–10 on Completeness, Clarity, Relevance.
- One-sentence improvement advice.

Return as a markdown table: Section | Score | Notes | Improvement.

Resume Text:
{text}
"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    resp = llm.invoke(prompt)
    return resp.content

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
        "Revenue Model, Margins, and EBITDA",
    ]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        pdf_path = tmp.name
    pages = PyPDFLoader(pdf_path).load()
    text = "\n\n".join([p.page_content for p in pages])
    prompt = f"""
You are a VC analyst AI reviewing a startup pitch deck. Analyze it thoroughly based on the following detailed evaluation criteria:
{chr(10).join(f"- {c}" for c in criteria)}

For each criterion:
1. Give a score out of 10.
2. Provide a short analysis highlighting strengths and risks.
3. Suggest an actionable improvement.

Return the output as a markdown table with the columns: Criterion | Score | Notes | Improvement.

Pitch Deck Text:
{text}
"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    resp = llm.invoke(prompt)
    return resp.content

# General GPT chat

def answer_with_openai(query: str) -> str:
    try:
        chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        return chat.invoke(query).content
    except Exception as e:
        return f"❌ GPT error: {str(e)}"

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
user_input = st.text_input("💬 Ask something about your file or just use GPT")

if user_input:
    st.markdown(f"**You:** {user_input}")
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

    if answer:
        st.markdown("**Manna:**")
        st.markdown(answer)
        st.session_state.chat_history.append((user_input, answer))
