import streamlit as st
import os
from dotenv import load_dotenv
import re
import PyPDF2
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from streamlit_chat import message

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Validate keys
if not openai_api_key or not tavily_api_key:
    st.error("❌ Set both OPENAI_API_KEY and TAVILY_API_KEY in your .env file.")
    st.stop()

# Set config
st.set_page_config("🤖 Manna: GPT + Analyzer", page_icon="🤖")
st.title("🤖 Manna: Resume + Pitch Deck Analyzer")

# Session setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Utilities ----
def clean_text(text):
    return re.sub(r"(\w)\n(\w)", r"\1\2", text).strip()

def extract_pdf_text(file_bytes):
    pdf = PyPDF2.PdfReader(BytesIO(file_bytes))
    return "\n".join([page.extract_text() or "" for page in pdf.pages])

def infer_format(query):
    q = query.lower()
    if "table" in q or "score" in q or "criteria" in q:
        return "table"
    if "map" in q:
        return "map"
    if "hypher" in q or "hierarchy" in q:
        return "hypher"
    return "summary"

# ---- Analyzer functions ----
def analyze_resume(text):
    prompt = f"""
You are a resume evaluator AI. Based on this text, assign section-wise scores and suggestions.

Resume Content:
{text}

Return in markdown table: Section | Score | Notes | Suggested Improvement
"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    return llm.invoke(prompt).content

def analyze_pitch_deck(text):
    criteria = [
        "Market Opportunity", "Competitive Landscape", "Business Model & Revenue Potential",
        "Traction & Product Validation", "Go-To-Market Strategy",
        "Founding Team & Execution Capability", "Financial Viability & Funding Ask",
        "Revenue Model, Margins, and EBITDA"
    ]
    prompt = f"""
You are a VC analyst AI. Score each of the following areas on a scale of 1–10 based on the pitch deck text.

Pitch Content:
{text}

Return in markdown table: Criterion | Score | Notes | Suggested Improvement
"""
    for crit in criteria:
        prompt += f"\n## {crit}\n"

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    return llm.invoke(prompt).content

def run_web_search(query, format_type):
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        if not results:
            return "🌐 No results found."
        combined = "\n\n".join(
            f"{r['title']}\n{clean_text(r['content'][:700])}" for r in results if r.get("content")
        )
        context = "\n\n".join(
            f"User: {u}\nManna: {a}" for u, a in st.session_state.chat_history[-5:]
        )
        prompt = f"""
You are Manna, a smart GPT-based analyst. Use the info below to respond.

Chat History:
{context}

Web Results:
{combined}

Respond to: {query}
"""
        return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content
    except Exception as e:
        return f"🌐 Web search failed: {e}"

def answer_general_query(query):
    history = "\n\n".join(f"User: {u}\nManna: {a}" for u, a in st.session_state.chat_history[-5:])
    prompt = f"""
You are Manna, a friendly and smart GPT assistant.

Chat History:
{history}

User Question:
{query}
"""
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content

# ---- UI ----
col1, col2 = st.columns(2)
with col1:
    resume_mode = st.checkbox("📄 Analyze as Resume", value=True)
with col2:
    web_search_enabled = st.checkbox("🌐 Enable Web Search (when you say `search web:`)", value=True)

uploaded_file = st.file_uploader("📎 Upload Resume or Pitch Deck (PDF)", type=["pdf"])
file_text = ""
if uploaded_file:
    file_bytes = uploaded_file.read()
    file_text = extract_pdf_text(file_bytes)
    st.success("✅ File uploaded and processed!")

# ---- Chat UI ----
user_input = st.chat_input("💬 Ask anything or analyze your file")

if user_input:
    st.session_state.chat_history.append((user_input, ""))
    response = None
    fmt = infer_format(user_input)

    if user_input.lower().startswith("search web:") and web_search_enabled:
        search_query = user_input[len("search web:"):].strip()
        response = run_web_search(search_query, fmt)

    elif uploaded_file:
        if resume_mode:
            response = analyze_resume(file_text)
        else:
            response = analyze_pitch_deck(file_text)
    else:
        response = answer_general_query(user_input)

    st.session_state.chat_history[-1] = (user_input, response)

# ---- Display chat ----
for i, (q, a) in enumerate(st.session_state.chat_history):
    message(q, is_user=True, key=f"user_{i}")
    if a:
        message(a, is_user=False, key=f"bot_{i}")
