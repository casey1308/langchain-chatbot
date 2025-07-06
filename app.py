import streamlit as st
import os
from dotenv import load_dotenv
import openai
import PyPDF2
import re
from io import BytesIO
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Load environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("❌ Please set both OPENAI_API_KEY and TAVILY_API_KEY in your .env file.")
    st.stop()

# Initial state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores (q, a, timestamp)
if "parsed_doc" not in st.session_state:
    st.session_state.parsed_doc = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

# Clean text
def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# Extract PDF content
def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))

# Split into logical sections
def split_into_sections(text: str) -> dict:
    sections = {}
    current = "General"
    headings = [
        "summary", "objective", "education", "experience", "projects", "skills", "market", "team",
        "business model", "financials", "revenue", "traction", "competition", "go-to-market", "ask", "funding"
    ]
    for line in text.splitlines():
        l = line.strip()
        if any(h in l.lower() for h in headings):
            current = l.strip()
            sections[current] = []
        sections.setdefault(current, []).append(l)
    return {k: "\n".join(v) for k, v in sections.items()}

# Inference format
def infer_format(query: str) -> str:
    query = query.lower()
    if "table" in query or "score" in query:
        return "table"
    if "map" in query:
        return "map"
    if "hierarchy" in query or "hypher" in query:
        return "hypher"
    return "summary"

# Evaluate pitch
def evaluate_pitch(sections: dict) -> str:
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
    prompt = (
        "You are a VC analyst. Score the pitch on the following criteria. "
        "For each, give a score (1-10), notes, and improvements. "
        "Return in markdown table format: Criterion | Score | Notes | Suggestion.\n\n"
    )
    for crit in criteria:
        prompt += f"\n## {crit}\n{sections.get(crit.lower(), 'Not mentioned')}\n"
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content

# Evaluate resume
def evaluate_resume(sections: dict) -> str:
    prompt = (
        "You are a resume reviewer AI. Score the resume on: "
        "Summary, Education, Experience, Projects, Skills, Formatting. "
        "Give 0-10 score, notes, and improvements in markdown table format.\n\n"
    )
    for sec, content in sections.items():
        prompt += f"\n## {sec.title()}\n{content}\n"
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content

# GPT answer
def answer_chat(query: str, context: str = "") -> str:
    history = "\n".join([f"User: {q}\nManna: {a}" for q, a, _ in st.session_state.chat_history[-5:]])
    prompt = f"""
You are Manna, a helpful AI assistant.

Chat History:
{history}

Document Context:
{context}

User Question:
{query}
"""
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content

# Web Search
def run_web_search(query: str, format_type="summary") -> str:
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        if not results:
            return "🌐 No results found."
        combined = "\n\n".join(f"{r['title']}:\n{clean_text(r['content'][:800])}" for r in results if r.get("content"))
        prompt = f"""
You are a helpful AI. Use the web results below to answer the user's query in {format_type} format.

Web Results:
{combined}

Query: {query}
"""
        return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content
    except Exception as e:
        return f"🌐 Web search failed: {str(e)}"

# UI setup
st.set_page_config(page_title="Manna - Resume & Pitch Evaluator", page_icon="🤖")
st.title("🤖 Manna: Resume & Pitch Deck Evaluator")

# Upload
st.subheader("📄 Upload a Resume or Pitch Deck (PDF)")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

# Chat options
resume_mode = st.checkbox("🧾 Treat as Resume (unchecked = Pitch Deck)")
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# On file upload
if uploaded_file:
    file_bytes = uploaded_file.read()
    text = extract_pdf_text(file_bytes)
    sections = split_into_sections(text)
    st.session_state.parsed_doc = text
    st.session_state.sections = sections
    st.session_state.file_uploaded = True
    st.success("✅ File uploaded and parsed!")

    if resume_mode:
        result = evaluate_resume(sections)
    else:
        result = evaluate_pitch(sections)
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append(("Evaluate this file", result, timestamp))
    st.markdown(result)

# Chat input
st.divider()
user_input = st.chat_input("💬 Ask Manna anything (e.g. 'What traction is mentioned?' or 'search web: NVIDIA')")

if user_input:
    timestamp = datetime.now().strftime("%H:%M")
    fmt = infer_format(user_input)
    is_web = user_input.lower().startswith("search web:")
    query = user_input[len("search web:"):].strip() if is_web else user_input

    if is_web:
        reply = run_web_search(query, fmt)
    elif st.session_state.file_uploaded and st.session_state.parsed_doc:
        reply = answer_chat(query, context=st.session_state.parsed_doc)
    else:
        reply = answer_chat(query)

    st.session_state.chat_history.append((user_input, reply, timestamp))

# Chat thread (with timestamps + scrollable)
st.markdown("## 🧵 Chat History")
scroll_style = """
<style>
.scrollbox {
  max-height: 400px;
  overflow-y: auto;
  padding-right: 10px;
}
</style>
"""
st.markdown(scroll_style, unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="scrollbox">', unsafe_allow_html=True)
    for i, (q, a, t) in enumerate(st.session_state.chat_history):
        st.markdown(f"""
        <div style="background:#f9f9f9;padding:10px;border-radius:8px;margin-bottom:10px;">
            <div><b>🕐 {t} | 🧑 You:</b> {q}</div>
            <div style="margin-top:5px;"><b>🤖 Manna:</b> {a}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
