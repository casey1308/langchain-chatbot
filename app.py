import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import openai
import PyPDF2
import re
from io import BytesIO
from datetime import datetime
import json
from difflib import get_close_matches

from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Load env vars
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not openai_api_key or not tavily_api_key:
    st.error("❌ Please set both OPENAI_API_KEY and TAVILY_API_KEY in your .env file.")
    st.stop()

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "parsed_doc" not in st.session_state:
    st.session_state.parsed_doc = None
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "sections" not in st.session_state:
    st.session_state.sections = {}

# Save chat history to file
def save_history():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chat_history, f, indent=2)

# Clean text
def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# PDF text extractor
def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return clean_text(text)

# Section-based chunking
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

# Match closest section
def match_section(key, sections):
    matches = get_close_matches(key.lower(), [k.lower() for k in sections.keys()], n=1, cutoff=0.4)
    if matches:
        for k in sections:
            if k.lower() == matches[0]:
                return sections[k]
    return "Not mentioned"

# Format inference
def infer_format(query: str) -> str:
    query = query.lower()
    if "table" in query or "score" in query:
        return "table"
    if "map" in query:
        return "map"
    if "hierarchy" in query or "hypher" in query:
        return "hypher"
    return "summary"

# Evaluate pitch deck
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
        "For each, give a score (1-10), a short comment, and improvement tip. "
        "Return markdown table: Criterion | Score | Notes | Suggestion.\n\n"
    )
    for crit in criteria:
        matched_text = match_section(crit, sections)
        prompt += f"\n## {crit}\n{matched_text}\n"
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content

# Evaluate resume
def evaluate_resume(sections: dict) -> str:
    prompt = (
        "You are a resume reviewer AI. Score the resume on the following sections: "
        "Summary, Education, Experience, Projects, Skills, Formatting. "
        "Give a 0-10 score with notes and improvements in markdown table format: "
        "Section | Score | Notes | Suggestion.\n\n"
    )
    for sec, content in sections.items():
        prompt += f"\n## {sec.title()}\n{content}\n"
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content

# Web search via Tavily
def run_web_search(query: str, format_type: str = "summary") -> str:
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

# Normal GPT response
def answer_chat(query: str, context: str = "") -> str:
    history = "\n".join([f"User: {u}\nManna: {a}" for u, a, *_ in st.session_state.chat_history[-5:]])
    prompt = f"""
You are Manna, an intelligent assistant. Use the conversation history and any provided context.

Chat History:
{history}

Document Context:
{context}

User Question:
{query}
"""
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content

# --- Streamlit UI ---
st.set_page_config(page_title="Manna - AI Deck & Resume Evaluator", page_icon="🤖")

st.title("🤖 Manna: Resume & Pitch Deck Evaluator")

st.subheader("📄 Upload PDF (Pitch Deck or Resume)")
file = st.file_uploader("Upload a PDF", type=["pdf"])

if file:
    file_bytes = file.read()
    text = extract_pdf_text(file_bytes)
    sections = split_into_sections(text)
    st.session_state.parsed_doc = text
    st.session_state.sections = sections
    st.session_state.file_uploaded = True

    st.success("✅ File uploaded and parsed!")
    if st.checkbox("Analyze as pitch deck?"):
        eval_result = evaluate_pitch(sections)
    else:
        eval_result = evaluate_resume(sections)
    st.markdown(eval_result)
    st.session_state.chat_history.append(("Evaluate this file", eval_result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    save_history()

st.divider()

user_input = st.chat_input("💬 Ask Manna anything (e.g. 'What traction is mentioned?' or 'search web: Nvidia news')")

if user_input:
    format_type = infer_format(user_input)
    is_web = user_input.lower().startswith("search web:")
    user_query = user_input[len("search web:"):].strip() if is_web else user_input

    if user_query.lower().strip() in ["evaluate this pitch", "re-evaluate pitch", "score pitch"]:
        answer = evaluate_pitch(st.session_state.sections)
    elif user_query.lower().strip() in ["evaluate this resume", "re-evaluate resume", "score resume"]:
        answer = evaluate_resume(st.session_state.sections)
    elif is_web:
        answer = run_web_search(user_query, format_type)
    elif st.session_state.file_uploaded and st.session_state.parsed_doc:
        answer = answer_chat(user_query, context=st.session_state.parsed_doc)
    else:
        answer = answer_chat(user_query)

    st.session_state.chat_history.append((user_input, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    save_history()

# --- Chat thread UI with styled chat bubbles ---
st.markdown("## 🧵 Chat History")

st.markdown(
    """
    <style>
.scrollbox {
  max-height: 500px;
  overflow-y: auto;
  border: 1px solid #444;      /* darker border for dark mode */
  padding: 1rem;
  background-color: transparent;  /* or use your dark bg color, e.g. #121212 */
  border-radius: 12px;
  margin-bottom: 1rem;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}


    .message {
      margin-bottom: 1rem;
      max-width: 80%;
      padding: 12px 16px;
      border-radius: 16px;
      line-height: 1.4;
      white-space: pre-wrap;
      box-shadow: 0 1px 3px rgb(0 0 0 / 0.1);
    }
    .user {
      background-color: #2b6cb0;
      color: white;
      margin-left: auto;
      border-bottom-right-radius: 2px;
    }
    .bot {
      background-color: #e2e8f0;
      color: #1a202c;
      margin-right: auto;
      border-bottom-left-radius: 2px;
    }
    .timestamp {
      font-size: 0.7rem;
      color: #718096;
      margin-bottom: 4px;
    }
    .icon {
      display: inline-block;
      vertical-align: middle;
      margin-right: 8px;
      font-size: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="scrollbox">', unsafe_allow_html=True)
for q, a, t in st.session_state.chat_history:
    # user message bubble
    st.markdown(
        f"""
        <div class="timestamp">🕐 {t}</div>
        <div class="message user"><span class="icon">👤</span><b>You:</b> {q}</div>
        <div class="message bot"><span class="icon">🤖</span><b>Manna:</b> {a}</div>
        """,
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)
