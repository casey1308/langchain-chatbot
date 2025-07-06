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
    ]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        pdf_path = tmp.name
    pages = PyPDFLoader(pdf_path).load()
    text = "\n\n".join([p.page_content for p in pages])
    prompt = f"""
You are a VC analyst AI reviewing a pitch deck. Evaluate it based on the following criteria:
{chr(10).join(f"- {c}" for c in criteria)}

For each criterion:
1. Provide a score from 0–10.
2. Write a brief note on strengths and weaknesses.
3. Suggest one improvement.

Return as a markdown table with columns: Criterion | Score | Notes | Improvement.

Pitch Deck Text:
{text}
"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    resp = llm.invoke(prompt)
    return resp.content

# Streamlit UI
st.set_page_config(page_title="Manna - AI VC & Resume Evaluator", page_icon="🤖")
st.title("🤖 Manna: Resume & Pitch Deck Scorer")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload mode
esume_mode = st.checkbox("📄 Analyze as Resume (uncheck for Pitch Deck)")
uploaded_file = st.file_uploader("Upload a PDF (Resume or Deck)", type=["pdf"])
# Optional explicit web search trigger
enable_web_search = st.checkbox("🌐 Allow web search on request (unchecked by default)")

file_bytes = None
if uploaded_file:
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    if resume_mode:
        st.success("✅ Resume loaded — ask me to score it!")
    else:
        st.success("✅ Pitch deck loaded — ask me to score it!")

user_input = st.text_input("💬 Enter any question (e.g., 'Score my document')")

if user_input:
    st.markdown(f"**You:** {user_input}")
    fmt = infer_format_from_query(user_input)
    answer = None
    # Determine if user explicitly requests web search
    wants_web = enable_web_search and user_input.lower().startswith("search web:")

    if wants_web:
        # strip trigger
        query = user_input[len("search web:"):].strip()
        answer = run_web_search(query, fmt)
    elif resume_mode and file_bytes:
        answer = analyze_resume(file_bytes, fmt)
    elif (not resume_mode) and file_bytes:
        answer = analyze_deck(file_bytes, fmt)
    else:
        st.warning("⚠️ Please upload a document first.")

    if answer:
        st.markdown(f"**Manna:**")
        st.markdown(answer)
        st.session_state.chat_history.append((user_input, answer))
