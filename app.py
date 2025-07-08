import streamlit as st
import os
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

# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("❌ Please set OPENAI_API_KEY and TAVILY_API_KEY in your .env file.")
    st.stop()

# Session state init
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else (False if key == "file_uploaded" else None)

# --- Utility Functions ---
def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\n{2,}", "\n", text).strip()

def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return clean_text(text)

def split_into_sections(text):
    sections = {}
    current = "General"
    headings = ["problem", "solution", "market", "team", "business", "revenue", "traction", "competition", "valuation", "ask", "funding"]
    for line in text.splitlines():
        l = line.strip()
        if any(h in l.lower() for h in headings):
            current = l
            sections[current] = []
        sections.setdefault(current, []).append(l)
    return {k: "\n".join(v) for k, v in sections.items()}

def match_section(key, sections):
    matches = get_close_matches(key.lower(), [k.lower() for k in sections.keys()], n=1, cutoff=0.4)
    if matches:
        for k in sections:
            if k.lower() == matches[0]:
                return sections[k]
    return "Not mentioned."

def search_web(query):
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=2)
        return "\n".join(f"{r['title']}: {clean_text(r['content'][:400])}" for r in results if r.get("content"))
    except Exception as e:
        return f"Search failed: {e}"

# Stage 1 Evaluation
def evaluate_pitch_stage1(sections):
    st.subheader("📊 Stage 1: Preliminary Evaluation")

    scoring = [
        ("Problem Statement", 0.08),
        ("Offered Solution", 0.05),
        ("Market Size", 0.03),
        ("Founder Background", 0.08),
        ("Business Model", 0.08),
        ("Stage of the business", 0.05),
        ("Revenue Model", 0.05),
        ("Tech Integration", 0.08),
        ("Traction", 0.05),
        ("Team Dynamics", 0.08),
        ("Team Size", 0.03),
        ("Cap Table", 0.08),
        ("Competitive Landscape", 0.08),
        ("Additional Investment Requirement", 0.08),
        ("Valuation", 0.05),
        ("Regulatory Impact", 0.03),
        ("Exit Opportunity", 0.03)
    ]

    prompt = """You are a startup investment analyst. Evaluate the startup based only on the provided input and web context. 
Respond ONLY in this markdown table format:

| Criterion | Score (/5) | Weightage | Remarks | Suggestions |
|-----------|------------|-----------|---------|-------------|
"""

    for criterion, weight in scoring:
        pitch_context = match_section(criterion, sections)
        web_context = search_web(f"{criterion} site:linkedin.com OR site:crunchbase.com")
        prompt += f"\n### {criterion}:\nPitch Context:\n{pitch_context}\nWeb Context:\n{web_context}\n"

    prompt += "\nOnly respond with the markdown table, followed by:\n- Final Weighted Score (out of 5)\n- Verdict (✅ Consider / ⚠️ Second Opinion / ❌ Pass)\n- Bullet list of Follow-up Documents (Stage 2)."

    model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    response = model.invoke(prompt).content.strip()

    return response

# Chat Q&A Handler
def answer_chat(query, context=""):
    history = "\n".join([f"User: {u}\nAI: {a}" for u, a, *_ in st.session_state.chat_history[-5:]])
    prompt = f"""You are a VC analyst chatbot. Use context if available.

Context:
{context}

Chat History:
{history}

User:
{query}

Respond in markdown only.
"""
    return ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key).invoke(prompt).content.strip()

# Streamlit UI
st.set_page_config(page_title="Manna VC Evaluator", page_icon="🧠")
st.title("💼 Manna: Startup Pitch Evaluator")

file = st.file_uploader("📄 Upload pitch deck PDF", type=["pdf"])
if file:
    file_bytes = file.read()
    text = extract_pdf_text(file_bytes)
    st.session_state.parsed_doc = text
    st.session_state.sections = split_into_sections(text)
    st.session_state.file_uploaded = True
    st.success("✅ File parsed successfully!")

user_input = st.chat_input("💬 Ask anything about this startup...")
if user_input:
    if not st.session_state.file_uploaded:
        answer = answer_chat(user_input)
    elif user_input.lower().strip() in ["evaluate", "stage 1", "stage 1 evaluation", "score", "score pitch"]:
        with st.spinner("Evaluating pitch..."):
            answer = evaluate_pitch_stage1(st.session_state.sections)
    else:
        answer = answer_chat(user_input, context=st.session_state.parsed_doc)

    st.session_state.chat_history.append((user_input, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# Display Chat
for user, bot, ts in st.session_state.chat_history:
    st.markdown(f"""
    <div style='background:#111;padding:12px;border-radius:8px;color:white;margin-bottom:6px'>
        <b>You:</b> {user}
    </div>
    <div style='background:#222;padding:12px;border-radius:8px;color:white;margin-bottom:12px'>
        <b>Manna:</b><br>{bot}
    </div>
    """, unsafe_allow_html=True)
