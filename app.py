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
    st.error("❌ Missing API keys! Please add OPENAI_API_KEY and TAVILY_API_KEY to your .env file.")
    st.stop()

# Initialize session state
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else (False if key == "file_uploaded" else None)

# Utility functions
def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\n{2,}", "\n", text).strip()

def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))

def split_into_sections(text):
    sections = {}
    current = "General"
    headings = ["summary", "problem", "solution", "team", "market", "product", "traction", "revenue", "financial", "competition", "ask", "cap table", "valuation", "exit"]
    for line in text.splitlines():
        if any(h in line.lower() for h in headings):
            current = line.strip()
            sections[current] = []
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items()}

def match_section(key, sections):
    matches = get_close_matches(key.lower(), [k.lower() for k in sections.keys()], n=1, cutoff=0.4)
    if matches:
        for k in sections:
            if k.lower() == matches[0]:
                return sections[k]
    return "Not mentioned"

def run_web_search(query):
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        combined = "\n\n".join(f"{r['title']}:\n{clean_text(r['content'][:800])}" for r in results if r.get("content"))
        return combined
    except Exception as e:
        return f"Search failed: {e}"

def evaluate_pitch_stage1(sections):
    scoring_table = [
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
        ("Cap Table", 0.10),
        ("Competitive Landscape", 0.08),
        ("Additional Investment Requirement", 0.08),
        ("Valuation", 0.05),
        ("Regulatory Impact", 0.03),
        ("Exit Opportunity", 0.03),
    ]

    prompt = """You are a VC analyst. Score the startup on the following parameters from 1-5 and return ONLY a markdown table in the format:

| Parameter | Score (/5) | Web Insights | Notes |
|-----------|------------|--------------|-------|
"""

    for label, weight in scoring_table:
        section = match_section(label, sections)
        web_data = run_web_search(f"{label} risks or insights for this startup")
        prompt += f"\n## {label}\nPitch Deck:\n{section}\n\nWeb Insights:\n{web_data}\n"

    prompt += "\nReturn scores out of 5 only. Be strict but fair. Add 1-2 line remark for each.\n"

    ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

    return result

# Main UI
st.set_page_config(page_title="Manna: VC Pitch Evaluator", page_icon="📊")
st.title("📊 Manna – VC Pitch Evaluator")

st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
}
.chat-bubble {
    max-width: 80%;
    padding: 12px 16px;
    margin: 6px 0;
    border-radius: 18px;
    font-size: 15px;
    line-height: 1.4;
    word-wrap: break-word;
    box-shadow: 0 0 6px rgba(0,0,0,0.3);
}
.user-msg {
    align-self: flex-start;
    background-color: #1f1f1f;
    color: white;
    border-top-left-radius: 2px;
}
.bot-msg {
    align-self: flex-end;
    background-color: #3a3a5a;
    color: white;
    border-top-right-radius: 2px;
}
.label {
    font-size: 13px;
    color: #999;
    margin-bottom: 3px;
}
</style>
""", unsafe_allow_html=True)

file = st.file_uploader("📄 Upload Pitch Deck (PDF)", type=["pdf"])

if file:
    file_bytes = file.read()
    text = extract_pdf_text(file_bytes)
    st.session_state.parsed_doc = text
    st.session_state.sections = split_into_sections(text)
    st.session_state.file_uploaded = True
    st.success("✅ Pitch deck uploaded and parsed.")

user_input = st.chat_input("💬 Ask Manna your query...")

def save_history():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chat_history, f, indent=2)

if user_input:
    with st.spinner("🧠 Thinking..."):
        if user_input.lower() in ["evaluate this pitch", "score pitch"]:
            answer = evaluate_pitch_stage1(st.session_state.sections)
        else:
            prompt = f"Context:\n{st.session_state.parsed_doc}\n\nUser Query:\n{user_input}"
            answer = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key).invoke(prompt).content.strip()

        st.session_state.chat_history.append((user_input, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        save_history()

# Chat history display
for user, bot, ts in st.session_state.chat_history:
    st.markdown(f"""
    <div class="chat-container">
        <div class="label">You:</div>
        <div class="chat-bubble user-msg">{user}</div>
        <div class="label" style="text-align:right;">Manna:</div>
        <div class="chat-bubble bot-msg">{bot}</div>
    </div>
    """, unsafe_allow_html=True)
