import streamlit as st
import os
import re
import json
import requests
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime
from difflib import get_close_matches
import PyPDF2

from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY is missing in your .env file.")
    st.stop()

# Session state initialization
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else (False if key == "file_uploaded" else None)

# Text cleaner
def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\n{2,}", "\n", text).strip()

# Extract PDF content
def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))

# Split content by rough heading matches
def split_sections(text):
    sections, current = {}, "General"
    headings = [
        "summary", "objective", "education", "experience", "projects", "skills",
        "market", "team", "business model", "financials", "revenue", "traction",
        "competition", "go-to-market", "ask", "funding"
    ]
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

# Strict tabular scoring for Stage 1
def evaluate_pitch_table(sections):
    criteria = [
        ("Problem Statement", "High"),
        ("Offered Solution", "Medium"),
        ("Market Size", "Low"),
        ("Founder Background", "High"),
        ("Business Model", "High"),
        ("Stage of the Business", "Medium"),
        ("Revenue Model", "Medium"),
        ("Tech Integration", "High"),
        ("Traction", "Medium"),
        ("Team Dynamics", "High"),
        ("Team Size", "Low"),
        ("Cap Table", "High"),
        ("Competitive Landscape", "High"),
        ("Additional Investment Requirement", "High"),
        ("Valuation", "Medium"),
        ("Regulatory Impact", "Low"),
        ("Exit Opportunity", "Low"),
    ]

    weight_map = {
        "High": 0.08,
        "Medium": 0.05,
        "Low": 0.03
    }

    prompt = (
        "You are a VC analyst AI.\n"
        "Evaluate the startup based on the following pitch content.\n\n"
        "For each criterion:\n"
        "- Give a score from 1 to 5\n"
        "- Use the correct weight (High=0.08, Medium=0.05, Low=0.03)\n"
        "- Write 1-line remark\n"
        "- Write 1-line suggestion\n\n"
        "Respond only in this markdown format:\n"
        "| Criterion | Score (/5) | Weightage | Weighted Score | Remarks | Suggestions |\n"
        "|-----------|-------------|-----------|----------------|---------|-------------|\n"
    )

    for label, priority in criteria:
        weight = weight_map[priority]
        text = match_section(label, sections)
        prompt += f"\n## {label} (Priority: {priority}, Weightage: {weight})\n{text}\n"

    prompt += (
        "\nThen compute:\n"
        "- Final Weighted Score = sum of (score × weightage)\n"
        "- Verdict:\n"
        "  - ✅ Consider: score ≥ 3.0\n"
        "  - ⚠️ Second Opinion: 2.25 – 2.99\n"
        "  - ❌ Pass: score < 2.25\n\n"
        "List required follow-up documents in bullet points (no explanations).\n"
    )

    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        return f"⚠️ Evaluation failed: {str(e)}"

# Web search function (on any query)
def search_web(query):
    try:
        wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
        return wrapper.run(query)
    except:
        try:
            headers = {"Authorization": f"Bearer {tavily_api_key}"}
            payload = {"query": query, "max_results": 3}
            r = requests.get("https://api.tavily.com/search", headers=headers, params=payload)
            if r.status_code == 200:
                results = r.json().get("results", [])
                if not results:
                    return "🔍 No results found."
                combined = "\n\n".join(f"{r['title']}\n{clean_text(r['content'][:700])}" for r in results if r.get("content"))
                llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
                return llm.invoke(f"Summarize from a VC perspective:\n{combined}").content.strip()
            return f"🌐 Web search failed: {r.status_code} {r.reason}"
        except Exception as e:
            return f"🌐 Web search failed: {e}"

# Streamlit UI
st.set_page_config(page_title="Manna — VC Evaluator", page_icon="📊")
st.title("📊 Manna — VC Pitch Evaluator")

file = st.file_uploader("📄 Upload a startup pitch deck (PDF)", type=["pdf"])

if file:
    with st.spinner("🔍 Parsing pitch deck..."):
        file_bytes = file.read()
        text = extract_pdf_text(file_bytes)
        st.session_state.parsed_doc = text
        st.session_state.sections = split_sections(text)
        st.session_state.file_uploaded = True
    st.success("✅ File parsed successfully!")

user_query = st.chat_input("💬 Ask anything about the startup or search the web")

if user_query:
    with st.spinner("🤖 Thinking..."):
        context = st.session_state.parsed_doc or ""
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)

        if any(x in user_query.lower() for x in ["lawsuit", "legal", "search", "controversy", "reputation"]):
            web_result = search_web(user_query)
            prompt = f"Document Context:\n{context[:1500]}\n\nWeb Results:\n{web_result}\n\nQuestion:\n{user_query}"
        else:
            prompt = f"Context:\n{context[:3000]}\n\nQuestion:\n{user_query}"

        answer = llm.invoke(prompt).content.strip()
        st.session_state.chat_history.append((user_query, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

for user, bot, timestamp in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {user}")
    st.markdown(f"**🤖 Manna:**\n{bot}")
