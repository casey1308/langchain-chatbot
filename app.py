import streamlit as st
import os
import re
import json
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

# Optional web search
def run_web_search(query):
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        context = "\n\n".join(f"{r['title']}:\n{clean_text(r['content'][:800])}" for r in results if r.get("content"))
        return ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key).invoke(
            f"Use the following search results to answer this question:\n\n{context}\n\nQuery: {query}"
        ).content.strip()
    except Exception as e:
        return f"🌐 Web search failed: {e}"

# Streamlit UI
st.set_page_config(page_title="Manna — VC Evaluator", page_icon="📊")
st.title("📊 Manna — VC Pitch Evaluator")

file = st.file_uploader("📄 Upload a startup pitch deck (PDF)", type=["pdf"])

if file:
    with st.spinner("🔍 Parsing and extracting..."):
        file_bytes = file.read()
        text = extract_pdf_text(file_bytes)
        st.session_state.parsed_doc = text
        st.session_state.sections = split_sections(text)
        st.session_state.file_uploaded = True
    st.success("✅ File processed!")

user_query = st.chat_input("💬 Ask a question or type 'score pitch'")

if user_query:
    with st.spinner("🤖 Thinking..."):
        if user_query.lower() in ["score pitch", "evaluate this pitch"]:
            answer = evaluate_pitch_table(st.session_state.sections)
        elif user_query.lower().startswith("search web:"):
            query = user_query.replace("search web:", "").strip()
            answer = run_web_search(query)
        else:
            context = st.session_state.parsed_doc if st.session_state.file_uploaded else ""
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
            answer = llm.invoke(f"Context:\n{context}\n\nQuestion:\n{user_query}").content.strip()
        st.session_state.chat_history.append((user_query, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# Show full chat history
for user, bot, timestamp in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {user}")
    st.markdown(f"**🤖 Manna:**\n{bot}")
