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

# Load keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Validate keys
if not openai_api_key:
    st.error("❌ Missing OPENAI_API_KEY in .env file.")
    st.stop()

# Init state
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else (False if key == "file_uploaded" else None)

# Utils
def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\n{2,}", "\n", text).strip()

def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))

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
    return "Not mentioned"

# Strict evaluation format with weightage
def evaluate_pitch_table(sections):
    criteria = [
        ("Problem Statement", 0.10),
        ("Offered Solution", 0.15),
        ("Market Size", 0.10),
        ("Founder Background", 0.05),
        ("Business Model", 0.15),
        ("Stage of the Business", 0.10),
        ("Revenue Model", 0.15),
        ("Tech Integration", 0.10),
        ("Traction", 0.10),
    ]

    prompt = "You are a startup evaluator AI.\n"
    prompt += "For each criterion below, evaluate based on the provided text.\n"
    prompt += "Give a markdown table:\n\n"
    prompt += "| Criterion | Score (/5) | Weightage | Remarks | Suggestions |\n"
    prompt += "|-----------|-------------|-----------|---------|-------------|\n"

    for name, weight in criteria:
        section_text = match_section(name, sections)
        prompt += f"\n## {name}\n{section_text}\n"

    prompt += """
Then return the output in this format:

| Criterion | Score (/5) | Weightage | Remarks | Suggestions |
|-----------|-------------|-----------|---------|-------------|
| Example   | 4           | 0.15      | Good effort | Add KPIs    |
| ...       | ...         | ...       | ...        | ...         |

At the end, calculate and show:

Final Weighted Score (out of 5): X.XX  
Verdict: [Choose: "Consider", "Second Opinion", "Pass"]

Suggested Follow-up Documents:
- List bullet points like Financial Model, Cap Table, Due Diligence etc.
"""

    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
        result = llm.invoke(prompt)
        return result.content.strip()
    except Exception as e:
        return f"⚠️ Error during evaluation: {str(e)}"

def run_web_search(query):
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        combined = "\n\n".join(f"{r['title']}:\n{clean_text(r['content'][:800])}" for r in results if r.get("content"))
        return ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key).invoke(
            f"Use the following web results to answer this query:\n\n{combined}\n\nQuery: {query}"
        ).content.strip()
    except Exception as e:
        return f"🌐 Web search failed: {e}"

# UI
st.set_page_config(page_title="Manna: Startup Evaluator", page_icon="📊")
st.title("📊 Manna — VC Pitch Evaluator")

file = st.file_uploader("📄 Upload your pitch deck (PDF)", type=["pdf"])

if file:
    with st.spinner("Parsing document..."):
        file_bytes = file.read()
        doc_text = extract_pdf_text(file_bytes)
        st.session_state.parsed_doc = doc_text
        st.session_state.sections = split_sections(doc_text)
        st.session_state.file_uploaded = True
    st.success("✅ File parsed and ready!")

user_query = st.chat_input("💬 Ask a question or type 'score pitch' to evaluate")

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

# Display chat history
for user, bot, _ in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {user}")
    st.markdown(f"**🤖 Manna:**\n{bot}")
