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

# Initialize session state
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

# Web search wrapper
def search_web(query):
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=2)
        return "\n".join(f"{r['title']}: {clean_text(r['content'][:400])}" for r in results if r.get("content"))
    except Exception as e:
        return f"Search failed: {e}"

# Final scoring function for Stage 1
def evaluate_pitch_stage1(sections):
    st.subheader("📊 Stage 1: Preliminary Evaluation")

    scoring = [
        ("Problem Statement", 3, 0.08),
        ("Offered Solution", 2, 0.05),
        ("Market Size", 1, 0.03),
        ("Founder Background", 3, 0.08),
        ("Business Model", 3, 0.08),
        ("Stage of the business", 2, 0.05),
        ("Revenue Model", 2, 0.05),
        ("Tech Integration", 3, 0.08),
        ("Traction", 2, 0.05),
        ("Team Dynamics", 3, 0.08),
        ("Team Size", 1, 0.03),
        ("Cap Table", 3, 0.08),
        ("Competitive Landscape", 3, 0.08),
        ("Additional Investment Requirement", 3, 0.08),
        ("Valuation", 2, 0.05),
        ("Regulatory Impact", 1, 0.03),
        ("Exit Opportunity", 1, 0.03)
    ]

    prompt = "You are a VC associate. Based only on the input provided, evaluate the startup using the following format:\n\n"
    prompt += "**Table Format:**\nCriterion | Score (/5) | Reason | Suggestion\n"
    prompt += "**Strict Format Only**. Do not write anything else. Use markdown table format.\n\n"

    for criterion, score, weight in scoring:
        section_text = match_section(criterion, sections)
        web_text = search_web(f"{criterion} site:linkedin.com OR site:crunchbase.com OR site:news.ycombinator.com")
        combined = f"Context from Pitch:\n{section_text}\n\nWeb Context:\n{web_text}\n"
        prompt += f"\n### {criterion}\n{combined}\n"

    model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
    response = model.invoke(prompt).content.strip()

    # Final weighted score
    weighted_score = sum(score * weight for _, score, weight in scoring)
    verdict = (
        "✅ Consider for Investment" if weighted_score >= 3 else
        "⚠️ Second Opinion (2.25 – 3.0)" if weighted_score >= 2.25 else
        "❌ Pass"
    )

    follow_ups = """
**📄 Suggested Follow-up Documents (Stage 2):**
- Financial Model (with HR plan)
- Cap Table + ESOP Details
- YTD MIS
- Startup India Certificate
- Prior Fund Raise Journey
- Founder Reference & Social Checks
- Due Diligence Reports
- Borrowing & Litigation Details
"""

    return f"{response}\n\n**Final Weighted Score**: {round(weighted_score, 2)}\n**Verdict**: {verdict}\n\n{follow_ups}"

# Chat Handler
def answer_chat(query, context=""):
    history = "\n".join([f"User: {u}\nAI: {a}" for u, a, *_ in st.session_state.chat_history[-5:]])
    prompt = f"""You are a VC assistant bot. Use any provided context.

Context:
{context}

Chat History:
{history}

User Question:
{query}

Only respond in markdown.
"""
    return ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key).invoke(prompt).content.strip()

# Streamlit UI
st.set_page_config(page_title="VC Evaluator Manna", page_icon="🧠")

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
    elif user_input.lower().strip() in ["evaluate", "stage 1 evaluation", "score pitch"]:
        with st.spinner("Evaluating pitch..."):
            answer = evaluate_pitch_stage1(st.session_state.sections)
    else:
        answer = answer_chat(user_input, context=st.session_state.parsed_doc)

    st.session_state.chat_history.append((user_input, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# Chat rendering
for user, bot, ts in st.session_state.chat_history:
    st.markdown(f"""
    <div style='background:#111;padding:12px;border-radius:8px;color:white;margin-bottom:6px'>
        <b>You:</b> {user}
    </div>
    <div style='background:#222;padding:12px;border-radius:8px;color:white;margin-bottom:12px'>
        <b>Manna:</b><br>{bot}
    </div>
    """, unsafe_allow_html=True)
