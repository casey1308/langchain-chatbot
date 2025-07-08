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

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("❌ Please set both OPENAI_API_KEY and TAVILY_API_KEY in your .env file.")
    st.stop()

# --- Session State Init ---
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections", "logged_in"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        elif key == "logged_in":
            st.session_state[key] = True  # Set to False if you’re actually implementing login
        else:
            st.session_state[key] = None if key == "sections" else False

# Save chat history
def save_history():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chat_history, f, indent=2)

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
    headings = [
        "summary", "objective", "education", "experience", "projects", "skills",
        "market", "team", "business model", "financials", "revenue", "traction",
        "competition", "go-to-market", "ask", "funding"
    ]
    for line in text.splitlines():
        l = line.strip()
        if any(h in l.lower() for h in headings):
            current = l.strip()
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

# --- Stage 1 Evaluation with Web Insights ---
def evaluate_pitch(sections):
    criteria = [
        "Problem Statement", "Offered Solution", "Market Size", "Founder Background", "Business Model",
        "Stage of the business", "Revenue Model", "Tech Integration", "Traction", "Team Dynamics",
        "Team Size", "Cap Table", "Competitive Landscape", "Additional Investment Requirement",
        "Valuation", "Regulatory Impact", "Exit Opportunity"
    ]

    extracted_sections = {crit: match_section(crit, sections) for crit in criteria}

    try:
        tavily = TavilySearchAPIWrapper()
        mq = f"{extracted_sections['Problem Statement'][:50]} market size India"
        cq = f"{extracted_sections['Problem Statement'][:50]} competitors India"

        market_results = tavily.results(query=mq, max_results=2)
        comp_results = tavily.results(query=cq, max_results=2)

        if market_results:
            extracted_sections["Market Size"] += "\n\n[Web Insight]:\n" + "\n".join(r['content'][:300] for r in market_results if r.get("content"))
        if comp_results:
            extracted_sections["Competitive Landscape"] += "\n\n[Web Insight]:\n" + "\n".join(r['content'][:300] for r in comp_results if r.get("content"))
    except Exception as e:
        extracted_sections["Market Size"] += f"\n\n(Web search failed: {str(e)})"
        extracted_sections["Competitive Landscape"] += f"\n\n(Web search failed: {str(e)})"

    prompt = (
        "You are a venture capital analyst. Based on the pitch content and market info below, "
        "evaluate each criterion fairly using this format:\n\n"
        "**Parameter | Score (/5) | Weight | Remarks | Suggested Docs (if score < 3)**\n\n"
    )

    for crit in criteria:
        prompt += f"\n## {crit}\n{extracted_sections[crit]}\n"

    prompt += """
Please return:
1. The full markdown table in the specified format.
2. ✅ Final Weighted Score out of 5.
3. Verdict:
   - ✅ *Consider for Investment* if score ≥ 3.0
   - ⚠️ *Second Opinion* if score between 2.25 – 3.0
   - ❌ *Pass* if score < 2.25
4. Bullet list of follow-up documents if score ≥ 2.25.
"""

    return ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key).invoke(prompt).content.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Manna VC Evaluator", page_icon="🤖")
st.markdown("<h1 style='margin-bottom:20px;'>🤖 Manna: Startup Pitch Evaluator</h1>", unsafe_allow_html=True)

if not st.session_state.logged_in:
    st.warning("🔒 Please log in to access this tool.")
    st.stop()

file = st.file_uploader("📄 Upload your Pitch Deck PDF", type=["pdf"])

if file:
    file_bytes = file.read()
    text = extract_pdf_text(file_bytes)
    st.session_state.parsed_doc = text
    st.session_state.sections = split_into_sections(text)
    st.session_state.file_uploaded = True
    st.success("✅ PDF parsed successfully!")

user_input = st.chat_input("💬 Ask your evaluation question…")

if user_input:
    if not st.session_state.file_uploaded:
        answer = "📄 Please upload a PDF before asking for evaluation."
    elif any(q in user_input.lower() for q in ["evaluate pitch", "run stage 1", "scorecard", "vc evaluation", "stage 1"]):
        answer = evaluate_pitch(st.session_state.sections)
    else:
        answer = "🤔 I can evaluate pitch decks using Stage 1. Try: 'run stage 1' or 'evaluate pitch'."

    st.session_state.chat_history.append((user_input, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    save_history()

    st.markdown(f"""
    <div class='chat-container'>
        <div class='label'>You:</div>
        <div class='chat-bubble user-msg'>{user_input}</div>
        <div class='label' style='text-align:right;'>Manna:</div>
        <div class='chat-bubble bot-msg'>{answer}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Styling ---
st.markdown("""
<style>
.chat-container { display: flex; flex-direction: column; }
.chat-bubble {
    max-width: 80%; padding: 12px 16px; margin: 6px 0;
    border-radius: 18px; font-size: 15px; line-height: 1.4;
    word-wrap: break-word; box-shadow: 0 0 6px rgba(0,0,0,0.3);
}
.user-msg { align-self: flex-start; background-color: #1f1f1f; color: white; border-top-left-radius: 2px; }
.bot-msg { align-self: flex-end; background-color: #3a3a5a; color: white; border-top-right-radius: 2px; }
.label { font-size: 13px; color: #999; margin-bottom: 3px; }
</style>
""", unsafe_allow_html=True)
