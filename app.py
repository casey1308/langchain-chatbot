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

for key in ["chat_history", "parsed_doc", "file_uploaded", "sections"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else (False if key == "file_uploaded" else None)

def save_history():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chat_history, f, indent=2)

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

def extract_metrics(doc):
    metrics = {}
    patterns = {
        "revenue": r"(revenue|sales)[^₹$€\d]*(₹|\$|€)?\s?([\d,.]+[MB]?)",
        "ebitda": r"(EBITDA)[^₹$€\d]*(₹|\$|€)?\s?([\d,.]+[MB]?)",
        "market_size": r"(market size|market opportunity)[^₹$€\d]*(₹|\$|€)?\s?([\d,.]+[MB]?)",
        "funding_ask": r"(ask|seeking)[^₹$€\d]*(₹|\$|€)?\s?([\d,.]+[MB]?)",
        "founder_name": r"(founder|co-founder|by)[^\n]{0,80}"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, doc, re.IGNORECASE)
        if match:
            metrics[key] = match.group(0).strip()
    return metrics

def infer_format(query):
    q = query.lower()
    if "trend" in q and "score" in q:
        return "trend_table"
    if "table" in q or "score" in q:
        return "table"
    if "map" in q:
        return "map"
    if "hierarchy" in q:
        return "hypher"
    return "summary"

def evaluate_pitch(sections):
    criteria = [
        "Market Opportunity", "Competitive Landscape", "Business Model & Revenue Potential",
        "Traction & Product Validation", "Go-To-Market Strategy",
        "Founding Team & Execution Capability", "Financial Viability & Funding Ask",
        "Revenue Model, Margins, and EBITDA"
    ]
    prompt = "You are a VC analyst. Score the pitch on the following criteria. Return markdown table: Criterion | Score | Notes | Suggestion.\n\n"
    for crit in criteria:
        matched = match_section(crit, sections)
        prompt += f"\n## {crit}\n{matched}\n"
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content.strip()

def evaluate_pitch_stage1():
    parameters = [
        {"name": "Problem Statement", "score": 2, "weight": 0.08},
        {"name": "Offered Solution", "score": 4, "weight": 0.05},
        {"name": "Market Size", "score": 3, "weight": 0.03},
        {"name": "Founder Background", "score": 3, "weight": 0.08},
        {"name": "Business Model", "score": 1.5, "weight": 0.08},
        {"name": "Stage of the Business", "score": 0.5, "weight": 0.05},
        {"name": "Revenue Model", "score": 2, "weight": 0.05},
        {"name": "Tech Integration", "score": 4, "weight": 0.08},
        {"name": "Traction", "score": 0.5, "weight": 0.05},
        {"name": "Team Dynamics", "score": 2, "weight": 0.08},
        {"name": "Team Size", "score": 1, "weight": 0.03},
        {"name": "Cap Table", "score": 2, "weight": 0.08},
        {"name": "Competitive Landscape", "score": 3, "weight": 0.08},
        {"name": "Additional Investment Requirement", "score": 1, "weight": 0.08},
        {"name": "Valuation", "score": 2.5, "weight": 0.05},
        {"name": "Regulatory Impact", "score": 4, "weight": 0.03},
        {"name": "Exit Opportunity", "score": 3, "weight": 0.03}
    ]

    total_score = round(sum(p["score"] * p["weight"] for p in parameters), 2)

    if total_score >= 3.0:
        verdict = "✅ **Consider for Investment**"
        color = "green"
    elif total_score >= 2.25:
        verdict = "⚠️ **Second Opinion Needed**"
        color = "orange"
    else:
        verdict = "❌ **Pass**"
        color = "red"

    report = f"### 📊 VC Scorecard Evaluation (Stage 1/3)\n\n"
    report += "| Parameter | Score (/5) | Weight |\n|---|---|---|\n"
    for p in parameters:
        report += f"| {p['name']} | {p['score']} | {p['weight']} |\n"
    report += f"\n**Total Weighted Score:** `{total_score} / 5.0`"
    report += f"\n\n<span style='color:{color}; font-weight:bold'>{verdict}</span>"

    if total_score < 2.25:
        report += "\n\n📌 Suggested Next Docs:\n- Financial Model\n- YTD MIS\n- Cap Table\n- Due Diligence Report"

    return report

def evaluate_resume(sections):
    prompt = "You are a resume reviewer AI. Score the resume on sections: Summary, Education, Experience, Projects, Skills, Formatting. Return markdown table.\n\n"
    for sec, content in sections.items():
        prompt += f"\n## {sec.title()}\n{content}\n"
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content.strip()

def run_web_search(query, format_type="summary"):
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        combined = "\n\n".join(f"{r['title']}:\n{clean_text(r['content'][:800])}" for r in results if r.get("content"))
        prompt = f"""You are a helpful AI. Use the web results below to answer the user's query in {format_type} format.

Web Results:
{combined}

Query: {query}
"""
        return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content.strip()
    except Exception as e:
        return f"🌐 Web search failed: {str(e)}"

def answer_chat(query, context=""):
    history = "\n".join([f"User: {u}\nAssistant: {a}" for u, a, *_ in st.session_state.chat_history[-5:]])
    prompt = f"""You are Manna. Use context if provided.

Chat History:
{history}

Context:
{context}

Question:
{query}
"""
    return ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="Manna", page_icon="🤖")

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

st.title("🤖 Manna: Resume & Pitch Deck Evaluator")
file = st.file_uploader("📄 Upload your PDF", type=["pdf"])

if file:
    file_bytes = file.read()
    text = extract_pdf_text(file_bytes)
    st.session_state.parsed_doc = text
    st.session_state.sections = split_into_sections(text)
    st.session_state.file_uploaded = True
    st.success("✅ PDF parsed successfully!")

user_input = st.chat_input("💬 Ask me anything...")
if user_input:
    fmt = infer_format(user_input)
    is_web = user_input.lower().startswith("search web:")
    query = user_input[len("search web:"):].strip() if is_web else user_input

    with st.spinner("🤖 Thinking..."):
        if is_web:
            answer = run_web_search(query, fmt)
        elif not st.session_state.file_uploaded:
            answer = answer_chat(query)
        else:
            keywords = ["revenue", "ebitda", "market size", "ask", "funding", "founder"]
            if any(k in query.lower() for k in keywords):
                metrics = extract_metrics(st.session_state.parsed_doc)
                if any(metrics.values()):
                    answer = "\n".join([
                        f"- **{k.replace('_',' ').title()}**: {metrics.get(k, 'Not found')}"
                        for k in ["revenue", "ebitda", "market_size", "funding_ask", "founder_name"]
                    ])
                else:
                    answer = "I couldn't find any financial metrics in the document."
            elif query.lower().strip() in [
                "evaluate this pitch", "score pitch", "trend score", "generate score",
                "run stage 1", "vc scorecard", "preliminary evaluation", "stage 1 score"
            ]:
                answer = evaluate_pitch_stage1()
            elif query.lower().strip() in ["evaluate this resume", "score resume"]:
                answer = evaluate_resume(st.session_state.sections)
            else:
                contextual_answer = answer_chat(query, context=st.session_state.parsed_doc)
                if any(kw in contextual_answer.lower() for kw in ["not mentioned", "no information", "not found", "couldn’t find", "no data"]):
                    answer = answer_chat(query)
                else:
                    answer = contextual_answer

    st.session_state.chat_history.append((user_input, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    save_history()

for user, bot, ts in st.session_state.chat_history:
    st.markdown(f"""
    <div class="chat-container">
        <div class="label">You:</div>
        <div class="chat-bubble user-msg">{user}</div>
        <div class="label" style="text-align:right;">Manna:</div>
        <div class="chat-bubble bot-msg">{bot}</div>
    </div>
    """, unsafe_allow_html=True)
