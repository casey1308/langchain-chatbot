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

# Load environment variables
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

# Save chat history
def save_history():
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.chat_history, f, indent=2)

# Clean text

def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def clean_assistant_prefix(text):
    return re.sub(r"^Manna:\s*", "", text.strip(), flags=re.IGNORECASE)

def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return clean_text(text)

def split_into_sections(text):
    sections = {}
    current = "General"
    headings = ["summary", "objective", "education", "experience", "projects", "skills", "market", "team", "business model", "financials", "revenue", "traction", "competition", "go-to-market", "ask", "funding"]
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
    revenue_match = re.search(r"(revenue|sales)[^₹$€\d]*(₹|\$|€)?\s?([\d,.]+[MB]?)", doc, re.IGNORECASE)
    if revenue_match:
        metrics["revenue"] = revenue_match.group(0)
    ebitda_match = re.search(r"(EBITDA)[^₹$€\d]*(₹|\$|€)?\s?([\d,.]+[MB]?)", doc, re.IGNORECASE)
    if ebitda_match:
        metrics["ebitda"] = ebitda_match.group(0)
    market_match = re.search(r"(market size|market opportunity)[^₹$€\d]*(₹|\$|€)?\s?([\d,.]+[MB]?)", doc, re.IGNORECASE)
    if market_match:
        metrics["market_size"] = market_match.group(0)
    ask_match = re.search(r"(ask|seeking)[^₹$€\d]*(₹|\$|€)?\s?([\d,.]+[MB]?)", doc, re.IGNORECASE)
    if ask_match:
        metrics["funding_ask"] = ask_match.group(0)
    founder_match = re.search(r"(founder|co-founder|by)[^\n]{0,80}", doc, re.IGNORECASE)
    if founder_match:
        metrics["founder_name"] = founder_match.group(0).strip()
    return metrics

def infer_format(query):
    query = query.lower()
    if "table" in query or "score" in query:
        return "table"
    if "map" in query:
        return "map"
    if "hierarchy" in query or "hypher" in query:
        return "hypher"
    return "summary"

def evaluate_pitch(sections):
    criteria = [
        "Market Opportunity", "Competitive Landscape", "Business Model & Revenue Potential",
        "Traction & Product Validation", "Go-To-Market Strategy",
        "Founding Team & Execution Capability", "Financial Viability & Funding Ask",
        "Revenue Model, Margins, and EBITDA"
    ]
    prompt = "You are a VC analyst. Score the pitch on the following criteria. "
    prompt += "For each, give a score (1-10), a short comment, and improvement tip. Return markdown table: Criterion | Score | Notes | Suggestion.\n\n"
    for crit in criteria:
        matched_text = match_section(crit, sections)
        prompt += f"\n## {crit}\n{matched_text}\n"
    result = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content
    if not result.strip().startswith("| Criterion") and "|" in result:
        result = "```\n" + result.strip() + "\n```"
    return clean_assistant_prefix(result)

def evaluate_resume(sections):
    prompt = "You are a resume reviewer AI. Score the resume on the following sections: Summary, Education, Experience, Projects, Skills, Formatting. "
    prompt += "Give a 0-10 score with notes and improvements in markdown table format: Section | Score | Notes | Suggestion.\n\n"
    for sec, content in sections.items():
        prompt += f"\n## {sec.title()}\n{content}\n"
    result = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content
    if not result.strip().startswith("| Section") and "|" in result:
        result = "```\n" + result.strip() + "\n```"
    return clean_assistant_prefix(result)

def run_web_search(query, format_type="summary"):
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
        result = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content
        return clean_assistant_prefix(result)
    except Exception as e:
        return f"🌐 Web search failed: {str(e)}"

def answer_chat(query, context=""):
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
    result = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key).invoke(prompt).content
    return clean_assistant_prefix(result)

# --- Streamlit UI ---
st.set_page_config(page_title="Manna - AI Deck & Resume Evaluator", page_icon="🤖")
st.markdown("""
    <style>
        body {
            background: #0f0f0f;
            color: white;
        }
        [data-testid="stChatMessage"] div {
            background-color: #2a2a2a;
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        [data-testid="stChatMessage"] span {
            color: white !important;
        }
        .block-container {
            padding: 2rem 2rem 5rem;
        }
    </style>
""", unsafe_allow_html=True)

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

st.divider()

user_input = st.chat_input("💬 Ask Manna anything (e.g. 'What traction is mentioned?' or 'search web: Nvidia news')")

if user_input:
    format_type = infer_format(user_input)
    is_web = user_input.lower().startswith("search web:")
    user_query = user_input[len("search web:"):].strip() if is_web else user_input

    if not st.session_state.file_uploaded:
        answer = answer_chat(user_query)
    elif is_web:
        answer = run_web_search(user_query, format_type)
    else:
        keywords = ["revenue", "ebitda", "market size", "ask", "funding", "founder"]
        if any(k in user_query.lower() for k in keywords):
            metrics = extract_metrics(st.session_state.parsed_doc)
            lines = []
            for k in ["revenue", "ebitda", "market_size", "funding_ask", "founder_name"]:
                value = metrics.get(k, "❌ Not found in document")
                lines.append(f"- **{k.replace('_', ' ').title()}**: {value}")
            answer = "\n".join(lines)
        elif user_query.lower().strip() in ["evaluate this pitch", "re-evaluate pitch", "score pitch"]:
            answer = evaluate_pitch(st.session_state.sections)
        elif user_query.lower().strip() in ["evaluate this resume", "re-evaluate resume", "score resume"]:
            answer = evaluate_resume(st.session_state.sections)
        else:
            answer = answer_chat(user_query, context=st.session_state.parsed_doc)

    answer = clean_assistant_prefix(answer)
    st.session_state.chat_history.append((user_input, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    save_history()
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(answer)

if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🧵 Chat History")
    for user, assistant, ts in reversed(st.session_state.chat_history):
        st.markdown(f"**🕒 {ts}**")
        with st.expander(f"🧍‍♂️ {user}"):
            st.markdown(f"{assistant}")
