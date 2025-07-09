import streamlit as st
import os
import re
import requests
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime
from difflib import get_close_matches
import PyPDF2

from langchain_openai import ChatOpenAI
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key or not serpapi_key:
    st.error("❌ Please set OPENAI_API_KEY and SERPAPI_API_KEY in your .env file.")
    st.stop()

# Streamlit UI setup
st.set_page_config(page_title="📊 Manna — VC Pitch Evaluator", page_icon="📊")
st.title("📊 Manna — VC Pitch Evaluator")

# Session init
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else (False if key == "file_uploaded" else None)

# PDF text cleaning
def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\n{2,}", "\n", text).strip()

# Extract PDF content
def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))

# Section splitter
def split_sections(text):
    section_dict = {}
    current = "General"
    keywords = [
        "summary", "team", "founder", "co-founder", "leadership", "valuation", "ask",
        "funding", "round", "series", "cap table", "investors", "traction", "market",
        "revenue", "business model", "problem", "solution", "go-to-market", "financials"
    ]
    for line in text.splitlines():
        l = line.strip()
        if any(k in l.lower() for k in keywords):
            current = l
            section_dict[current] = []
        section_dict.setdefault(current, []).append(l)
    return {k: "\n".join(v) for k, v in section_dict.items()}

# Match best section
def match_section(prompt, sections):
    synonyms = {
        "founder": ["founder", "founders", "team", "leadership", "ceo", "management"],
        "ask": ["ask", "funding ask", "investment", "capital requirement"],
        "valuation": ["valuation", "valuation cap", "pre-money", "post-money"],
        "round": ["round", "series", "seed", "pre-seed", "a round", "funding round"],
        "investor": ["investors", "cap table", "vc", "backers", "stakeholders"]
    }
    for key, alias_list in synonyms.items():
        if any(x in prompt.lower() for x in alias_list):
            for section in sections:
                if any(a in section.lower() for a in alias_list):
                    return sections[section]
    # fallback fuzzy
    match = get_close_matches(prompt.lower(), [k.lower() for k in sections], n=1, cutoff=0.3)
    if match:
        for k in sections:
            if match[0] in k.lower():
                return sections[k]
    return "Not mentioned in deck."

# SERP web search
def search_serpapi(query):
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
            "num": 3
        }
        r = requests.get("https://serpapi.com/search", params=params)
        if r.status_code == 200:
            results = r.json().get("organic_results", [])
            if not results:
                return "No search results found."
            combined = ""
            for res in results:
                title = res.get("title", "")
                snippet = res.get("snippet", "")
                combined += f"{title}\n{snippet}\n\n"
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
            return llm.invoke(f"Summarize like a VC analyst:\n{combined}").content.strip()
        return f"❌ SERP API error: {r.status_code}"
    except Exception as e:
        return f"❌ SERP error: {str(e)}"

# Upload and parse deck
file = st.file_uploader("📄 Upload a startup pitch deck (PDF)", type=["pdf"])
if file:
    with st.spinner("📄 Parsing pitch deck..."):
        file_bytes = file.read()
        text = extract_pdf_text(file_bytes)
        st.session_state.parsed_doc = text
        st.session_state.sections = split_sections(text)
        st.session_state.file_uploaded = True
    st.success("✅ Pitch deck parsed!")

# Chat input
user_query = st.chat_input("💬 Ask anything like 'founder', 'valuation', 'funding round', 'cap table', etc.")

if user_query:
    with st.spinner("🤖 Thinking..."):
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, streaming=True)
        stream_handler = StreamlitCallbackHandler(st.empty())

        section = match_section(user_query, st.session_state.sections)
        if section == "Not mentioned in deck.":
            web_summary = search_serpapi(user_query)
            prompt = f"You are a VC analyst. Document lacks info. Here's web search:\n{web_summary}\n\nAnswer user query: {user_query}"
        else:
            prompt = f"You are a VC analyst. Based on the pitch deck section:\n{section}\n\nAnswer: {user_query}"

        st.markdown(f"**🧑 You:** {user_query}")
        st.markdown("**🤖 Manna:**")
        response = llm.invoke(prompt, callbacks=[stream_handler])
        st.session_state.chat_history.append((user_query, response.content.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# Display chat history
for user, bot, timestamp in st.session_state.chat_history:
    st.markdown(f"**🧑 You ({timestamp}):** {user}")
    st.markdown(f"**🤖 Manna:**\n{bot}")
