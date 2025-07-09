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
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key or not serpapi_key:
    st.error("❌ Add your OPENAI_API_KEY and SERPAPI_API_KEY in .env")
    st.stop()

# Initialize session state
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else (False if key == "file_uploaded" else None)

# Clean PDF text
def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\n{2,}", "\n", text).strip()

def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))

# Sectioning logic
def split_sections(text):
    sections, current = {}, "General"
    headings = [
        "summary", "objective", "education", "experience", "projects", "skills",
        "market", "team", "business model", "financials", "revenue", "traction",
        "competition", "go-to-market", "ask", "funding", "valuation", "round",
        "investors", "series", "cap table", "founder"
    ]
    for line in text.splitlines():
        l = line.strip()
        if any(h in l.lower() for h in headings):
            current = l
            sections[current] = []
        sections.setdefault(current, []).append(l)
    return {k: "\n".join(v) for k, v in sections.items()}

def match_section(key, sections):
    key = key.lower()
    lookup = {
        "founder": ["founder", "co-founder", "team", "ceo", "background"],
        "valuation": ["valuation", "valuation cap"],
        "ask": ["ask", "funding", "investment required"],
        "round": ["round", "series", "pre-seed", "seed", "series a", "series b"],
        "investor": ["investor", "cap table", "backer", "vc"]
    }
    if key in lookup:
        for tag in lookup[key]:
            for section_key in sections:
                if tag in section_key.lower():
                    return sections[section_key]
        return "Not mentioned."
    else:
        matches = get_close_matches(key, [k.lower() for k in sections.keys()], n=1, cutoff=0.4)
        if matches:
            for k in sections:
                if k.lower() == matches[0]:
                    return sections[k]
    return "Not mentioned."

# Google SERP API Search
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
            data = r.json()
            if "organic_results" not in data:
                return "No search results found."

            combined = ""
            for res in data["organic_results"]:
                title = res.get("title", "")
                snippet = res.get("snippet", "")
                combined += f"{title}\n{snippet}\n\n"
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
            return llm.invoke(f"Summarize for a VC:\n{combined}").content.strip()
        return f"❌ SERP API error: {r.status_code}"
    except Exception as e:
        return f"❌ Web search failed: {str(e)}"

# Streamlit app
st.set_page_config(page_title="Manna — VC Evaluator", page_icon="📊")
st.title("📊 Manna — VC Pitch Evaluator")

file = st.file_uploader("📄 Upload a startup pitch deck (PDF)", type=["pdf"])

if file:
    with st.spinner("📄 Parsing pitch deck..."):
        file_bytes = file.read()
        text = extract_pdf_text(file_bytes)
        st.session_state.parsed_doc = text
        st.session_state.sections = split_sections(text)
        st.session_state.file_uploaded = True
    st.success("✅ Pitch deck parsed!")

# Input
user_query = st.chat_input("💬 Ask about the startup, funding, team, legal issues, etc.")

if user_query:
    with st.spinner("🤖 Thinking..."):
        context = st.session_state.parsed_doc or ""
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, streaming=True)
        stream_handler = StreamlitCallbackHandler(st.container())
        lower_q = user_query.lower()

        direct_keys = {
            "founder": ["founder", "ceo"],
            "valuation": ["valuation"],
            "ask": ["ask", "funding required"],
            "round": ["round", "series"],
            "investor": ["investors", "vc", "backers"]
        }

        matched_key = next((k for k, v in direct_keys.items() if any(q in lower_q for q in v)), None)

        if matched_key:
            section_text = match_section(matched_key, st.session_state.sections)
            if section_text == "Not mentioned.":
                serp_result = search_serpapi(user_query)
                prompt = f"Deck didn't include this info. Here's what I found online:\n{serp_result}\n\nAnswer: {user_query}"
            else:
                prompt = f"{section_text}\n\nAnswer this: {user_query}"
        elif any(x in lower_q for x in ["lawsuit", "legal", "controversy", "reputation"]):
            serp_result = search_serpapi(user_query)
            prompt = f"Document Context:\n{context[:1500]}\n\nWeb Results:\n{serp_result}\n\nAnswer this:\n{user_query}"
        else:
            prompt = f"Context:\n{context[:3000]}\n\nQuestion:\n{user_query}"

        with st.container():
            st.markdown(f"**🧑 You:** {user_query}")
            st.markdown("**🤖 Manna:**")
            response = llm.invoke(prompt, callbacks=[stream_handler])
            st.session_state.chat_history.append((user_query, response.content.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# Display full chat
for user, bot, timestamp in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {user}")
    st.markdown(f"**🤖 Manna:**\n{bot}")
