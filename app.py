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

# Text cleaner
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

def match_section(query, sections):
    key = query.lower()
    lookup = {
        "founder": ["founder", "co-founder", "team", "ceo", "leadership"],
        "valuation": ["valuation", "valuation cap"],
        "ask": ["ask", "funding", "investment required", "seeking"],
        "round": ["round", "series", "pre-seed", "seed", "series a", "series b"],
        "investor": ["investor", "cap table", "backer", "vc", "existing investor"],
    }
    tags = lookup.get(key, [key])
    for tag in tags:
        for section_key in sections:
            if tag in section_key.lower():
                return sections[section_key]
    return "Not mentioned."

# Web search fallback (SERP API)
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
            return llm.invoke(f"Summarize for a venture capital analyst:\n{combined}").content.strip()
        return f"❌ SERP API error: {r.status_code}"
    except Exception as e:
        return f"❌ Web search failed: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Manna — VC Pitch Evaluator", page_icon="📊")
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

# Chat input
user_query = st.chat_input("💬 Ask anything — e.g., 'Who are the founders?', 'What is their funding ask?', 'Any legal issues?'")

if user_query:
    with st.spinner("🤖 Thinking..."):
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, streaming=True)
        stream_handler = StreamlitCallbackHandler(st.empty())

        context = st.session_state.parsed_doc or ""
        lower_q = user_query.lower()

        # Determine intent
        categories = ["founder", "valuation", "ask", "round", "investor", "legal", "lawsuit", "controversy", "reputation"]
        matched_key = next((cat for cat in categories if cat in lower_q), None)

        if matched_key in ["legal", "lawsuit", "controversy", "reputation"]:
            serp_result = search_serpapi(user_query)
            prompt = (
                f"The pitch deck content is:\n{context[:1500]}\n\n"
                f"Online findings:\n{serp_result}\n\n"
                f"As a VC analyst, answer this:\n{user_query}"
            )
        elif matched_key in ["founder", "valuation", "ask", "round", "investor"]:
            section_text = match_section(matched_key, st.session_state.sections)
            if section_text == "Not mentioned.":
                serp_result = search_serpapi(user_query)
                prompt = (
                    f"The pitch deck doesn't contain this. Here's what I found online:\n{serp_result}\n\n"
                    f"As a VC analyst, answer:\n{user_query}"
                )
            else:
                prompt = (
                    f"The following text is from the startup's pitch deck:\n\n{section_text}\n\n"
                    f"Extract only the relevant answer to this VC-style query: {user_query}\n"
                    f"If information is unclear or missing, respond with: ❌ Info not clearly stated in the deck."
                )
        else:
            prompt = (
                f"Deck Context:\n{context[:3000]}\n\n"
                f"As a venture analyst, answer the following:\n{user_query}"
            )

        with st.container():
            st.markdown(f"**🧑 You:** {user_query}")
            st.markdown("**🤖 Manna:**")
            response = llm.invoke(prompt, callbacks=[stream_handler])
            st.session_state.chat_history.append((user_query, response.content.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# Show history
for user, bot, timestamp in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {user}")
    st.markdown(f"**🤖 Manna:**\n{bot}")
