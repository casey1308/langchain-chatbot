import streamlit as st
import os
import re
import requests
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime
from difflib import get_close_matches
import PyPDF2
import logging
from openai import OpenAIError, RateLimitError, APIError

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# PDF extraction
def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))

# Section splitting
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

# Match input intent to deck section
def match_section(key, sections):
    key = key.lower()
    lookup = {
        "founder": ["founder", "co-founder", "team", "ceo", "leadership"],
        "valuation": ["valuation", "valuation cap"],
        "ask": ["ask", "funding required", "investment", "capital"],
        "round": ["round", "series", "pre-seed", "seed", "series a", "series b"],
        "investor": ["investor", "cap table", "backer", "vc", "existing investor"],
        "team": ["team", "leadership", "core team"]
    }
    if key in lookup:
        for tag in lookup[key]:
            for section_key in sections:
                if tag in section_key.lower():
                    return sections[section_key]
        return "Not mentioned in deck."
    else:
        matches = get_close_matches(key, [k.lower() for k in sections.keys()], n=1, cutoff=0.4)
        if matches:
            for k in sections:
                if k.lower() == matches[0]:
                    return sections[k]
    return "Not mentioned in deck."

# Google SERP API fallback
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

# UI layout
st.set_page_config(page_title="Manna — VC Pitch Evaluator", page_icon="📊")
st.title("📊 Manna — VC Pitch Evaluator")

# Upload
file = st.file_uploader("📄 Upload a startup pitch deck (PDF)", type=["pdf"])

if file:
    with st.spinner("📄 Parsing pitch deck..."):
        file_bytes = file.read()
        text = extract_pdf_text(file_bytes)
        st.session_state.parsed_doc = text
        st.session_state.sections = split_sections(text)
        st.session_state.file_uploaded = True
    st.success("✅ Pitch deck parsed!")

# Prompt input
user_query = st.chat_input("💬 Ask about founders, funding, valuation, team, etc.")

if user_query:
    with st.spinner("🤖 Thinking..."):
        try:
            context = st.session_state.parsed_doc or ""
            llm = ChatOpenAI(
                model="gpt-4o", 
                openai_api_key=openai_api_key, 
                streaming=True,
                temperature=0.7,
                max_tokens=1500
            )
            
            lower_q = user_query.lower()

            # Intents for structured section queries
            intent_keys = {
                "founder": ["founder", "who is the founder", "co-founder"],
                "team": ["team", "leadership"],
                "valuation": ["valuation", "company worth"],
                "ask": ["ask", "funding ask", "how much funding"],
                "round": ["previous round", "funding round", "series a", "seed"],
                "investor": ["investors", "vc", "backers", "existing investors"]
            }

            matched_key = next((k for k, v in intent_keys.items() if any(q in lower_q for q in v)), None)

            if matched_key:
                section_text = match_section(matched_key, st.session_state.sections)
                if section_text == "Not mentioned in deck.":
                    web_result = search_serpapi(user_query)
                    context_msg = f"Deck didn't include this info. Here's what I found online:\n{web_result}"
                else:
                    context_msg = f"Deck Section:\n{section_text}"
            elif any(x in lower_q for x in ["lawsuit", "legal", "controversy", "reputation"]):
                web_result = search_serpapi(user_query)
                context_msg = f"Deck + Web:\n{context[:1500]}\n\nWeb:\n{web_result}"
            else:
                context_msg = f"Deck:\n{context[:3000]}"

            # Use streaming messages
            messages = [
                SystemMessage(content="You are a VC analyst AI. Give crisp, factual answers from pitch deck or web."),
                HumanMessage(content=f"{context_msg}\n\nQuestion: {user_query}")
            ]

            # Debug: Check message format
            logger.info(f"Messages format: {[type(msg).__name__ for msg in messages]}")

            # Chat output
            st.markdown(f"**🧑 You:** {user_query}")
            st.markdown("**🤖 Manna:**")

            def generate_response():
                for chunk in llm.stream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content

            full_output = st.write_stream(generate_response())

        except RateLimitError:
            st.error("❌ Rate limit exceeded. Please try again in a minute.")
            full_output = "Rate limit exceeded."
        except APIError as e:
            st.error(f"❌ OpenAI API error: {str(e)}")
            full_output = f"API error: {str(e)}"
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Streaming error: {str(e)}", exc_info=True)
            
            # Fallback to non-streaming
            try:
                response = llm.invoke(messages)
                full_output = response.content
                st.markdown(full_output)
            except Exception as fallback_e:
                st.error(f"❌ Fallback also failed: {str(fallback_e)}")
                full_output = "Error generating response."

        st.session_state.chat_history.append(
            (user_query, full_output.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

# Chat history
for user, bot, timestamp in st.session_state.chat_history:
    st.markdown(f"**🧑 You:** {user}")
    st.markdown(f"**🤖 Manna:**\n{bot}")
