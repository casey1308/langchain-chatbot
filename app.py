import streamlit as st
import os
import re
from io import BytesIO
from dotenv import load_dotenv
import PyPDF2
import logging
from datetime import datetime
from openai import OpenAIError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("❌ Please add your OPENAI_API_KEY to the .env file.")
    st.stop()

# Configure page
st.set_page_config(page_title="Augmento - Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Augmento – Chatbot Mode")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""

# Helper functions
def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_pdf_text(file_bytes):
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return clean_text(text)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

# Sidebar file uploader
st.sidebar.header("📄 Upload a Pitch Deck (Optional)")
file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if file:
    pdf_bytes = file.read()
    parsed_text = extract_pdf_text(pdf_bytes)
    if parsed_text:
        st.session_state.pdf_text = parsed_text
        st.success("✅ PDF successfully uploaded and parsed.")
    else:
        st.warning("⚠️ Couldn't extract text from the PDF.")

# Main chat interface
st.header("💬 Chat Interface")

user_input = st.text_input("Enter your message:", key="user_message")

col1, col2 = st.columns([1, 4])
with col1:
    send = st.button("Send", type="primary")

with col2:
    reset = st.button("🗑️ Clear History")

# Handle message
if send and user_input.strip():
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.2)
        context_text = f"{st.session_state.pdf_text}\n\n{user_input}" if st.session_state.pdf_text else user_input
        messages = [
            SystemMessage(content="You are a helpful AI assistant. Respond clearly and concisely."),
            HumanMessage(content=context_text)
        ]
        with st.spinner("Thinking..."):
            response = llm.invoke(messages)
        st.session_state.chat_history.append((user_input, response.content, datetime.now().strftime("%Y-%m-%d %H:%M")))
        st.rerun()
    except OpenAIError as e:
        st.error(f"❌ OpenAI Error: {str(e)}")

# Reset history
if reset:
    st.session_state.chat_history = []
    st.experimental_rerun()

# Show chat history
if st.session_state.chat_history:
    st.subheader("📜 Chat History")
    for i, (q, a, timestamp) in enumerate(reversed(st.session_state.chat_history[-10:])):
        with st.expander(f"{i+1}. {q} ({timestamp})", expanded=(i == 0)):
            st.markdown(f"**🤖 Answer:** {a}")
