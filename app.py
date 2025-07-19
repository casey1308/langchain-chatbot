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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ Please add your OPENAI_API_KEY to the .env file.")
    st.stop()

st.set_page_config(page_title="Augmento - RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Augmento – RAG-Enhanced Pitch Deck Chatbot")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Helper: Text cleaning
def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# PDF Text Extraction
def extract_pdf_text(file_bytes):
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return clean_text(text)
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return ""

# Text Chunking
def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# RAG-style retriever: get top chunk
def get_relevant_chunk(query, chunks):
    vectorizer = TfidfVectorizer().fit_transform([query] + chunks)
    sims = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    top_index = sims.argmax()
    return chunks[top_index]

# Upload PDF
st.sidebar.header("📄 Upload a Pitch Deck (Optional)")
file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
if file:
    pdf_bytes = file.read()
    parsed_text = extract_pdf_text(pdf_bytes)
    if parsed_text:
        st.session_state.pdf_text = parsed_text
        st.session_state.chunks = chunk_text(parsed_text)
        st.success("✅ PDF uploaded and processed with RAG.")
    else:
        st.warning("⚠️ Couldn't extract text from the PDF.")

# Chat
st.header("💬 Chat Interface")
user_input = st.text_input("Enter your message:", key="user_message")

col1, col2 = st.columns([1, 4])
with col1:
    send = st.button("Send", type="primary")
with col2:
    reset = st.button("🗑️ Clear History")

# Message handling
if send and user_input.strip():
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.2)

        if st.session_state.chunks:
            context = get_relevant_chunk(user_input, st.session_state.chunks)
        else:
            context = ""

        prompt = f"""You are a startup pitch analyst AI. Based only on the context below, answer the user query.
Context:
\"\"\"
{context}
\"\"\"

User Question:
{user_input}"""

        with st.spinner("Thinking..."):
            response = llm.invoke([SystemMessage(content="Answer concisely and professionally."), HumanMessage(content=prompt)])
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
