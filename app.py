import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import csv
import pandas as pd
import logging
from dotenv import load_dotenv
from datetime import datetime
from io import StringIO
from openai import OpenAIError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_community.tools import SerpAPIWrapper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key:
    st.error("❌ Please add your OPENAI_API_KEY to the .env file or Secrets.")
    st.stop()

# Set Streamlit page config
st.set_page_config(page_title="Investment FAQ Chatbot with RAG & Web Search", page_icon="💼", layout="wide")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# Sidebar: Document Upload
st.sidebar.markdown("### 📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF or text files", type=["pdf", "txt"], accept_multiple_files=True)

def load_document(file):
    try:
        if file.type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            os.unlink(tmp_file_path)
        elif file.type == "text/plain":
            content = str(file.read(), "utf-8")
            documents = [Document(page_content=content, metadata={"source": file.name})]
        else:
            st.error(f"Unsupported file type: {file.type}")
            return None
        return documents
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return None

if uploaded_files and st.sidebar.button("🔄 Process Documents"):
    all_docs = []
    for file in uploaded_files:
        docs = load_document(file)
        if docs:
            all_docs.extend(docs)

    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_docs)
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(splits, embeddings)
        st.session_state.vector_store = vector_store
        st.session_state.uploaded_docs = [file.name for file in uploaded_files]
        st.sidebar.success(f"✅ Processed {len(uploaded_files)} document(s)")

# FAQ data
faq_categories = {
    "Fundraising Process": {
        "What documents are needed for fundraising?":
            "You typically need a pitch deck, cap table, financial projections, and details of your funding ask (amount, valuation, instrument).",
        "What is your typical check size?":
            "We typically invest between ₹1.5 Cr to ₹5 Cr depending on the stage and category of the company.",
        "Do you lead rounds or co-invest?":
            "We are flexible. We can lead, co-lead, or follow depending on round dynamics and our conviction.",
    },
    "Evaluation & Due Diligence": {
        "What is the due diligence process like?":
            "Due diligence includes evaluating your legal, financial, and business details. We'll request company registration docs, past financials, founder backgrounds, customer data, etc.",
        "How long does it take to get an investment decision?":
            "It typically takes 3–6 weeks from the first call to decision, depending on how quickly we receive documents and conduct diligence.",
        "Do you invest in pre-revenue startups?":
            "Yes, we do evaluate pre-revenue startups if they are solving a clear problem with a strong founding team and early traction.",
    },
    "Investment Focus": {
        "What sectors do you focus on?":
            "We are sector-agnostic but have a preference for tech-led consumer businesses, B2B SaaS, healthtech, and sustainability.",
    }
}

# Sidebar: Category selection
st.sidebar.title("📚 FAQ Category")
selected_category = st.sidebar.selectbox("Choose a category", list(faq_categories.keys()))
faq_data = faq_categories[selected_category]
faq_questions = list(faq_data.keys())

# Search helpers
def get_best_faq_response(user_input):
    vectorizer = TfidfVectorizer().fit_transform([user_input] + faq_questions)
    sims = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    top_index = sims.argmax()
    top_score = sims[top_index]
    return faq_questions[top_index], faq_data[faq_questions[top_index]], top_score

def get_rag_context(query, k=3):
    if st.session_state.vector_store:
        docs = st.session_state.vector_store.similarity_search(query, k=k)
        return "\n\nAdditional context from uploaded documents:\n" + "\n".join([f"{i+1}. {doc.page_content[:500]}..." for i, doc in enumerate(docs)])
    return ""

def run_web_search(query):
    serp_tool = SerpAPIWrapper()
    return serp_tool.run(query)

# Main UI
st.title("💼 Investment FAQ Chatbot with RAG + Web Search")
st.header("💬 Ask a Question")

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False
input_key = "user_input"
if st.session_state.clear_input:
    st.session_state.clear_input = False
    if input_key in st.session_state:
        del st.session_state[input_key]

user_input = st.text_input("Your question:", key=input_key)
col1, col2 = st.columns([1, 4])
send = col1.button("Send")
reset = col2.button("🗑️ Clear History")

if send and user_input.strip():
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.2)
        best_q, best_a, score = get_best_faq_response(user_input)
        rag_context = get_rag_context(user_input)

        prompt = f"""You are a professional investment FAQ assistant. The user asked: \"{user_input}\"

FAQ Response:
Question: \"{best_q}\"
Answer: {best_a}
{rag_context}

Instructions:
- Use the FAQ answer as your primary response.
- If uploaded documents contain helpful info, incorporate it.
- If nothing is helpful, run a web search.
- Mention if info came from docs or search.

Answer:"""

        with st.spinner("Thinking..."):
            response = llm.invoke([
                SystemMessage(content="Answer concisely and clearly. Incorporate doc context if relevant."),
                HumanMessage(content=prompt)
            ])

        final_response = response.content

        if not rag_context and score < 0.3:
            web_result = run_web_search(user_input)
            final_response += f"\n\n🌐 Web Search Result:\n{web_result}"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.chat_history.append((user_input, final_response, timestamp))

    except OpenAIError as e:
        st.error(f"❌ OpenAI Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")

if reset:
    st.session_state.chat_history = []
    st.rerun()

if st.session_state.chat_history:
    st.subheader("📜 Chat History")
    for i, (q, a, timestamp) in enumerate(reversed(st.session_state.chat_history[-10:])):
        with st.expander(f"{i+1}. {q} ({timestamp})", expanded=(i == 0)):
            st.markdown(f"**🤖 Answer:** {a}")
