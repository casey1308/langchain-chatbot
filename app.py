import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import tempfile
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAIError
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.serpapi.tool import SerpAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key or not serpapi_api_key:
    st.error("❌ Please add your OPENAI_API_KEY and SERPAPI_API_KEY to the .env file.")
    st.stop()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit config
st.set_page_config(page_title="Investment FAQ Chatbot", page_icon="💼", layout="wide")
st.title("💼 Investment FAQ Chatbot with Docs + Web Search")

# Session state setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# Sidebar upload
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
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(all_docs)
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(splits, embeddings)
        st.session_state.vector_store = vector_store
        st.session_state.uploaded_docs = [file.name for file in uploaded_files]
        st.sidebar.success(f"✅ Processed {len(uploaded_files)} document(s)")

# Static FAQ
faq_categories = {
    "Fundraising Process": {
        "What documents are needed for fundraising?": "Pitch deck, cap table, financials, and funding ask details.",
        "What is your typical check size?": "₹1.5 Cr to ₹5 Cr depending on stage and sector.",
        "Do you lead rounds or co-invest?": "We can lead, co-lead, or follow depending on deal dynamics.",
    },
    "Evaluation & Due Diligence": {
        "What is the due diligence process like?": "Legal, financial, and business diligence. Docs like registration, past P&L, founder background, etc.",
        "How long does it take to get an investment decision?": "Typically 3–6 weeks post first call if docs are provided quickly.",
        "Do you invest in pre-revenue startups?": "Yes, if problem and team are strong with early validation.",
    },
    "Investment Focus": {
        "What sectors do you focus on?": "Tech-led consumer, B2B SaaS, healthtech, and sustainability.",
    }
}

# FAQ matching
st.sidebar.title("📚 FAQ Category")
selected_category = st.sidebar.selectbox("Choose a category", list(faq_categories.keys()))
faq_data = faq_categories[selected_category]
faq_questions = list(faq_data.keys())

def get_best_faq_response(user_input):
    vectorizer = TfidfVectorizer().fit_transform([user_input] + faq_questions)
    sims = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    top_index = sims.argmax()
    return faq_questions[top_index], faq_data[faq_questions[top_index]], sims[top_index]

def get_rag_context(query, k=3):
    if st.session_state.vector_store:
        docs = st.session_state.vector_store.similarity_search(query, k=k)
        return "\n\n📄 Document Context:\n" + "\n".join([f"{i+1}. {doc.page_content[:400]}..." for i, doc in enumerate(docs)])
    return ""

def run_web_search(query):
    tool = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    return tool.run(query)

# Chat UI
st.header("💬 Ask a Question")
user_input = st.text_input("Your question:")
send = st.button("Send")
reset = st.button("🗑️ Clear History")

if send and user_input.strip():
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.2)
        best_q, best_a, score = get_best_faq_response(user_input)
        rag_context = get_rag_context(user_input)

        prompt = f"""User asked: \"{user_input}\"

📌 Closest FAQ:
Q: {best_q}
A: {best_a}
{rag_context}

Instructions:
- Answer clearly based on FAQ first
- If documents are useful, include that
- If no match, do a web search
- Say where your answer came from (FAQ, Docs, or Web)
"""

        response = llm.invoke([
            SystemMessage(content="You are a helpful investment FAQ assistant."),
            HumanMessage(content=prompt)
        ])

        final_answer = response.content

        if not rag_context and score < 0.3:
            web_result = run_web_search(user_input)
            final_answer += f"\n\n🌐 Web Search Result:\n{web_result}"

        st.session_state.chat_history.append((user_input, final_answer, datetime.now().strftime("%Y-%m-%d %H:%M")))

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if reset:
    st.session_state.chat_history = []
    st.rerun()

if st.session_state.chat_history:
    st.subheader("📜 Chat History")
    for i, (q, a, timestamp) in enumerate(reversed(st.session_state.chat_history[-10:])):
        with st.expander(f"{i+1}. {q} ({timestamp})", expanded=(i == 0)):
            st.markdown(f"**🤖 Answer:**\n{a}")
