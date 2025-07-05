import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import openai
from io import BytesIO
import re

# LangChain components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("❌ OPENAI_API_KEY or TAVILY_API_KEY missing from .env")
    st.stop()

# 🔍 Web Search with cleaning
def clean_search_text(text: str) -> str:
    # Remove single-character vertical splits like "R\ne\nv\ne\nn\nu\ne"
    cleaned = re.sub(r"(\w)\n(\w)", r"\1\2", text)
    # Remove excessive newlines
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned

def run_web_search(query: str) -> str:
    try:
        search = TavilySearchResults()
        results = search.results(query)
        if not results.get("results"):
            return "🌐 No relevant web results found."
        top = results["results"][:3]
        formatted = []
        for r in top:
            title = r.get("title", "Untitled")
            url = r.get("url", "#")
            content = clean_search_text(r.get("content", ""))[:300].strip()
            formatted.append(f"### 🔗 [{title}]({url})\n{content}...")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"🌐 Web search failed: {str(e)}"

# 📄 Build PDF QA Chain
def build_qa_chain(uploaded_file) -> RetrievalQA:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    system_template = (
        "You are Manna, a friendly and expert AI assistant VC pitch deck evaluator. "
        "You must use ONLY the structured evaluation data in the context below to generate your answer. "
        "If the necessary data is missing, reply: 'Insufficient data to evaluate.'\n\n"
        "Instructions:\n"
        "1. Carefully read each evaluation criteria, its score (out of 10), and key insight.\n"
        "2. Write a 3-line summary that reflects the strengths and risks of the opportunity.\n"
        "3. Calculate and return the average Fit Score from all the given scores, rounded to one decimal place.\n\n"
        "{context}\n\nQuestion: {question}"
    )

    prompt = PromptTemplate(input_variables=["context", "question"], template=system_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

# 🧠 Streamlit UI
st.set_page_config(page_title="Manna - Your AI Assistant", page_icon="🤖")
st.title("🤖 Meet Manna - Your AI VC Assistant")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("📄 Upload a startup pitch deck (PDF)", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Reading and indexing your pitch deck…"):
        st.session_state.qa_chain = build_qa_chain(uploaded_file)
    st.success("✅ Pitch deck processed! Ask Manna anything below.")

with st.chat_message("ai"):
    st.markdown("Hi! I'm **Manna**, your AI VC evaluator. Upload a deck or ask me anything.")

# 🎙️ Voice input
st.subheader("🎙️ Speak to Manna")
audio_file = st.file_uploader("Upload your voice (WAV only)", type=["wav"])
if audio_file:
    st.audio(audio_file, format='audio/wav')
    with st.spinner("Transcribing…"):
        try:
            audio_io = BytesIO(audio_file.read())
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_io,
                api_key=openai_api_key,
                response_format="json",
                language="auto",
                prompt="Translate into English if needed."
            )
            user_input = response["text"]
            st.success(f"🗣️ You said: **{user_input}**")
        except Exception as e:
            st.error(f"Transcription failed: {str(e)}")
            user_input = None
else:
    user_input = st.chat_input("💬 Ask something…")

# 💬 Handle chat input
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        if st.session_state.qa_chain:
            answer = st.session_state.qa_chain.run(user_input)
            if "Insufficient data" in answer or answer.strip() == "":
                raise ValueError("Fallback to web")
        else:
            raise ValueError("No PDF")
    except:
        with st.spinner("🌐 Searching the web..."):
            answer = run_web_search(user_input)

    with st.chat_message("ai"):
        st.markdown(answer)
