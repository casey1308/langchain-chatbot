import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import openai
from io import BytesIO
import re

# LangChain core components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# 🌱 Load keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("❌ Please set both OPENAI_API_KEY and TAVILY_API_KEY in your .env file.")
    st.stop()

# 🧹 Fix Tavily's content formatting
def clean_text(text: str) -> str:
    text = re.sub(r"(\w)\n(\w)", r"\1\2", text)  # Collapse broken words
    text = re.sub(r"\n{2,}", "\n", text)  # Reduce multiple newlines
    return text.strip()

# 🌐 Web Search via Tavily
def run_web_search(query: str) -> str:
    try:
        search = TavilySearchAPIWrapper()
        raw = search.run(query)  # returns string summary
        return clean_text(raw)
    except Exception as e:
        return f"🌐 Web search failed: {str(e)}"

# 📄 RAG Chain
def build_qa_chain(uploaded_file) -> RetrievalQA:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    documents = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
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

    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

# 🧠 Streamlit Interface
st.set_page_config(page_title="Manna - Your AI VC Assistant", page_icon="🤖")
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

# 💬 Main Answer Logic
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        if st.session_state.qa_chain:
            answer = st.session_state.qa_chain.run(user_input)
            if "Insufficient data" in answer or not answer.strip():
                raise ValueError("fallback")
        else:
            raise ValueError("no chain")
    except:
        with st.spinner("🌐 Not enough info in deck. Searching the web..."):
            answer = run_web_search(user_input)

    with st.chat_message("ai"):
        st.markdown(answer)
