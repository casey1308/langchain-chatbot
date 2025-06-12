import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import openai
from io import BytesIO
from streamlit_js_eval import streamlit_js_eval

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in a .env file.")
    st.stop()

# Page config
st.set_page_config(page_title="Manna - Your AI Assistant", page_icon="🤖")
st.title("🤖 Meet Manna - Your AI Chat Assistant")

# RAG Chain
def build_qa_chain(uploaded_file) -> RetrievalQA:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are Manna, a friendly and helpful AI assistant. "
            "Use ONLY the following context to answer the user. "
            "If the answer is not in the context, say you don’t know.\n\n"
            "{context}\n\nQuestion: {question}"
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("📄 Upload a PDF to chat with it", type=["pdf"])

if uploaded_file:
    with st.spinner("Indexing your document..."):
        st.session_state.qa_chain = build_qa_chain(uploaded_file)
    st.success("✅ Document indexed!")

# Assistant intro
with st.chat_message("ai"):
    st.markdown("Hi there! I'm **Manna**, your helpful AI assistant. Ask me anything!")

# Voice input section
st.subheader("🎙️ Speak to Manna (Live Microphone)")
st.markdown("Click to record your voice (5 seconds):")

audio_bytes = streamlit_js_eval(js_expressions="await record_audio()", key="live-audio")
user_input = None

if audio_bytes:
    st.audio(audio_bytes, format="audio/webm")
    with st.spinner("Transcribing your voice..."):
        try:
            audio_file = BytesIO(audio_bytes)
            response = openai.Audio.transcribe("whisper-1", audio_file, api_key=openai_api_key)
            user_input = response["text"]
            st.success(f"🗣️ You said: **{user_input}**")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")

# Chat fallback
user_input = st.chat_input("Or type your question") or user_input

# Process input
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.qa_chain:
        answer = st.session_state.qa_chain.run(user_input)
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Manna, a friendly and helpful AI assistant."),
            ("human", "{input}")
        ])
        chain = prompt | ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        answer = chain.invoke({"input": user_input}).content

    with st.chat_message("ai"):
        st.markdown(answer)
