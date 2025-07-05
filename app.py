import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import openai
from io import BytesIO
import requests

# LangChain and OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = "20c2139054e5f0f80d6571e8f09229f5d037bcc1e09bb51673394d5776850291"  # Hardcoded

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in a .env file.")
    st.stop()

# Build RAG Chain
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

    system_template = (
        "You are Manna, a friendly and expert AI assistant VC pitch deck evaluator. "
        "You must use ONLY the structured evaluation data in the context below to generate your answer. "
        "If the necessary data is missing, reply: 'Insufficient data to evaluate.'\n\n"
        "Instructions:\n"
        "1. Carefully read each evaluation criteria, its score (out of 10), and key insight.\n"
        "2. Write a 3-line summary that reflects the strengths and risks of the opportunity.\n"
        "3. Calculate and return the average Fit Score from all the given scores, rounded to one decimal place.\n\n"
        "Do not answer questions outside this evaluation task.\n\n"
        "{context}\n\n"
        "Question: {question}"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=system_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain

# Web search fallback using SerpAPI
def build_web_search_agent():
    search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

    search_tool = Tool(
        name="Web Search",
        func=search.run,
        description="Use this to answer general questions from the internet."
    )

    agent = initialize_agent(
        tools=[search_tool],
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )

    return agent

# UI Setup
st.set_page_config(page_title="Manna - Your AI Assistant", page_icon="🤖")
st.title("🤖 Meet Manna - Your AI Chat Assistant")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "web_agent" not in st.session_state:
    st.session_state.web_agent = build_web_search_agent()

uploaded_file = st.file_uploader("📄 Upload a PDF to chat with it", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Indexing your document…"):
        st.session_state.qa_chain = build_qa_chain(uploaded_file)
    st.success("✅ Document indexed! Ask your questions below.")

with st.chat_message("ai"):
    st.markdown("Hi there! I'm **Manna**, your helpful AI assistant. Ask me anything!")

# Voice input section using audio file upload + Whisper
st.subheader("🎙️ Speak to Manna (Upload Your Voice)")

audio_file = st.file_uploader("Upload a WAV audio file to transcribe and ask", type=["wav"])

if audio_file:
    st.audio(audio_file, format='audio/wav')
    with st.spinner("🔍 Transcribing your voice with Whisper..."):
        try:
            audio_bytes = audio_file.read()
            audio_io = BytesIO(audio_bytes)

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
            st.error(f"❌ Transcription failed: {str(e)}")
            user_input = None
else:
    user_input = st.chat_input("💬 Ask me something…")

# Handle user input from chat or audio
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        if st.session_state.qa_chain is not None:
            answer = st.session_state.qa_chain.run(user_input)
            if "Insufficient data" in answer or answer.strip() == "":
                raise ValueError("Fallback to web")
        else:
            raise ValueError("No PDF context")
    except:
        with st.spinner("📡 Searching the web..."):
            answer = st.session_state.web_agent.run(user_input)

    with st.chat_message("ai"):
        st.markdown(answer)
