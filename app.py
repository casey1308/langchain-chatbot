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
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Load .env variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("❌ Please set both OPENAI_API_KEY and TAVILY_API_KEY in your .env file.")
    st.stop()

# Clean text
def clean_text(text: str) -> str:
    text = re.sub(r"(\w)\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# Format inference
def infer_format_from_query(query: str) -> str:
    q = query.lower()
    if "hypher" in q or "hierarchy" in q:
        return "hypher"
    elif "map" in q or "mapping" in q:
        return "map"
    elif "table" in q or "score" in q or "criteria" in q:
        return "table"
    return "summary"

# Web search
def run_web_search(query: str, format_type="summary") -> str:
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        if not results:
            return "🌐 No results found."

        combined_content = "\n\n".join(
            f"{r['title']}:\n{clean_text(r['content'][:1000])}" for r in results if r.get("content")
        )

        chat_context = "\n\n".join(
            f"User: {q}\nManna: {a}" for q, a in st.session_state.chat_history[-5:]
        )

        system_prompt = (
            "You are a VC analyst AI. Use the following web search results and prior conversation to answer in a structured format.\n\n"
            f"Previous Conversation:\n{chat_context}\n\n"
            f"Web Results:\n{combined_content}\n\n"
            f"Format: {format_type}\n\n"
            f"Current User Query: {query}"
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        response = llm.invoke(system_prompt)
        return response.content

    except Exception as e:
        return f"🌐 Web search failed: {str(e)}"

# Pitch deck QA chain
def build_qa_chain(uploaded_file) -> RetrievalQA:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    documents = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    system_template = (
        "You are Manna, a VC analyst AI. Use ONLY the data below. "
        "Respond in {format} style. If insufficient data, say so.\n\n"
        "{context}\n\nQuestion: {question}"
    )
    prompt = PromptTemplate(input_variables=["context", "question", "format"], template=system_template)

    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

# Resume analyzer
def analyze_resume(file, format_type="table") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = "\n\n".join([p.page_content for p in pages])

    prompt = (
        "You are a resume reviewer AI. Analyze the resume text below.\n"
        "Provide a structured evaluation of:\n"
        "- Skills match for Product/VC/Data roles\n"
        "- Formatting issues\n"
        "- Suggestions to improve\n"
        "- Score out of 10 for Readability, ATS Match, and Role Fit\n\n"
        f"Return in this format: {format_type}\n\n"
        f"Resume Text:\n{text}"
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    response = llm.invoke(prompt)
    return response.content

# Streamlit UI
st.set_page_config(page_title="Manna - Your AI VC Assistant", page_icon="🤖")
st.title("🤖 Meet Manna - Your AI VC Evaluator")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

resume_mode = st.checkbox("📄 Analyze as Resume (instead of Pitch Deck)")

uploaded_file = st.file_uploader("Upload a PDF file (Resume or Deck)", type=["pdf"])

if uploaded_file:
    with st.spinner("📚 Reading and analyzing..."):
        if resume_mode:
            answer = analyze_resume(uploaded_file)
            st.session_state.chat_history.append(("Uploaded Resume", answer))
            st.session_state.qa_chain = None
        else:
            st.session_state.qa_chain = build_qa_chain(uploaded_file)
            st.success("✅ Pitch deck processed!")

# Greeting
with st.chat_message("ai"):
    st.markdown("Hi! I'm **Manna**, your AI VC evaluator. Upload a resume or pitch deck and ask questions.")

# Voice input
st.subheader("🎙️ Speak to Manna")
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])
if audio_file:
    st.audio(audio_file, format="audio/wav")
    with st.spinner("🗣️ Transcribing..."):
        try:
            audio_io = BytesIO(audio_file.read())
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_io,
                api_key=openai_api_key,
                response_format="json",
                language="auto",
                prompt="Translate to English if needed."
            )
            user_input = response["text"]
            st.success(f"✅ Transcribed: **{user_input}**")
        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")
            user_input = None
else:
    user_input = st.chat_input("💬 Ask your question")

# Main logic
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    format_type = infer_format_from_query(user_input)

    try:
        if resume_mode:
            answer = analyze_resume(uploaded_file, format_type)
        elif st.session_state.qa_chain:
            answer = st.session_state.qa_chain.run({
                "question": user_input,
                "format": format_type
            })
            if "Insufficient data" in answer:
                raise ValueError("Fallback")
        else:
            raise ValueError("No document uploaded")

    except:
        with st.spinner("🌐 Not enough info. Searching the web..."):
            answer = run_web_search(user_input, format_type)

    with st.chat_message("ai"):
        st.markdown(answer)

    st.session_state.chat_history.append((user_input, answer))
