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

# ─── Utilities ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"(\w)\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def infer_format_from_query(query: str) -> str:
    q = query.lower()
    if "hypher" in q or "hierarchy" in q:
        return "hypher"
    if "map" in q or "mapping" in q:
        return "map"
    if "table" in q or "score" in q or "criteria" in q:
        return "table"
    return "summary"

# ─── Web-search-driven LLM summary ────────────────────────────────────────────

def run_web_search(query: str, format_type="summary") -> str:
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        if not results:
            return "🌐 No results found."

        combined = "\n\n".join(
            f"{r['title']}:\n{clean_text(r['content'][:800])}"
            for r in results if r.get("content")
        )

        history = "\n\n".join(
            f"User: {u}\nManna: {a}"
            for u, a in st.session_state.chat_history[-5:]
        )

        system = (
            "You are a VC analyst AI. Use the web results below and prior chat history.\n\n"
            f"History:\n{history}\n\n"
            f"Web Results:\n{combined}\n\n"
            f"Answer in {format_type} format to user’s query: {query}"
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        resp = llm.invoke(system)
        return resp.content

    except Exception as e:
        return f"🌐 Web search failed: {str(e)}"

# ─── Pitch-deck RAG chain ─────────────────────────────────────────────────────

def build_qa_chain(uploaded_file) -> RetrievalQA:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # requires: pip install pypdf, pip install faiss-cpu
    documents = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    system_template = """
You are Manna, a friendly VC analyst AI. Use ONLY the context below to answer.
If data is missing, say 'Insufficient data to evaluate.'.

{context}

Now answer the question below.
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=system_template)

    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

# ─── Resume analyzer ───────────────────────────────────────────────────────────

def analyze_resume(file, format_type="table") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        pdf_path = tmp.name

    pages = PyPDFLoader(pdf_path).load()
    text = "\n\n".join([p.page_content for p in pages])

    prompt = f"""
You are a resume reviewer AI. Analyze the resume below.

- Skills match for Product/VC/Data roles
- Formatting issues
- Suggestions to improve
- Score out of 10 for Readability, ATS Match, Role Fit

Answer in {format_type} format.

Resume Text:
{text}
"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    resp = llm.invoke(prompt)
    return resp.content

# ─── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Manna - Your AI VC Assistant", page_icon="🤖")
st.title("🤖 Meet Manna - Your AI VC Evaluator")

# Session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload & mode
resume_mode = st.checkbox("📄 Analyze as Resume (instead of Pitch Deck)")
uploaded_file = st.file_uploader("Upload a PDF (Resume or Deck)", type=["pdf"])

if uploaded_file:
    with st.spinner("📚 Preparing document…"):
        if resume_mode:
            # just preload nothing
            st.success("✅ Resume ready for analysis")
        else:
            st.session_state.qa_chain = build_qa_chain(uploaded_file)
            st.success("✅ Pitch deck processed!")

# Initial bot greeting
with st.chat_message("ai"):
    st.markdown("Hi! I'm **Manna** — upload a resume or pitch deck, then ask me anything.")

# Voice input
st.subheader("🎙️ Speak to Manna")
audio_file = st.file_uploader("Upload a WAV file", type=["wav"], key="audio")
if audio_file:
    st.audio(audio_file, format="audio/wav")
    with st.spinner("🗣️ Transcribing…"):
        try:
            audio_io = BytesIO(audio_file.read())
            result = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_io,
                api_key=openai_api_key,
                response_format="json",
                language="auto",
                prompt="Translate to English if needed."
            )
            user_input = result["text"]
            st.success(f"✅ Transcribed: **{user_input}**")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
            user_input = None
else:
    user_input = st.chat_input("💬 Ask your question")

# ─── Main logic ────────────────────────────────────────────────────────────────

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    fmt = infer_format_from_query(user_input)

    # decide path
    if resume_mode and uploaded_file:
        answer = analyze_resume(uploaded_file, fmt)

    elif st.session_state.qa_chain:
        # embed format request in the question text
        question = f"{user_input}\n\nPlease answer in {fmt} format."
        answer = st.session_state.qa_chain.run(question)
        if "Insufficient data" in answer:
            answer = f"Insufficient data in deck to answer `{user_input}`."

    else:
        st.warning("⚠️ Please upload a PDF first—either a resume or pitch deck.")
        answer = None

    # display
    if answer:
        with st.chat_message("ai"):
            st.markdown(answer)
        st.session_state.chat_history.append((user_input, answer))
