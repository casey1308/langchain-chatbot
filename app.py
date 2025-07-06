import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import openai
from io import BytesIO
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not openai_api_key or not tavily_api_key:
    st.error("❌ Please set both OPENAI_API_KEY and TAVILY_API_KEY in your .env file.")
    st.stop()

# Clean Tavily search result text
def clean_text(text: str) -> str:
    text = re.sub(r"(\w)\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# Tavily Search
def run_web_search(query: str, format_type="summary") -> str:
    try:
        search = TavilySearchAPIWrapper()
        results = search.results(query=query, max_results=3)
        if not results:
            return "🌐 No results found."

        # Extract the top contents
        combined_content = "\n\n".join(
            f"{r['title']}:\n{clean_text(r['content'][:1000])}" for r in results if r.get("content")
        )

        system_prompt = (
            "You are a VC research analyst. Based on the following web content, answer the user's query in a clear, structured format.\n\n"
            "If the user asked for scoring, respond with a VC scorecard table.\n"
            "If the user asked for hypher mapping, return bullet-style structured evaluation.\n"
            "If the user asked for mapping, compare major dimensions in text.\n"
            "If not specified, provide a strong summary in ~5 sentences.\n\n"
            f"Format: {format_type}\n\n"
            f"Web Results:\n{combined_content}\n\n"
            f"Query: {query}"
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        response = llm.invoke(system_prompt)
        return response.content

    except Exception as e:
        return f"🌐 Web search failed: {str(e)}"

# PDF Vector DB + RAG
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
        "You are Manna, a friendly and expert VC analyst AI. "
        "Use ONLY the structured evaluation data below. "
        "If the data is missing, respond with 'Insufficient data to evaluate.'\n\n"
        "Instructions:\n"
        "1. Score and evaluate the company using standard VC metrics (team, market, traction, etc).\n"
        "2. Return output in {format} style.\n"
        "3. End with average score if available.\n\n"
        "{context}\n\nQuestion: {question}"
    )
    prompt = PromptTemplate(input_variables=["context", "question", "format"], template=system_template)

    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

# Streamlit App UI
st.set_page_config(page_title="Manna - Your AI VC Assistant", page_icon="🤖")
st.title("🤖 Meet Manna - Your AI VC Evaluator")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("📄 Upload a startup pitch deck (PDF)", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("🔍 Reading and indexing your pitch deck…"):
        st.session_state.qa_chain = build_qa_chain(uploaded_file)
    st.success("✅ Pitch deck processed!")

with st.chat_message("ai"):
    st.markdown("Hi! I'm **Manna**, your VC assistant. Upload a deck or ask anything — even scorecards!")

# Voice Upload
st.subheader("🎙️ Speak to Manna")
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])
if audio_file:
    st.audio(audio_file, format='audio/wav')
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

# Output Format Inference
def infer_format_from_query(query: str) -> str:
    q = query.lower()
    if "hypher" in q or "hierarchy" in q:
        return "hypher"
    elif "map" in q or "mapping" in q:
        return "map"
    elif "table" in q or "score" in q or "parameter" in q or "criteria" in q:
        return "table"
    return "summary"

# Main QA Logic
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    format_type = infer_format_from_query(user_input)

    try:
        if st.session_state.qa_chain:
            answer = st.session_state.qa_chain.run({
                "question": user_input,
                "format": format_type
            })
            if "Insufficient data" in answer:
                raise ValueError("Fallback to search")
        else:
            raise ValueError("No pitch deck loaded")
    except:
        with st.spinner("🌐 Searching web for insights..."):
            answer = run_web_search(user_input)

    with st.chat_message("ai"):
        st.markdown(answer)
