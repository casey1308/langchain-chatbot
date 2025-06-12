import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from io import BytesIO
import openai
import av
import numpy as np
import wave

# LangChain and OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Audio recording
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found in .env.")
    st.stop()

# === LangChain QA Chain ===
def build_qa_chain(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=openai_api_key))

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are Manna, a friendly and helpful AI assistant. "
            "Use ONLY the following context to answer the user. "
            "If unsure, say you don’t know.\n\n{context}\n\nQuestion: {question}"
        ),
    )

    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

# === Audio Processor ===
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame):
        self.audio_frames.append(frame)
        return frame

    def get_audio_bytes(self):
        audio = b''.join(f.planes[0].to_bytes() for f in self.audio_frames)
        return audio

# === Transcription ===
def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        with wave.open(f.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(audio_bytes)
        f.seek(0)
        audio_file = open(f.name, "rb")
        result = openai.Audio.transcribe("whisper-1", audio_file, api_key=openai_api_key)
        return result['text']

# === Streamlit UI ===
st.set_page_config(page_title="Manna - AI Chatbot", page_icon="🤖")
st.title("🤖 Meet Manna - Your AI Chat Assistant")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("📄 Upload a PDF to chat with it", type=["pdf"])
if uploaded_file:
    with st.spinner("Indexing your document…"):
        st.session_state.qa_chain = build_qa_chain(uploaded_file)
    st.success("✅ Document indexed!")

with st.chat_message("ai"):
    st.markdown("Hi! I'm **Manna**. Ask me anything using text or voice.")

# === Voice Recording Section ===
with st.expander("🎙️ Record Your Voice"):
    ctx = webrtc_streamer(key="voice", audio_processor_factory=AudioProcessor)
    if ctx.audio_processor:
        if st.button("Transcribe Audio"):
            audio_bytes = ctx.audio_processor.get_audio_bytes()
            if audio_bytes:
                with st.spinner("Transcribing..."):
                    try:
                        user_input = transcribe_audio(audio_bytes)
                        st.success(f"🗣️ You said: **{user_input}**")
                        st.session_state.last_voice_input = user_input
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")

# === Chat Handling ===
text_input = st.chat_input("Type your message...") or st.session_state.get("last_voice_input")

if text_input:
    with st.chat_message("user"):
        st.markdown(text_input)

    if st.session_state.qa_chain:
        answer = st.session_state.qa_chain.run(text_input)
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Manna, a friendly and helpful AI assistant."),
            ("human", "{input}")
        ])
        chain = prompt | ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
        answer = chain.invoke({"input": text_input}).content

    with st.chat_message("ai"):
        st.markdown(answer)
