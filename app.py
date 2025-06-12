import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import openai
import wave
from io import BytesIO

# LangChain and OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Voice recording
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import av

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

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
        "You are Manna, a friendly and helpful AI assistant. "
        "Use ONLY the following context to answer the user. If the answer "
        "is not in the context, say you don’t know."
        "\n\n{context}\n\nQuestion: {question}"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"], template=system_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key),
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain

# Audio Processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_frames.append(frame)
        return frame

# Transcription Function
def transcribe_audio_frames(frames):
    if not frames:
        return ""

    audio = BytesIO()
    with wave.open(audio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        for frame in frames:
            wf.writeframes(frame.planes[0].to_bytes())

    audio.seek(0)
    response = openai.Audio.transcribe("whisper-1", audio, api_key=openai_api_key)
    return response["text"]

# UI Setup
st.set_page_config(page_title="Manna - Your AI Assistant", page_icon="🤖")
st.title("🤖 Meet Manna - Your AI Chat Assistant")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("📄 Upload a PDF to chat with it", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Indexing your document…"):
        st.session_state.qa_chain = build_qa_chain(uploaded_file)
    st.success("✅ Document indexed! Ask your questions below.")

with st.chat_message("ai"):
    st.markdown("Hi there! I'm **Manna**, your helpful AI assistant. Ask me anything!")

# Voice input section
with st.expander("🎙️ Record Your Voice"):
    st.write("Press the 'Start' button below and speak. Then click 'Transcribe Audio'.")
    ctx = webrtc_streamer(key="speech", mode="sendonly", audio_processor_factory=AudioProcessor)

    if ctx.state.playing and ctx.audio_processor:
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing your voice..."):
                try:
                    user_input = transcribe_audio_frames(ctx.audio_processor.audio_frames)
                    if user_input:
                        st.success(f"🗣️ You said: **{user_input}**")
                    else:
                        st.warning("⚠️ No audio captured. Please try again.")
                except Exception as e:
                    st.error(f"❌ Transcription failed: {str(e)}")

# Text input
user_input = st.chat_input("Say something…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.qa_chain is not None:
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
