import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from io import BytesIO
import base64
import openai
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found. Please add it in a .env file.")
    st.stop()

st.set_page_config(page_title="Manna - Voice Chatbot", page_icon="🎙️")
st.title("🎙️ Talk to Manna - Live Mic Voice Input")

# Mic button
st.markdown("Click the button below and speak. It records 5 seconds of audio:")
audio_bytes = streamlit_js_eval(js_expressions="await record_audio()", key="audio")

user_input = None

if audio_bytes:
    st.audio(audio_bytes, format="audio/webm")
    with st.spinner("Transcribing..."):
        try:
            audio_file = BytesIO(audio_bytes)
            transcript = openai.Audio.transcribe("whisper-1", audio_file, api_key=openai_api_key)
            user_input = transcript["text"]
            st.success(f"🗣️ You said: **{user_input}**")
        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")

# Fallback: manual text
user_input = st.chat_input("Or type your question here") or user_input

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Simple direct reply from Manna
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        api_key=openai_api_key,
        messages=[
            {"role": "system", "content": "You are Manna, a helpful and friendly AI assistant."},
            {"role": "user", "content": user_input},
        ]
    )
    response = chat.choices[0].message["content"]

    with st.chat_message("ai"):
        st.markdown(response)
