import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Fetch the API key
openai_api_key = os.getenv("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")

# Validate
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in a .env file.")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=openai_api_key
)

# UI
st.set_page_config(page_title="Manna - Your AI Assistant", page_icon="🤖")
st.title("🤖 Meet Manna - Your AI Chat Assistant")

# Introduction message
with st.chat_message("ai"):
    st.markdown("Hi there! I'm **Manna**, your helpful AI assistant. Ask me anything!")

# Input
user_input = st.chat_input("Say something...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Manna, a friendly and helpful AI assistant."),
        ("human", "{input}")
    ])

    chain = prompt | llm
    response = chain.invoke({"input": user_input})

    with st.chat_message("ai"):
        st.markdown(response.content)
