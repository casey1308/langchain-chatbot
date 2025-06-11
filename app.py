import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Set Streamlit page configuration **at the top**
st.set_page_config(page_title="Manna - Your AI Buddy", page_icon="🤖")

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Handle missing API key
if not api_key:
    st.error("❌ OPENAI_API_KEY not found. Please set it in a .env file.")
    st.stop()

# Initialize OpenAI Chat LLM using LangChain
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key
)

# UI Title
st.title("🤖 Meet Manna - Your AI Chat Assistant")

# Intro message
with st.chat_message("ai"):
    st.markdown("Hi there! I'm **Manna**, your helpful AI assistant. Ask me anything!")

# Input box
user_input = st.chat_input("Say something...")

# Handle chat
if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Manna, a friendly and helpful AI assistant."),
        ("human", "{input}")
    ])
    chain = prompt | llm

    # Get AI response
    response = chain.invoke({"input": user_input})

    # Show AI reply
    with st.chat_message("ai"):
        st.markdown(response.content)
