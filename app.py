import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Prompt template with Manna's introduction
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Manna, a friendly and helpful AI assistant. Always introduce yourself as Manna."),
    ("human", "{input}")
])

# Streamlit UI
st.set_page_config(page_title="Manna - AI Assistant", page_icon="🤖")
st.title("🤖 Manna – Your AI Assistant")

# Chat input
user_input = st.chat_input("Ask Manna anything...")

# Response
if user_input:
    st.chat_message("user").write(user_input)
    response = llm.invoke(prompt.format_messages(input=user_input))
    st.chat_message("ai").write(response.content)
