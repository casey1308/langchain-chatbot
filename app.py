from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not set")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)


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
