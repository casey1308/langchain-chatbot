import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import csv
import pandas as pd
import logging
from dotenv import load_dotenv
from datetime import datetime
from io import StringIO
from openai import OpenAIError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.tools import SerpAPIWrapper

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key:
    st.error("❌ Please add your OPENAI_API_KEY to the .env file or Secrets.")
    st.stop()
if not serpapi_api_key:
    st.error("❌ Please add your SERPAPI_API_KEY to the .env file or Secrets.")
    st.stop()

serp_tool = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Categories
faq_categories = {
    "Fundraising Process": {
        "What documents are needed for fundraising?":
            "You typically need a pitch deck, cap table, financial projections, and details of your funding ask (amount, valuation, instrument).",
        "What is your typical check size?":
            "We typically invest between ₹1.5 Cr to ₹5 Cr depending on the stage and category of the company.",
        "Do you lead rounds or co-invest?":
            "We are flexible. We can lead, co-lead, or follow depending on round dynamics and our conviction.",
    },
    "Evaluation & Due Diligence": {
        "What is the due diligence process like?":
            "Due diligence includes evaluating your legal, financial, and business details. We'll request company registration docs, past financials, founder backgrounds, customer data, etc.",
        "How long does it take to get an investment decision?":
            "It typically takes 3–6 weeks from the first call to decision, depending on how quickly we receive documents and conduct diligence.",
        "Do you invest in pre-revenue startups?":
            "Yes, we do evaluate pre-revenue startups if they are solving a clear problem with a strong founding team and early traction.",
    },
    "Investment Focus": {
        "What sectors do you focus on?":
            "We are sector-agnostic but have a preference for tech-led consumer businesses, B2B SaaS, healthtech, and sustainability.",
    }
}

# Select category
st.sidebar.title("📚 FAQ Category")
selected_category = st.sidebar.selectbox("Choose a category", list(faq_categories.keys()))
faq_data = faq_categories[selected_category]
faq_questions = list(faq_data.keys())

# Get best FAQ match
def get_best_faq_response(user_input):
    vectorizer = TfidfVectorizer().fit_transform([user_input] + faq_questions)
    sims = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    top_index = sims.argmax()
    top_score = sims[top_index]
    return faq_questions[top_index], faq_data[faq_questions[top_index]], top_score

# Header and input
st.set_page_config(page_title="Investment FAQ Chatbot", page_icon="💼", layout="wide")
st.title("💼 Investment Process FAQ Chatbot")

st.header("💬 Ask a Question")

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

input_key = "user_input"
if st.session_state.clear_input:
    st.session_state.clear_input = False
    if input_key in st.session_state:
        del st.session_state[input_key]

user_input = st.text_input("Your question:", key=input_key)

col1, col2 = st.columns([1, 4])
with col1:
    send = st.button("Send", type="primary")
with col2:
    reset = st.button("🗑️ Clear History")

if send and user_input.strip():
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.2)
        best_q, best_a, score = get_best_faq_response(user_input)

        prompt = f"""You are a professional investment FAQ assistant. The user asked: \"{user_input}\"
This is the closest FAQ: \"{best_q}\"
Answer concisely and expand slightly if helpful.

Answer:
{best_a}"""

        with st.spinner("Thinking..."):
            response = llm.invoke([
                SystemMessage(content="Answer concisely and clearly."),
                HumanMessage(content=prompt)
            ])

        final_response = response.content

        # Web search fallback if FAQ match is poor
        if score < 0.3:
            web_result = serp_tool.run(user_input)
            final_response += f"\n\n🌐 Web Search Result:\n{web_result}"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.chat_history.append((user_input, final_response, timestamp))

        with open("chat_log.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_input, final_response, ""])

        st.session_state.clear_input = True
        st.rerun()

    except OpenAIError as e:
        st.error(f"❌ OpenAI Error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")

if reset:
    st.session_state.chat_history = []
    st.rerun()

if st.session_state.chat_history:
    st.subheader("📜 Chat History")
    for i, (q, a, timestamp) in enumerate(reversed(st.session_state.chat_history[-10:])):
        with st.expander(f"{i+1}. {q} ({timestamp})", expanded=(i == 0)):
            st.markdown(f"**🤖 Answer:** {a}")
