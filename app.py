import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import csv
import difflib
import logging
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAIError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY missing.")
    st.stop()

st.set_page_config(page_title="Augmento FAQ Chatbot", page_icon="💼", layout="wide")
st.title("💼 Augmento FAQ Chatbot")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

# FAQ categories
faq_categories = {
    "Fundraising Process": {
        "What documents are needed for fundraising?":
            "You typically need a pitch deck, cap table, financial projections, and funding ask details.",
        "What is your typical check size?":
            "We invest ₹1.5–5 Cr depending on stage and traction.",
        "Do you lead rounds or co-invest?":
            "We lead, co-lead, or follow based on conviction and structure.",
    },
    "Evaluation & Due Diligence": {
        "What is the due diligence process like?":
            "We evaluate legal, financial, and business documentation.",
        "How long does it take to get an investment decision?":
            "Usually 3–6 weeks, depending on document readiness and team responsiveness.",
        "Do you invest in pre-revenue startups?":
            "Yes, if there’s a strong team and clear market need.",
    },
    "Investment Focus": {
        "What sectors do you focus on?":
            "We prefer tech-led consumer, B2B SaaS, healthtech, and sustainability sectors.",
    }
}

# Sidebar
st.sidebar.title("⚙️ Settings")
selected_category = st.sidebar.selectbox("📚 Choose FAQ Category", list(faq_categories.keys()))
bot_style = st.sidebar.selectbox("🎭 Bot Mood", ["Formal VC", "Friendly Analyst", "Cool Mentor"])

style_prompt_map = {
    "Formal VC": "Answer professionally like a VC during diligence.",
    "Friendly Analyst": "Be helpful and warm like an analyst helping a founder.",
    "Cool Mentor": "Respond casually like a startup mentor talking to a peer."
}

faq_data = faq_categories[selected_category]
faq_questions = list(faq_data.keys())

# User input
user_input = st.text_input("💬 Ask your question:", value=st.session_state.chat_input, key="chat_input")

# Suggestions
if user_input:
    suggestions = difflib.get_close_matches(user_input, faq_questions, n=3)
    if suggestions:
        st.markdown("🔎 **Suggested Questions:**")
        for s in suggestions:
            if st.button(f"👉 {s}", key=f"suggest_{s}"):
                st.session_state.chat_input = s
                st.rerun()

# Vector match
def get_best_faq_response(query):
    vectorizer = TfidfVectorizer().fit_transform([query] + faq_questions)
    sims = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    top_index = sims.argmax()
    return faq_questions[top_index], faq_data[faq_questions[top_index]]

# Buttons
col1, col2 = st.columns([1, 4])
with col1:
    send = st.button("🚀 Send")
with col2:
    clear = st.button("🗑️ Clear History")

# Handle send
if send and user_input.strip():
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.2)
        best_q, best_a = get_best_faq_response(user_input)
        tone = style_prompt_map.get(bot_style, "")
        prompt = f"""You are a startup investment chatbot.
Use this tone: {tone}

The user asked: "{user_input}"
This matched FAQ: "{best_q}"
Provide a helpful answer.

Answer:
{best_a}"""

        with st.spinner("🤖 Thinking..."):
            response = llm.invoke([
                SystemMessage(content=tone),
                HumanMessage(content=prompt)
            ])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.chat_history.append((user_input, response.content, timestamp))
        st.session_state.last_question = best_q

        # Append to CSV log
        with open("chat_log.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_input, response.content, ""])

        st.session_state.chat_input = ""  # reset input
        st.rerun()

    except OpenAIError as e:
        st.error(f"❌ OpenAI Error: {str(e)}")

# Handle clear
if clear:
    st.session_state.chat_history = []
    st.session_state.chat_input = ""
    st.experimental_rerun()

# Show conversation
if st.session_state.chat_history:
    st.markdown("## 🧾 Conversation")
    for i, (q, a, timestamp) in enumerate(st.session_state.chat_history[-10:]):
        with st.chat_message("user"):
            st.markdown(f"**{timestamp}**  \n{q}")
        with st.chat_message("assistant"):
            st.markdown(a)

                fb1, fb2, fb3 = st.columns(3)
        if fb1.button("❤️", key=f"like_{i}"):
            with open("chat_log.csv", "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            rows[-(i+1)][3] = "❤️"
            with open("chat_log.csv", "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            st.success("Thanks for the love!")
