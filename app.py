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

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ Please add your OPENAI_API_KEY to the .env file or Secrets.")
    st.stop()

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
            "Due diligence includes evaluating your legal, financial, and business details. We’ll request company registration docs, past financials, founder backgrounds, customer data, etc.",
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
user_input = st.text_input("Your question:", key="user_message")

col1, col2 = st.columns([1, 4])
with col1:
    send = st.button("Send", type="primary")
with col2:
    reset = st.button("🗑️ Clear History")

# Process user query
if send and user_input.strip():
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.2)
        best_q, best_a, score = get_best_faq_response(user_input)

        prompt = f"""You are a professional investment FAQ assistant. The user asked: "{user_input}"
This is the closest FAQ: "{best_q}"
Answer concisely and expand slightly if helpful.

Answer:
{best_a}"""

        with st.spinner("Thinking..."):
            response = llm.invoke([
                SystemMessage(content="Answer concisely and clearly."),
                HumanMessage(content=prompt)
            ])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.chat_history.append((user_input, response.content, timestamp))

        with open("chat_log.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_input, response.content, ""])

        st.rerun()
    except OpenAIError as e:
        st.error(f"❌ OpenAI Error: {str(e)}")

# Clear history
if reset:
    st.session_state.chat_history = []
    st.experimental_rerun()

# Display chat history
if st.session_state.chat_history:
    st.subheader("📜 Chat History")
    for i, (q, a, timestamp) in enumerate(reversed(st.session_state.chat_history[-10:])):
        with st.expander(f"{i+1}. {q} ({timestamp})", expanded=(i == 0)):
            st.markdown(f"**🤖 Answer:** {a}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"👍 Helpful", key=f"up_{i}"):
                    with open("chat_log.csv", "r", encoding="utf-8") as f:
                        rows = list(csv.reader(f))
                    rows[-(i+1)][3] = "👍"
                    with open("chat_log.csv", "w", newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button(f"👎 Not Helpful", key=f"down_{i}"):
                    with open("chat_log.csv", "r", encoding="utf-8") as f:
                        rows = list(csv.reader(f))
                    rows[-(i+1)][3] = "👎"
                    with open("chat_log.csv", "w", newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)
                    st.warning("Feedback noted. Thank you!")

# Download Chat Log
if os.path.exists("chat_log.csv"):
    with open("chat_log.csv", "r", encoding="utf-8") as f:
        chat_log_data = f.read()
    st.download_button("📥 Export Chat Log", data=chat_log_data, file_name="chat_log.csv", mime="text/csv")

# Feedback Summary
if os.path.exists("chat_log.csv"):
    df = pd.read_csv("chat_log.csv", names=["Timestamp", "Question", "Answer", "Feedback"])
    pos = df["Feedback"].value_counts().get("👍", 0)
    neg = df["Feedback"].value_counts().get("👎", 0)
    st.sidebar.markdown("### 📊 Feedback Summary")
    st.sidebar.metric("👍 Helpful", pos)
    st.sidebar.metric("👎 Not Helpful", neg)

# 📊 Feedback Analytics
st.sidebar.markdown("---")
if st.sidebar.checkbox("📊 Show Feedback Analytics"):
    st.subheader("📈 Feedback Analytics")

    if os.path.exists("chat_log.csv"):
        df = pd.read_csv("chat_log.csv", names=["Timestamp", "Question", "Answer", "Feedback"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Line chart of feedback over time
        feedback_timeline = df[df["Feedback"].isin(["👍", "👎"])].copy()
        feedback_timeline["Date"] = feedback_timeline["Timestamp"].dt.date
        trend = feedback_timeline.groupby(["Date", "Feedback"]).size().reset_index(name="Count")

        import altair as alt
        chart = alt.Chart(trend).mark_line(point=True).encode(
            x="Date:T",
            y="Count:Q",
            color="Feedback:N"
        ).properties(width=700, height=300, title="Feedback Trend Over Time")

        st.altair_chart(chart, use_container_width=True)

        # Most liked questions
        st.markdown("#### 🥇 Top Helpful Questions")
        pos_df = df[df["Feedback"] == "👍"]["Question"].value_counts().head(5)
        st.write(pos_df)

        # Most disliked questions
        st.markdown("#### ⚠️ Most Unhelpful Questions")
        neg_df = df[df["Feedback"] == "👎"]["Question"].value_counts().head(5)
        st.write(neg_df)

        # Bar chart of feedback summary
        feedback_summary = df["Feedback"].value_counts().reset_index()
        feedback_summary.columns = ["Feedback", "Count"]
        bar = alt.Chart(feedback_summary).mark_bar().encode(
            x="Feedback:N",
            y="Count:Q",
            color="Feedback:N"
        ).properties(width=400, height=200)
        st.altair_chart(bar)
