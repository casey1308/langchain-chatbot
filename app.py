import streamlit as st
import os
import csv
import logging
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAIError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from serpapi import GoogleSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serp_api_key = os.getenv("SERPAPI_KEY")

if not openai_api_key:
    st.error("❌ Please add your OPENAI_API_KEY to the .env file.")
    st.stop()
if not serp_api_key:
    st.error("❌ Please add your SERPAPI_KEY to the .env file.")
    st.stop()

# Page config
st.set_page_config(page_title="Investment FAQ Chatbot", page_icon="💼", layout="wide")
st.title("💼 Investment Process FAQ Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

faq_data = {
    "What documents are needed for fundraising?":
        "You typically need a pitch deck, cap table, financial projections, and details of your funding ask (amount, valuation, instrument).",
    "What is the due diligence process like?":
        "Due diligence includes evaluating your legal, financial, and business details. We’ll request company registration docs, past financials, founder backgrounds, customer data, etc.",
    "How long does it take to get an investment decision?":
        "It typically takes 3–6 weeks from the first call to decision, depending on how quickly we receive documents and conduct diligence.",
    "Do you invest in pre-revenue startups?":
        "Yes, we do evaluate pre-revenue startups if they are solving a clear problem with a strong founding team and early traction.",
    "What is your typical check size?":
        "We typically invest between ₹1.5 Cr to ₹5 Cr depending on the stage and category of the company.",
    "Do you lead rounds or co-invest?":
        "We are flexible. We can lead, co-lead, or follow depending on round dynamics and our conviction.",
    "What sectors do you focus on?":
        "We are sector-agnostic but have a preference for tech-led consumer businesses, B2B SaaS, healthtech, and sustainability.",
}

faq_questions = list(faq_data.keys())

def get_best_faq_response(user_input):
    vectorizer = TfidfVectorizer().fit_transform([user_input] + faq_questions)
    sims = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    top_index = sims.argmax()
    top_score = sims[top_index]
    return faq_questions[top_index], faq_data[faq_questions[top_index]], top_score

def fetch_from_serpapi(query):
    params = {
        "engine": "google",
        "q": query,
        "api_key": serp_api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    if "answer_box" in results:
        return results["answer_box"].get("answer") or results["answer_box"].get("snippet")
    elif "organic_results" in results and results["organic_results"]:
        return results["organic_results"][0].get("snippet")
    return "Sorry, I couldn’t find any relevant results online."

# UI
st.header("💬 Ask a Question About Investment Process")
user_input = st.text_input("Your question:", key="user_message")

col1, col2 = st.columns([1, 4])
with col1:
    send = st.button("Send", type="primary")
with col2:
    reset = st.button("🗑️ Clear History")

if send and user_input.strip():
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.2)
        best_q, best_a, score = get_best_faq_response(user_input)

        if score >= 0.4:
            prompt = f"""You are a professional investment FAQ assistant. The user asked: "{user_input}"
This is the closest FAQ: "{best_q}"
Answer concisely and expand slightly if helpful.

Answer:
{best_a}"""
        else:
            serp_result = fetch_from_serpapi(user_input)
            prompt = f"""The user asked: "{user_input}"
Here is some information from a web search:
"{serp_result}"
Use this to answer clearly and professionally."""

        with st.spinner("Thinking..."):
            response = llm.invoke([
                SystemMessage(content="Answer concisely and clearly."),
                HumanMessage(content=prompt)
            ])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.chat_history.append((user_input, response.content, timestamp))

        # Append to CSV log
        with open("chat_log.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_input, response.content, ""])  # feedback empty for now

        st.rerun()
    except OpenAIError as e:
        st.error(f"❌ OpenAI Error: {str(e)}")

# Clear history
if reset:
    st.session_state.chat_history = []
    st.experimental_rerun()

# Chat history + feedback
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
                    with open("chat_log.csv", "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button(f"👎 Not Helpful", key=f"down_{i}"):
                    with open("chat_log.csv", "r", encoding="utf-8") as f:
                        rows = list(csv.reader(f))
                    rows[-(i+1)][3] = "👎"
                    with open("chat_log.csv", "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows)
                    st.warning("Feedback noted. Thank you!")
