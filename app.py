
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import csv
import pandas as pd
import logging
import requests
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta
from io import StringIO
from openai import OpenAIError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import hashlib

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key:
    st.error("❌ Please add your OPENAI_API_KEY to the .env file or Secrets.")
    st.stop()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "user_analytics" not in st.session_state:
    st.session_state.user_analytics = {
        "total_questions": 0,
        "session_start": datetime.now(),
        "question_categories": Counter(),
        "response_types": Counter(),
        "hourly_activity": Counter()
    }

# Load and Save Feedback
def load_feedback_from_csv():
    try:
        if os.path.exists("feedback_log.csv"):
            df = pd.read_csv("feedback_log.csv")
            st.session_state.feedback_log = df.to_dict('records')
    except Exception as e:
        logger.warning(f"Could not load feedback log: {e}")

def save_feedback_to_csv(feedback_entry):
    try:
        with open("feedback_log.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if os.path.getsize("feedback_log.csv") == 0:
                writer.writerow(["timestamp", "question", "response_preview", "rating", "category", "response_type", "comments"])
            writer.writerow([
                feedback_entry["timestamp"],
                feedback_entry["question"],
                feedback_entry["response_preview"],
                feedback_entry["rating"],
                feedback_entry["category"],
                feedback_entry["response_type"],
                feedback_entry.get("comments", "")
            ])
    except Exception as e:
        logger.warning(f"Failed to save feedback to CSV: {e}")

# Load feedback on startup
load_feedback_from_csv()

# Web Search via SerpAPI
def perform_web_search(query):
    if not serpapi_api_key:
        return "Web search requires SERPAPI_API_KEY to be configured."
    try:
        search_url = "https://serpapi.com/search"
        params = {
            "q": f"{query} startup investment venture capital",
            "api_key": serpapi_api_key,
            "engine": "google",
            "num": 3
        }
        response = requests.get(search_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "organic_results" in data:
                formatted_results = []
                for result in data["organic_results"][:3]:
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    if title and snippet:
                        formatted_results.append(f"• {title}: {snippet}")
                return "\n".join(formatted_results) if formatted_results else "No relevant results found."
            else:
                return "No search results available."
        else:
            return f"Search API returned status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Web search request error: {e}")
        return "Web search temporarily unavailable due to network issues."
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return f"Web search error: {str(e)}"

# FAQ Knowledge Base
faq_categories = {
    "Fundraising Process": {
        "What documents are needed for fundraising?": "You typically need a pitch deck, projections, cap table, funding details, and company registration.",
        "What is your typical check size?": "₹1.5 Cr to ₹5 Cr depending on stage. Pre-seed as low as ₹50L, Series A up to ₹10 Cr.",
        "Do you lead rounds or co-invest?": "We can lead, co-lead, or follow depending on the case.",
        "What is the fundraising timeline?": "Typically 2–4 months including meetings, diligence, and legal.",
        "What equity percentage do you typically take?": "We target 8–15% ownership depending on valuation and round."
    }
}

def get_best_faq_response(user_input):
    all_questions, all_answers, question_categories = [], [], []
    for category, qa_dict in faq_categories.items():
        for question, answer in qa_dict.items():
            all_questions.append(question)
            all_answers.append(answer)
            question_categories.append(category)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([user_input] + all_questions)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    top_index = similarities.argmax()
    top_score = similarities[top_index]

    return (all_questions[top_index], all_answers[top_index], question_categories[top_index], top_score)

# Feedback function
def collect_feedback(question, response, category, response_type):
    st.subheader("📝 Rate this response")
    hash_key = hashlib.md5(f"{question}_{response}".encode()).hexdigest()[:10]
    col1, col2 = st.columns([3, 2])
    with col1:
        rating = st.radio(
            "How helpful was this response?",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: "⭐" * x + " " + ["Very Poor", "Poor", "Average", "Good", "Excellent"][x-1],
            key=f"rating_{hash_key}",
            horizontal=True
        )
    with col2:
        submit_feedback = st.button("Submit Feedback", key=f"submit_{hash_key}")
    comments = st.text_area(
        "Additional comments (optional):",
        key=f"comments_{hash_key}",
        placeholder="Any specific feedback or suggestions..."
    )
    if submit_feedback:
        feedback_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "response_preview": response[:100] + "..." if len(response) > 100 else response,
            "rating": rating,
            "category": category,
            "response_type": response_type,
            "comments": comments
        }
        st.session_state.feedback_log.append(feedback_entry)
        save_feedback_to_csv(feedback_entry)
        st.success("Thank you for your feedback! 🙏")
        if rating == 5:
            st.balloons()

# UI
st.set_page_config(page_title="Investment FAQ Chatbot", page_icon="💼", layout="wide")
st.title("💼 Augmento – Your Investment Assistant")
st.markdown("*Ask investment-related questions below.*")

# Chat interface
user_input = st.text_input("Ask a question:", placeholder="e.g., What documents are needed for fundraising?")
if st.button("Send"):
    if user_input.strip():
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.3)
        best_question, best_answer, category, similarity_score = get_best_faq_response(user_input)
        with st.spinner("Thinking..."):
            if similarity_score >= 0.3:
                prompt = f"You are a helpful assistant. User asked: {user_input}. Based on FAQ: {best_answer}"
                response = llm.invoke([SystemMessage(content="You are a knowledgeable investment advisor."), HumanMessage(content=prompt)])
                final_response = response.content
                response_type = "📋 FAQ Response"
            else:
                search_result = perform_web_search(user_input)
                prompt = f"User asked: {user_input}. Search Results: {search_result}"
                response = llm.invoke([SystemMessage(content="You are a helpful investment assistant."), HumanMessage(content=prompt)])
                final_response = response.content + "\n\n🌐 Powered by web search"
                response_type = "🌐 Web Search"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append((user_input, final_response, timestamp, response_type, category))
        st.markdown(f"**{response_type}**")
        st.markdown(final_response)
        collect_feedback(user_input, final_response, category, response_type)
