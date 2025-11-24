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
# Keep your LangChain/OpenAI client imports as used previously
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key:
    st.error("❌ Please add your OPENAI_API_KEY to the .env file or Secrets.")
    st.stop()

def perform_web_search(query):
    # keep this function but it's optional for coding queries; fallback only
    if not serpapi_api_key:
        return "Web search requires SERPAPI_API_KEY to be configured."
    try:
        search_url = "https://serpapi.com/search"
        params = {
            "q": query,
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

# Session state initialization (kept)
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

def load_feedback_from_csv():
    try:
        if os.path.exists("feedback_log.csv"):
            df = pd.read_csv("feedback_log.csv")
            st.session_state.feedback_log = df.to_dict('records')
    except Exception as e:
        logger.warning(f"Could not load feedback log: {e}")

def save_feedback_to_csv(feedback_entry):
    try:
        file_exists = os.path.exists("feedback_log.csv")
        with open("feedback_log.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize("feedback_log.csv") == 0:  # Write header if file is empty
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

load_feedback_from_csv()

# --- FAQ categories rewritten for technical interview domain ---
faq_categories = {
    "Coding & Algorithms": {
        "How to approach two-pointer problems?":
            "Identify when the array/string can be processed from both ends towards center. Use left/right pointers, move pointers based on condition checks to reduce complexity to O(n).",
        "What is the difference between DFS and BFS?":
            "DFS uses a stack (recursion or explicit) and explores deep paths first; BFS uses a queue and explores level by level. Use BFS for shortest path in unweighted graphs and DFS for path existence and topological tasks.",
        "When to use hashing vs sorting?":
            "Hashing gives O(1) average lookup — use for membership checks, frequency counts. Sorting is O(n log n) and useful when order matters or two-pointer techniques are needed after sorting."
    },
    "System Design Basics": {
        "How to design a URL shortener?":
            "Use a key-value store mapping short codes to long URLs, generate unique short keys (base62), include a cache layer (Redis) and a persistent store (Postgres). Consider analytics, custom aliases, and collision handling.",
        "How to scale a chat application?":
            "Use WebSocket or WebRTC for real-time connections, partition users across servers, use message brokers (Kafka) and presence services, deploy sticky sessions or centralized connection managers, and horizontally scale stateless backend services."
    },
    "Behavioral & HR": {
        "How to answer 'Tell me about yourself' in 2 minutes?":
            "Start with a brief background, highlight 2-3 key technical projects or achievements, mention relevant skills, and end with current goals aligned to the role you're interviewing for.",
        "How to explain a failure in an interview?":
            "Use the STAR format: Situation, Task, Action, Result. Focus on what you learned and how you improved afterward."
    },
    "Coding Practice & Tools": {
        "How to set up local test cases for scripts?":
            "Use input redirection, pytest for unit tests, and create sample input/output files. Automate tests with CI and write edge-case tests.",
        "Which editors and plugins help coding interviews?":
            "VSCode with Code Runner, file templates, snippets, and language-specific linters. Practice in an environment that mirrors the interview editor (e.g., Monaco-based editors)."
    }
}

def get_best_faq_response(user_input):
    # Similarity-based matching for FAQ questions (same as before)
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

# Feedback UI preserved
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

# Keep analytics display function unchanged (works for new domain)
def display_analytics():
    if not st.session_state.feedback_log:
        st.info("No feedback data available yet. Start asking questions and rating responses to see analytics!")
        return

    df_feedback = pd.DataFrame(st.session_state.feedback_log)

    df_feedback['timestamp'] = pd.to_datetime(df_feedback['timestamp'])
    df_feedback['date'] = df_feedback['timestamp'].dt.date
    df_feedback['hour'] = df_feedback['timestamp'].dt.hour

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_rating = df_feedback['rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.2f}/5",
                 delta=f"{avg_rating-3:.2f}" if avg_rating >= 3 else f"{avg_rating-3:.2f}")

    with col2:
        total_feedback = len(df_feedback)
        st.metric("Total Responses", total_feedback)

    with col3:
        satisfaction_rate = (df_feedback['rating'] >= 4).mean() * 100
        st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")

    with col4:
        recent_feedback = df_feedback[df_feedback['timestamp'] >= datetime.now() - timedelta(days=7)]
        st.metric("This Week", len(recent_feedback))

    tab1, tab2, tab3, tab4 = st.tabs(["Ratings", "Categories", "Activity", "Comments"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Rating Distribution")
            rating_counts = df_feedback['rating'].value_counts().sort_index()
            rating_df = pd.DataFrame({
                'Rating': rating_counts.index,
                'Count': rating_counts.values
            })
            st.bar_chart(rating_df.set_index('Rating'))

            for rating in range(1, 6):
                count = rating_counts.get(rating, 0)
                percentage = (count / len(df_feedback)) * 100 if len(df_feedback) > 0 else 0
                st.write(f"⭐ {rating} stars: {count} responses ({percentage:.1f}%)")

        with col2:
            st.subheader("📈 Rating Trend Over Time")
            daily_ratings = df_feedback.groupby('date')['rating'].mean().reset_index()
            if len(daily_ratings) > 0:
                daily_ratings = daily_ratings.set_index('date')
                st.line_chart(daily_ratings)

                if len(daily_ratings) >= 2:
                    recent_trend = daily_ratings['rating'].iloc[-1] - daily_ratings['rating'].iloc[-2]
                    trend_emoji = "📈" if recent_trend > 0 else "📉" if recent_trend < 0 else "➡️"
                    st.write(f"Recent trend: {trend_emoji} {recent_trend:+.2f}")
            else:
                st.info("Need more data points to show trend")

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Category Performance")
            category_stats = df_feedback.groupby('category').agg({
                'rating': ['mean', 'count']
            }).round(2)
            category_stats.columns = ['Avg_Rating', 'Count']
            category_stats = category_stats.sort_values('Avg_Rating', ascending=False)

            category_chart_df = category_stats.reset_index()
            category_chart_df = category_chart_df.set_index('category')
            st.bar_chart(category_chart_df['Avg_Rating'])

            st.write("**Detailed Category Stats:**")
            for category, row in category_stats.iterrows():
                st.write(f"• **{category}**: {row['Avg_Rating']:.2f}⭐ ({row['Count']} responses)")

        with col2:
            st.subheader("Response Type Performance")
            response_type_stats = df_feedback.groupby('response_type').agg({
                'rating': ['mean', 'count']
            }).round(2)
            response_type_stats.columns = ['Avg Rating', 'Count']

            st.dataframe(response_type_stats, use_container_width=True)

            if len(response_type_stats) > 0:
                response_chart_df = response_type_stats.reset_index().set_index('response_type')
                st.bar_chart(response_chart_df['Avg Rating'])

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🕐 Activity by Hour")
            hourly_activity = df_feedback.groupby('hour').size()
            if len(hourly_activity) > 0:
                hourly_df = pd.DataFrame({
                    'Hour': hourly_activity.index,
                    'Interactions': hourly_activity.values
                }).set_index('Hour')
                st.bar_chart(hourly_df)

                peak_hour = hourly_activity.idxmax()
                peak_count = hourly_activity.max()
                st.write(f"🔥 Peak activity: {peak_hour}:00 ({peak_count} interactions)")
            else:
                st.info("No hourly data available yet")

        with col2:
            st.subheader("Daily Activity Trend")
            daily_activity = df_feedback.groupby('date').size()
            if len(daily_activity) > 0:
                daily_df = pd.DataFrame({
                    'Date': daily_activity.index,
                    'Interactions': daily_activity.values
                }).set_index('Date')
                st.line_chart(daily_df)
                avg_daily = daily_activity.mean()
                max_daily = daily_activity.max()
                st.write(f"Average daily interactions: {avg_daily:.1f}")
                st.write(f"Highest single day: {max_daily} interactions")
            else:
                st.info("Need more data to show daily trends")

    with tab4:
        st.subheader("Recent Comments")
        recent_comments = df_feedback[df_feedback['comments'].notna() & (df_feedback['comments'] != '')].tail(10)

        if not recent_comments.empty:
            for _, row in recent_comments.iterrows():
                with st.expander(f"⭐{row['rating']} - {row['category']} ({row['timestamp']})"):
                    st.write(f"**Question:** {row['question']}")
                    st.write(f"**Comment:** {row['comments']}")
        else:
            st.info("No comments yet. Users can leave comments when rating responses.")

    with st.expander("📋 Detailed Feedback Data"):
        st.dataframe(df_feedback.sort_values('timestamp', ascending=False), use_container_width=True)

        csv_data = df_feedback.to_csv(index=False)
        st.download_button(
            label="📥 Download Feedback Data",
            data=csv_data,
            file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Sidebar with technical categories
st.sidebar.title("FAQ Categories")
selected_category = st.sidebar.selectbox("Choose a category", list(faq_categories.keys()))
faq_data = faq_categories[selected_category]
faq_questions = list(faq_data.keys())

st.sidebar.subheader(f"📋 {selected_category} FAQs")
for i, question in enumerate(faq_questions, 1):
    st.sidebar.write(f"{i}. {question}")

st.sidebar.info("💡 Tip: Use the main chat to practice coding questions, request mock interviews, or ask for code reviews.")

if st.session_state.feedback_log:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Stats")
    df_temp = pd.DataFrame(st.session_state.feedback_log)
    avg_rating = df_temp['rating'].mean()
    st.sidebar.metric("Avg Rating", f"{avg_rating:.1f}⭐")
    st.sidebar.metric("Total Feedback", len(df_temp))

# Page config and CSS tweaks
st.set_page_config(page_title="Elevate Master - Interview Mentor", page_icon="🧠", layout="wide")
st.markdown("""
    <style>
        .block-container { padding-top: 1rem !important; }
        section[data-testid="stSidebar"] > div:first-child { padding-top: 0rem !important; margin-top: -1rem !important; }
        button[data-testid="collapsedControl"] {
            visibility: visible !important; display: block !important; position: fixed !important;
            top: 4rem !important; left: 0.7rem !important; z-index: 1001 !important;
            background-color: white !important; border: 1px solid #ccc !important; border-radius: 6px !important;
            width: 2.2rem; height: 2.2rem; padding: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description for Elevate Master
st.title("Elevate Master - Hybrid Interview Mentor")
st.markdown("*Practice mock technical interviews, solve coding problems, get instant feedback and learn — interaction mode: HYBRID (mock interviewer + teacher)*")
st.markdown("**How to use:** Ask a coding question, paste code for review, request a mock interview prompt, or ask for behavioral answer prep. The assistant will first simulate interview behavior, then provide a teaching-style explanation and actionable improvement steps.")

main_tab, analytics_tab = st.tabs(["Chatbot", "Analytics"])

with main_tab:
    st.header("Ask your technical question or start a mock interview")

    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False

    input_key = "user_input"
    if st.session_state.clear_input:
        st.session_state.clear_input = False
        if input_key in st.session_state:
            del st.session_state[input_key]

    user_input = st.text_input("Type your technical question, paste code, or say 'mock interview' to start:", key=input_key, placeholder="e.g., 'Implement binary search in python' or paste code here")

    col1, col2 = st.columns([5,1])
    with col1:
        send = st.button("Send", type="primary")
    with col2:
        reset = st.button("Clear History")

    if send and user_input.strip():
        try:
            # instantiate LLM client (keeps your prior usage)
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.15)

            best_question, best_answer, category, similarity_score = get_best_faq_response(user_input)

            st.session_state.user_analytics["total_questions"] += 1
            st.session_state.user_analytics["question_categories"][category] += 1
            current_hour = datetime.now().hour
            st.session_state.user_analytics["hourly_activity"][current_hour] += 1

            # Build hybrid-system prompt for "Elevate Master"
            with st.spinner("Elevate Master is preparing your session..."):
                # If user input matches a FAQ strongly, return a short, focused FAQ-based response first
                if similarity_score >= 0.35:
                    # Use the FAQ answer as quick guidance, then ask if user wants deep dive
                    response_text = f"**Quick Answer ({category}):**\n\n{best_answer}\n\nIf you'd like a deeper mock-interview style interaction (example tests, code, hints, follow-ups), reply 'deep dive' or ask for 'mock interview'."
                    final_response = response_text
                    response_type = f"📋 FAQ Quick Response ({category})"
                    st.session_state.user_analytics["response_types"]["FAQ"] += 1
                else:
                    # For coding / mock interview requests, create a hybrid prompt:
                    # 1) act as a short mock interviewer: ask 1-2 clarifying Qs or give a prompt
                    # 2) evaluate code if present
                    # 3) provide solution, complexity, edge-cases, tests, and improvement suggestions
                    # 4) offer 2 follow-up mock questions and one timed challenge
                    # Check if user pasted code (simple heuristic)
                    contains_code = ("```" in user_input) or ("\n" in user_input and ("def " in user_input or "class " in user_input or ";" in user_input))
                    if contains_code:
                        instruction = f"""
You are Elevate Master — a hybrid mock interview system and tutor. The user provided code or a coding question. Follow this hybrid workflow:

1) Act like an interviewer: give a short, pointed assessment of the submitted code (1-2 concise bullets).
2) Detect bugs or inefficiencies and propose exact fixes. If you can, provide corrected code with minimal changes.
3) Provide runtime and space complexity, and point out edge cases and test inputs (include at least 3 test cases).
4) Offer a succinct explanation of the algorithm and why your fix works.
5) Suggest 2 follow-up questions that an interviewer might ask based on this solution (one harder, one simpler).
6) Provide a short 'coach' section with actionable study tips and resources.

User input:
\"\"\"{user_input}\"\"\"
Be concise in the interviewer-style bullets, then expand in the tutor/coach section.
"""
                    else:
                        # User asked a question or wants mock interview
                        instruction = f"""
You are Elevate Master — hybrid interview mentor (mock interviewer + teacher). The user will be treated as an interview candidate.

Follow this flow:
1) If user asked for a specific problem (e.g., 'implement X'), present a short mock prompt and ask 1 clarifying question or give constraints.
2) If user asked conceptual or behavioral question, answer briefly as an interviewer would evaluate, then provide a teaching-style explanation.
3) Always include:
   - Example solution outline or pseudocode
   - Time and space complexity
   - 3 test cases (input -> expected output)
   - One 'follow-up' interviewer question and one 'hint' for it
   - A short coach note with study advice and common pitfalls
User question:
\"\"\"{user_input}\"\"\"
Be direct and structured. Start with interviewer-tone (short), then tutor-tone with details and code where applicable.
"""
                    # Call LLM
                    response = llm.invoke([
                        SystemMessage(content="You are Elevate Master — a hybrid mock-interviewer and technical coach. Be professional, precise, and helpful."),
                        HumanMessage(content=instruction)
                    ])
                    final_response = response.content
                    response_type = "🤖 Hybrid Interview-Tutor Response"
                    st.session_state.user_analytics["response_types"]["Hybrid"] += 1

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append((user_input, final_response, timestamp, response_type, category))

            # Log chat to CSV
            try:
                with open("chat_log.csv", mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, user_input, final_response, response_type])
            except Exception as e:
                logger.warning(f"Failed to log to CSV: {e}")

            st.session_state.clear_input = True
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    if reset:
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
        st.rerun()

    if st.session_state.chat_history:
        st.header("Recent Conversations")
        for i, (question, answer, timestamp, resp_type, category) in enumerate(reversed(st.session_state.chat_history[-8:])):
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {question} ({timestamp})", expanded=(i == 0)):
                st.markdown(f"**{resp_type}**")
                # Display answer formatted - allow code blocks
                st.markdown(answer)
                st.markdown("---")
                collect_feedback(question, answer, category, resp_type)

st.markdown("---")
st.markdown("💡 **Tip:** To get the best mock-interview practice: paste your code, ask for a 'timed mock' or request 'whiteboard explanation'. Ask for 'follow-ups' to simulate an aggressive interviewer.")

st.markdown("---")

st.subheader(f"📋 FAQs for: {selected_category}")
for question, answer in faq_categories[selected_category].items():
    with st.expander(f"❓ {question}"):
        st.markdown(answer)


with analytics_tab:
    st.header("Feedback Analytics Dashboard")
    display_analytics()
