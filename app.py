import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import csv
import pandas as pd
import logging
import requests
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from datetime import datetime, timedelta
from io import StringIO
from openai import OpenAIError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key:
    st.error("❌ Please add your OPENAI_API_KEY to the .env file or Secrets.")
    st.stop()

# Simple web search function using SerpAPI REST API
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

# Load existing feedback from CSV if exists
def load_feedback_from_csv():
    try:
        if os.path.exists("feedback_log.csv"):
            df = pd.read_csv("feedback_log.csv")
            st.session_state.feedback_log = df.to_dict('records')
    except Exception as e:
        logger.warning(f"Could not load feedback log: {e}")

# Save feedback to CSV
def save_feedback_to_csv(feedback_entry):
    try:
        with open("feedback_log.csv", mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if os.path.getsize("feedback_log.csv") == 0:  # Write header if file is empty
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

# FAQ Categories
faq_categories = {
    "Fundraising Process": {
        "What documents are needed for fundraising?":
            "You typically need a pitch deck (10-12 slides), detailed financial projections (3-5 years), current cap table, funding ask details (amount, valuation, instrument type), company registration documents, and any existing investor commitments.",
        "What is your typical check size?":
            "We typically invest between ₹1.5 Cr to ₹5 Cr depending on the stage and category of the company. For pre-seed, we might go as low as ₹50L, and for Series A, we can go up to ₹10 Cr in exceptional cases.",
        "Do you lead rounds or co-invest?":
            "We are flexible in our approach. We can lead rounds (especially in pre-seed and seed), co-lead with other VCs, or follow in larger rounds depending on our conviction level and the startup's needs.",
        "What is the fundraising timeline?":
            "The typical fundraising process takes 2-4 months from start to close. This includes 2-3 weeks for initial meetings, 3-4 weeks for due diligence, and 2-4 weeks for documentation and legal closure.",
        "What equity percentage do you typically take?":
            "We typically seek 8-15% equity stake, depending on the stage, valuation, and our check size. We're flexible and focus more on fair valuation than specific ownership percentages."
    },
    "Evaluation & Due Diligence": {
        "What is the due diligence process like?":
            "Our due diligence covers legal (incorporation, IP, contracts), financial (revenue, unit economics, projections), technical (product demo, architecture), market (size, competition), and team (backgrounds, references) aspects.",
        "How long does it take to get an investment decision?":
            "From first meeting to term sheet: 3-6 weeks. After term sheet acceptance, legal closure takes another 2-4 weeks. We aim to be transparent about our decision timeline throughout the process.",
        "Do you invest in pre-revenue startups?":
            "Yes, we evaluate pre-revenue startups if they demonstrate strong founder-market fit, clear problem-solution validation, early user traction (even if not monetized), and a viable path to revenue within 12-18 months.",
        "What metrics do you focus on during evaluation?":
            "For early-stage: user engagement, retention, NPS, founder-market fit. For growth-stage: ARR growth, customer acquisition cost, lifetime value, gross margins, burn rate, and runway.",
        "Do you require board seats?":
            "For investments above ₹2 Cr, we typically seek a board seat or board observer rights. For smaller investments, we're flexible and may only require information rights and investor updates."
    },
    "Investment Focus": {
        "What sectors do you focus on?":
            "We are sector-agnostic but have strong preferences for tech-enabled businesses, B2B SaaS, consumer tech, healthtech, fintech, edtech, and sustainability/climate tech solutions.",
        "Do you invest in B2C or B2B companies?":
            "We invest in both B2C and B2B companies. Our B2C focus is on consumer tech with strong network effects or unique value propositions. For B2B, we prefer SaaS models with recurring revenue.",
        "What stage companies do you invest in?":
            "We primarily invest in pre-seed to Series A stages. Occasionally, we participate in seed extensions or Series A extensions for our portfolio companies.",
        "Do you have any geographic preferences?":
            "We primarily focus on Indian startups, especially those based in major startup hubs like Bangalore, Mumbai, Delhi NCR, Pune, and Hyderabad. We're open to startups from other cities with strong execution capabilities."
    },
    "Post-Investment Support": {
        "What support do you provide after investment?":
            "We provide strategic guidance, introductions to customers and partners, help with hiring key positions, support for follow-on fundraising, and access to our network of mentors and advisors.",
        "How often do you interact with portfolio companies?":
            "We have monthly check-ins with all portfolio companies, quarterly board meetings (where applicable), and are available for ad-hoc support as needed. We believe in being hands-on without being intrusive.",
        "Do you help with follow-on funding?":
            "Yes, we actively support our portfolio companies in raising follow-on rounds by making introductions to other VCs, helping refine pitch materials, and often participating in subsequent rounds ourselves."
    }
}

# FAQ matching function
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

# Feedback collection function
def collect_feedback(question, response, category, response_type):
    st.subheader("📝 Rate this response")
    
    feedback_key = f"feedback_{len(st.session_state.chat_history)}"
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        rating = st.radio(
            "How helpful was this response?",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: "⭐" * x + " " + ["Very Poor", "Poor", "Average", "Good", "Excellent"][x-1],
            key=f"rating_{feedback_key}",
            horizontal=True
        )
    
    with col2:
        submit_feedback = st.button("Submit Feedback", key=f"submit_{feedback_key}")
    
    comments = st.text_area(
        "Additional comments (optional):",
        key=f"comments_{feedback_key}",
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
        st.balloons()

# Analytics functions
def display_analytics():
    if not st.session_state.feedback_log:
        st.info("📊 No feedback data available yet. Start asking questions and rating responses to see analytics!")
        return
    
    df_feedback = pd.DataFrame(st.session_state.feedback_log)
    
    # Convert timestamp to datetime
    df_feedback['timestamp'] = pd.to_datetime(df_feedback['timestamp'])
    df_feedback['date'] = df_feedback['timestamp'].dt.date
    df_feedback['hour'] = df_feedback['timestamp'].dt.hour
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_rating = df_feedback['rating'].mean()
        st.metric("📊 Average Rating", f"{avg_rating:.2f}/5", 
                 delta=f"{avg_rating-3:.2f}" if avg_rating >= 3 else f"{avg_rating-3:.2f}")
    
    with col2:
        total_feedback = len(df_feedback)
        st.metric("📝 Total Responses", total_feedback)
    
    with col3:
        satisfaction_rate = (df_feedback['rating'] >= 4).mean() * 100
        st.metric("😊 Satisfaction Rate", f"{satisfaction_rate:.1f}%")
    
    with col4:
        recent_feedback = df_feedback[df_feedback['timestamp'] >= datetime.now() - timedelta(days=7)]
        st.metric("📅 This Week", len(recent_feedback))
    
    # Charts
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Ratings", "📋 Categories", "🕐 Activity", "💬 Comments"])
    
    with tab1:
        # Rating distribution
        col1, col2 = st.columns(2)
        
        with col1:
            rating_counts = df_feedback['rating'].value_counts().sort_index()
            fig_rating = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Count'},
                title="Rating Distribution",
                color=rating_counts.values,
                color_continuous_scale="RdYlGn"
            )
            fig_rating.update_layout(showlegend=False)
            st.plotly_chart(fig_rating, use_container_width=True)
        
        with col2:
            # Rating trend over time
            daily_ratings = df_feedback.groupby('date')['rating'].mean().reset_index()
            fig_trend = px.line(
                daily_ratings,
                x='date',
                y='rating',
                title="Average Rating Trend",
                markers=True
            )
            fig_trend.update_yaxis(range=[1, 5])
            st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Category performance
            category_stats = df_feedback.groupby('category').agg({
                'rating': ['mean', 'count']
            }).round(2)
            category_stats.columns = ['Avg Rating', 'Count']
            category_stats = category_stats.sort_values('Avg Rating', ascending=False)
            
            fig_category = px.bar(
                x=category_stats.index,
                y=category_stats['Avg Rating'],
                title="Average Rating by Category",
                color=category_stats['Avg Rating'],
                color_continuous_scale="RdYlGn"
            )
            fig_category.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_category, use_container_width=True)
        
        with col2:
            # Response type performance
            response_type_stats = df_feedback.groupby('response_type').agg({
                'rating': ['mean', 'count']
            }).round(2)
            response_type_stats.columns = ['Avg Rating', 'Count']
            
            st.subheader("Performance by Response Type")
            st.dataframe(response_type_stats, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly activity
            hourly_activity = df_feedback.groupby('hour').size()
            fig_hourly = px.bar(
                x=hourly_activity.index,
                y=hourly_activity.values,
                title="Activity by Hour of Day",
                labels={'x': 'Hour', 'y': 'Number of Interactions'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Daily activity
            daily_activity = df_feedback.groupby('date').size()
            fig_daily = px.line(
                x=daily_activity.index,
                y=daily_activity.values,
                title="Daily Activity Trend",
                labels={'x': 'Date', 'y': 'Number of Interactions'},
                markers=True
            )
            st.plotly_chart(fig_daily, use_container_width=True)
    
    with tab4:
        # Recent comments
        st.subheader("Recent Comments")
        recent_comments = df_feedback[df_feedback['comments'].notna() & (df_feedback['comments'] != '')].tail(10)
        
        if not recent_comments.empty:
            for _, row in recent_comments.iterrows():
                with st.expander(f"⭐{row['rating']} - {row['category']} ({row['timestamp'].strftime('%Y-%m-%d %H:%M')})"):
                    st.write(f"**Question:** {row['question']}")
                    st.write(f"**Comment:** {row['comments']}")
        else:
            st.info("No comments yet. Users can leave comments when rating responses.")
    
    # Detailed data table
    with st.expander("📋 Detailed Feedback Data"):
        st.dataframe(df_feedback.sort_values('timestamp', ascending=False), use_container_width=True)
        
        # Download option
        csv_data = df_feedback.to_csv(index=False)
        st.download_button(
            label="📥 Download Feedback Data",
            data=csv_data,
            file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Sidebar
st.sidebar.title("📚 FAQ Categories")
selected_category = st.sidebar.selectbox("Choose a category", list(faq_categories.keys()))
faq_data = faq_categories[selected_category]
faq_questions = list(faq_data.keys())

st.sidebar.subheader(f"📋 {selected_category} FAQs")
for i, question in enumerate(faq_questions, 1):
    st.sidebar.write(f"{i}. {question}")

st.sidebar.info("🌐 Web search: Enabled for investment topics.")

# Analytics in sidebar
if st.session_state.feedback_log:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quick Stats")
    df_temp = pd.DataFrame(st.session_state.feedback_log)
    avg_rating = df_temp['rating'].mean()
    st.sidebar.metric("Avg Rating", f"{avg_rating:.1f}⭐")
    st.sidebar.metric("Total Feedback", len(df_temp))

# Main UI
st.set_page_config(page_title="Investment FAQ Chatbot", page_icon="💼", layout="wide")
st.title("💼 Investment Process FAQ Chatbot")
st.markdown("*Ask questions about our investment process, evaluation criteria, and more!*")

# Navigation tabs
main_tab, analytics_tab = st.tabs(["💬 Chatbot", "📊 Analytics"])

with main_tab:
    st.header("💬 Ask Your Question")

    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False

    input_key = "user_input"
    if st.session_state.clear_input:
        st.session_state.clear_input = False
        if input_key in st.session_state:
            del st.session_state[input_key]

    user_input = st.text_input("Type your investment-related question here:", key=input_key, placeholder="e.g., What documents do I need for fundraising?")

    col1, col2 = st.columns([1, 1])
    with col1:
        send = st.button("Send 📤", type="primary")
    with col2:
        reset = st.button("Clear History 🗑️")

    if send and user_input.strip():
        try:
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.3)
            best_question, best_answer, category, similarity_score = get_best_faq_response(user_input)

            # Update analytics
            st.session_state.user_analytics["total_questions"] += 1
            st.session_state.user_analytics["question_categories"][category] += 1
            current_hour = datetime.now().hour
            st.session_state.user_analytics["hourly_activity"][current_hour] += 1

            with st.spinner("Processing your question..."):
                if similarity_score >= 0.3:
                    prompt = f"""You are a professional investment advisor assistant.

User Question: \"{user_input}\"
Best Matching FAQ: \"{best_question}\" (Category: {category})
FAQ Answer: \"{best_answer}\"

Provide a comprehensive response based on the FAQ answer. Keep it professional, helpful, and actionable."""

                    response = llm.invoke([
                        SystemMessage(content="You are a knowledgeable investment advisor."),
                        HumanMessage(content=prompt)
                    ])

                    final_response = response.content
                    response_type = f"📋 FAQ Response (Category: {category})"
                    st.session_state.user_analytics["response_types"]["FAQ"] += 1
                else:
                    web_search_result = perform_web_search(user_input)
                    prompt = f"""You are a professional investment advisor assistant.

User Question: \"{user_input}\"
Web Search Results: \"{web_search_result}\"

Use the web search results to provide a helpful response about startup investments, fundraising, or venture capital. Keep it professional and actionable."""
                    response = llm.invoke([
                        SystemMessage(content="You are a knowledgeable investment advisor."),
                        HumanMessage(content=prompt)
                    ])
                    final_response = response.content + f"\n\n---\n🌐 **Enhanced with web search results**"
                    response_type = "🌐 Web-Enhanced Response"
                    st.session_state.user_analytics["response_types"]["Web Search"] += 1

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append((user_input, final_response, timestamp, response_type, category))

            # Log to CSV
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

    # Display chat history with feedback
    if st.session_state.chat_history:
        st.header("📜 Recent Conversations")
        for i, (question, answer, timestamp, resp_type, category) in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {question} ({timestamp})", expanded=(i == 0)):
                st.markdown(f"**{resp_type}**")
                st.markdown(f"**Answer:** {answer}")
                
                # Add feedback collection for each response
                st.markdown("---")
                collect_feedback(question, answer, category, resp_type)

    st.markdown("---")
    st.markdown("💡 **Tip:** Try asking about fundraising documents, evaluation criteria, investment focus, or due diligence process!")

    # Quick FAQ access
    st.markdown("---")
    st.header("📌 Quick FAQ Access")
    fundraising_faqs = faq_categories.get("Fundraising Process", {})
    for question, answer in list(fundraising_faqs.items())[:3]:  # Show top 3
        with st.expander(f"❓ {question}"):
            st.markdown(answer)

with analytics_tab:
    st.header("📊 Feedback Analytics Dashboard")
    display_analytics()
