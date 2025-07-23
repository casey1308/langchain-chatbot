import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import csv
import pandas as pd
import logging
import requests
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

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# Sidebar for category selection
st.sidebar.title("📚 FAQ Categories")
selected_category = st.sidebar.selectbox("Choose a category", list(faq_categories.keys()))
faq_data = faq_categories[selected_category]
faq_questions = list(faq_data.keys())

st.sidebar.subheader(f"📋 {selected_category} FAQs")
for i, question in enumerate(faq_questions, 1):
    st.sidebar.write(f"{i}. {question}")

st.sidebar.info("🌐 Web search: Enabled for investment topics.")

# FAQ matching

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

# Main UI
st.set_page_config(page_title="Investment FAQ Chatbot", page_icon="💼", layout="wide")
st.title("💼 Investment Process FAQ Chatbot")
st.markdown("*Ask questions about our investment process, evaluation criteria, and more!*")

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

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append((user_input, final_response, timestamp, response_type))

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
    st.header("📜 Recent Conversations")
    for i, (question, answer, timestamp, resp_type) in enumerate(reversed(st.session_state.chat_history[-10:])):
        with st.expander(f"Q{len(st.session_state.chat_history)-i}: {question} ({timestamp})", expanded=(i == 0)):
            st.markdown(f"**{resp_type}**")
            st.markdown(f"**Answer:** {answer}")

st.markdown("---")
st.markdown("💡 **Tip:** Try asking about fundraising documents, evaluation criteria, investment focus, or due diligence process!")

# 🔹 Fundraising FAQs Clickable Expanders
st.markdown("---")
st.header("📌 Fundraising FAQs (Click to Expand)")
fundraising_faqs = faq_categories.get("Fundraising Process", {})
for question, answer in fundraising_faqs.items():
    with st.expander(f"❓ {question}"):
        st.markdown(answer)

# Show Feedback UI
st.markdown("---")
st.header("📊 Feedback Analytics")

if st.session_state.feedback_log:
    df_feedback = pd.DataFrame(st.session_state.feedback_log, columns=["timestamp", "question", "response", "rating"])
    st.dataframe(df_feedback)

    st.subheader("Average Feedback Score")
    avg_rating = df_feedback['rating'].mean()
    st.metric(label="Avg. Rating", value=f"{avg_rating:.2f} / 5")

    st.subheader("Most Asked Questions")
    most_asked = df_feedback['question'].value_counts().head(5)
    st.bar_chart(most_asked)
else:
    st.info("No feedback yet. Engage with the chatbot and rate responses.")
