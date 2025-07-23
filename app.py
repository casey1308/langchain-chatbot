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

# Initialize SerpAPI with error handling
try:
    serp_tool = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    web_search_available = True
except Exception as e:
    logger.warning(f"SerpAPI initialization failed: {e}")
    web_search_available = False
    st.warning("⚠️ Web search is currently unavailable. Only FAQ responses will be provided.")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Enhanced FAQ Categories with more comprehensive content
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

# Display current category FAQs in sidebar
st.sidebar.subheader(f"📋 {selected_category} FAQs")
for i, question in enumerate(faq_questions, 1):
    st.sidebar.write(f"{i}. {question}")

# Web search availability indicator
if web_search_available:
    st.sidebar.success("🌐 Web search: Available")
else:
    st.sidebar.error("🌐 Web search: Unavailable")

# Enhanced FAQ matching function
def get_best_faq_response(user_input):
    # Create a comprehensive question list from all categories
    all_questions = []
    all_answers = []
    question_categories = []
    
    for category, qa_dict in faq_categories.items():
        for question, answer in qa_dict.items():
            all_questions.append(question)
            all_answers.append(answer)
            question_categories.append(category)
    
    # Use TF-IDF to find best match
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([user_input] + all_questions)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    
    top_index = similarities.argmax()
    top_score = similarities[top_index]
    
    return (all_questions[top_index], 
            all_answers[top_index], 
            question_categories[top_index],
            top_score)

# Web search function with better formatting
def perform_web_search(query):
    try:
        if not web_search_available:
            return "Web search is currently unavailable."
        
        search_results = serp_tool.run(f"{query} startup investment venture capital")
        
        # Clean and format the search results
        if len(search_results) > 500:
            search_results = search_results[:500] + "..."
        
        return search_results
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return "Unable to perform web search at this time."

# Main app interface
st.set_page_config(page_title="Investment FAQ Chatbot", page_icon="💼", layout="wide")
st.title("💼 Investment Process FAQ Chatbot")
st.markdown("*Ask questions about our investment process, evaluation criteria, and more!*")

# Main input section
st.header("💬 Ask Your Question")

# Input handling
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

input_key = "user_input"
if st.session_state.clear_input:
    st.session_state.clear_input = False
    if input_key in st.session_state:
        del st.session_state[input_key]

user_input = st.text_input("Type your investment-related question here:", 
                          key=input_key,
                          placeholder="e.g., What documents do I need for fundraising?")

# Buttons
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    send = st.button("Send 📤", type="primary")
with col2:
    reset = st.button("Clear History 🗑️")

if send and user_input.strip():
    try:
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.3)
        
        # Get best FAQ match
        best_question, best_answer, category, similarity_score = get_best_faq_response(user_input)
        
        with st.spinner("Processing your question..."):
            if similarity_score >= 0.3:  # Good FAQ match
                prompt = f"""You are a professional investment advisor assistant. 
                
User Question: "{user_input}"
Best Matching FAQ: "{best_question}" (Category: {category})
FAQ Answer: "{best_answer}"

Provide a comprehensive response based on the FAQ answer. If the user's question has additional nuances not covered in the FAQ, acknowledge them and provide additional relevant insights about startup investments and fundraising.

Keep the response professional, helpful, and actionable."""

                response = llm.invoke([
                    SystemMessage(content="You are a knowledgeable investment advisor. Provide clear, actionable advice."),
                    HumanMessage(content=prompt)
                ])
                
                final_response = response.content
                response_type = f"📋 FAQ Response (Category: {category})"
                
            else:  # Poor FAQ match - use web search
                web_search_result = perform_web_search(user_input)
                
                prompt = f"""You are a professional investment advisor assistant.

User Question: "{user_input}"
Closest FAQ: "{best_question}" (but similarity is low)
Web Search Results: "{web_search_result}"

The user's question doesn't closely match our FAQ database. Use the web search results to provide a helpful response about startup investments, fundraising, or venture capital. If the web search results are not relevant, provide general guidance based on your knowledge of investment processes.

Keep the response professional and actionable."""

                response = llm.invoke([
                    SystemMessage(content="You are a knowledgeable investment advisor. Use web search results to provide helpful insights."),
                    HumanMessage(content=prompt)
                ])
                
                final_response = response.content
                if web_search_available:
                    final_response += f"\n\n---\n🌐 **Web Search Context:** This response incorporates recent information from web search."
                
                response_type = "🌐 Web-Enhanced Response"

        # Add to chat history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append((user_input, final_response, timestamp, response_type))

        # Log to CSV
        try:
            with open("chat_log.csv", mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, user_input, final_response, response_type, similarity_score])
        except Exception as e:
            logger.warning(f"Failed to log to CSV: {e}")

        # Clear input and refresh
        st.session_state.clear_input = True
        st.rerun()

    except OpenAIError as e:
        st.error(f"❌ OpenAI API Error: {str(e)}")
        logger.error(f"OpenAI error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        logger.error(f"Unexpected error: {e}")

# Clear history
if reset:
    st.session_state.chat_history = []
    st.success("Chat history cleared!")
    st.rerun()

# Display chat history
if st.session_state.chat_history:
    st.header("📜 Recent Conversations")
    
    # Show last 10 conversations
    for i, (question, answer, timestamp, resp_type) in enumerate(reversed(st.session_state.chat_history[-10:])):
        with st.expander(f"Q{len(st.session_state.chat_history)-i}: {question} ({timestamp})", 
                         expanded=(i == 0)):
            st.markdown(f"**{resp_type}**")
            st.markdown(f"**Answer:** {answer}")
            st.markdown(f"*Asked at: {timestamp}*")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Try asking about fundraising documents, evaluation criteria, investment focus, or due diligence process!")
