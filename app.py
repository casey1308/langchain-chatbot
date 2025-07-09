import streamlit as st
import os
import re
import requests
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime
from difflib import get_close_matches
import PyPDF2
import logging
from openai import OpenAIError, RateLimitError, APIError

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key or not serpapi_key:
    st.error("❌ Add your OPENAI_API_KEY and SERPAPI_API_KEY in .env")
    st.stop()

# Initialize session state
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections", "structured_data"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "chat_history" else (False if key == "file_uploaded" else None)

# Enhanced text cleaner
def clean_text(text):
    # Remove hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Remove extra whitespace and newlines
    text = re.sub(r"\n{2,}", "\n", text)
    # Remove bullet points and formatting
    text = re.sub(r"[•◦▪▫‣⁃]", "", text)
    # Clean up spacing
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Enhanced PDF extraction with better text processing
def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    full_text = ""
    
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        # Add page break markers for better section detection
        full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
    
    return clean_text(full_text)

# Enhanced section splitting with better pattern matching
def split_sections(text):
    sections = {"General": []}
    current_section = "General"
    
    # More comprehensive heading patterns
    heading_patterns = [
        # Founder/Team patterns
        r"(?i)(founder|co-founder|team|leadership|management|ceo|cto|cfo|about\s+us)",
        # Business patterns
        r"(?i)(problem|solution|market|business\s+model|revenue|traction|competition)",
        # Financial patterns
        r"(?i)(financials|funding|valuation|ask|investment|series|round|cap\s+table)",
        # Product patterns
        r"(?i)(product|technology|demo|features|roadmap|development)",
        # Market patterns
        r"(?i)(market\s+size|addressable\s+market|tam|sam|som|opportunity)",
        # Strategy patterns
        r"(?i)(go-to-market|marketing|sales|strategy|growth|expansion)"
    ]
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a heading
        is_heading = False
        for pattern in heading_patterns:
            if re.match(pattern, line) and len(line) < 100:  # Likely a heading
                current_section = line
                sections[current_section] = []
                is_heading = True
                break
        
        if not is_heading:
            sections.setdefault(current_section, []).append(line)
    
    return {k: "\n".join(v) for k, v in sections.items() if v}

# Enhanced data extraction using LLM
def extract_structured_data(text):
    """Extract structured data from pitch deck using LLM"""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)
        
        extraction_prompt = """
        Analyze this pitch deck text and extract the following information in JSON format:
        
        {
            "founders": [
                {
                    "name": "Full Name",
                    "role": "CEO/CTO/etc",
                    "background": "Brief background",
                    "experience": "Relevant experience"
                }
            ],
            "company": {
                "name": "Company Name",
                "description": "Brief description",
                "industry": "Industry/Sector",
                "stage": "Seed/Series A/etc"
            },
            "funding": {
                "ask_amount": "Amount seeking",
                "valuation": "Current valuation",
                "use_of_funds": "How funds will be used",
                "previous_rounds": "Previous funding info"
            },
            "market": {
                "size": "Market size",
                "problem": "Problem being solved",
                "solution": "Solution provided"
            },
            "traction": {
                "revenue": "Current revenue",
                "customers": "Customer count/info",
                "growth": "Growth metrics"
            }
        }
        
        Extract only information that is explicitly mentioned. Use "Not mentioned" if information is not found.
        
        Text to analyze:
        """
        
        messages = [
            SystemMessage(content="You are an expert at extracting structured data from pitch decks. Be precise and only extract explicitly mentioned information."),
            HumanMessage(content=f"{extraction_prompt}\n\n{text[:4000]}")  # Limit text to avoid token limits
        ]
        
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        logger.error(f"Error extracting structured data: {e}")
        return "Error extracting structured data"

# Enhanced section matching
def match_section(key, sections, structured_data=None):
    key = key.lower()
    
    # First check structured data if available
    if structured_data and "founders" in key:
        return structured_data
    
    # Enhanced lookup patterns
    lookup = {
        "founder": ["founder", "co-founder", "team", "ceo", "leadership", "management", "about us"],
        "valuation": ["valuation", "valuation cap", "pre-money", "post-money", "worth"],
        "ask": ["ask", "funding required", "investment", "capital", "raise", "seeking"],
        "round": ["round", "series", "pre-seed", "seed", "series a", "series b", "funding round"],
        "investor": ["investor", "cap table", "backer", "vc", "existing investor", "previous investors"],
        "team": ["team", "leadership", "core team", "management", "about us"],
        "market": ["market", "market size", "tam", "sam", "som", "opportunity"],
        "problem": ["problem", "pain point", "challenge", "issue"],
        "solution": ["solution", "product", "technology", "platform"],
        "traction": ["traction", "revenue", "customers", "growth", "metrics"],
        "competition": ["competition", "competitors", "competitive", "landscape"],
        "business_model": ["business model", "revenue model", "monetization"]
    }
    
    # Find matching sections
    best_match = None
    best_score = 0
    
    if key in lookup:
        for tag in lookup[key]:
            for section_key, section_content in sections.items():
                if tag in section_key.lower():
                    return section_content
                # Also check content for keywords
                if tag in section_content.lower():
                    score = section_content.lower().count(tag)
                    if score > best_score:
                        best_match = section_content
                        best_score = score
    
    if best_match:
        return best_match
    
    # Fuzzy matching as fallback
    matches = get_close_matches(key, [k.lower() for k in sections.keys()], n=1, cutoff=0.3)
    if matches:
        for k in sections:
            if k.lower() == matches[0]:
                return sections[k]
    
    return "Not mentioned in deck."

# Google SERP API fallback
def search_serpapi(query):
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
            "num": 3
        }
        r = requests.get("https://serpapi.com/search", params=params)
        if r.status_code == 200:
            data = r.json()
            if "organic_results" not in data:
                return "No search results found."
            combined = ""
            for res in data["organic_results"]:
                title = res.get("title", "")
                snippet = res.get("snippet", "")
                combined += f"{title}\n{snippet}\n\n"
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
            return llm.invoke(f"Summarize for a VC:\n{combined}").content.strip()
        return f"❌ SERP API error: {r.status_code}"
    except Exception as e:
        return f"❌ Web search failed: {str(e)}"

# UI layout
st.set_page_config(page_title="Manna — VC Pitch Evaluator", page_icon="📊")
st.title("📊 Manna — VC Pitch Evaluator")

# Sidebar content
with st.sidebar:
    st.header("📋 Detected Sections")
    if st.session_state.sections:
        for section_name in st.session_state.sections.keys():
            if st.button(section_name, key=f"section_{section_name}"):
                st.session_state.selected_section = section_name
    
    # Show quick stats
    if st.session_state.file_uploaded:
        st.header("📊 Quick Stats")
        st.write(f"Total sections: {len(st.session_state.sections)}")
        st.write(f"Total text length: {len(st.session_state.parsed_doc)} chars")
    
    # Show structured data
    if st.session_state.structured_data:
        st.header("🔍 Extracted Data")
        with st.expander("View Structured Data"):
            st.text_area("Structured Analysis", st.session_state.structured_data, height=300, key="structured_data_sidebar")
    
    # Chat History in Sidebar
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        
        # Add clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Display chat history in reverse order (newest first)
        for i, (user, bot, timestamp) in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {user[:30]}..."):
                st.markdown(f"**🧑 You:** {user}")
                st.markdown(f"**🤖 Manna:**")
                st.markdown(bot)
                st.markdown(f"*{timestamp}*")

# Main content area
# Upload
file = st.file_uploader("📄 Upload a startup pitch deck (PDF)", type=["pdf"])

if file:
    with st.spinner("📄 Parsing pitch deck..."):
        file_bytes = file.read()
        text = extract_pdf_text(file_bytes)
        st.session_state.parsed_doc = text
        st.session_state.sections = split_sections(text)
        st.session_state.file_uploaded = True
        
        # Extract structured data
        with st.spinner("🔍 Extracting structured data..."):
            st.session_state.structured_data = extract_structured_data(text)
        
    st.success("✅ Pitch deck parsed and analyzed!")

# Show selected section
if hasattr(st.session_state, 'selected_section') and st.session_state.selected_section:
    st.subheader(f"📖 {st.session_state.selected_section}")
    st.text_area("Content", st.session_state.sections[st.session_state.selected_section], height=200)

# Prompt input
user_query = st.chat_input("💬 Ask about founders, funding, valuation, team, etc.")

if user_query:
    with st.spinner("🤖 Thinking..."):
        try:
            context = st.session_state.parsed_doc or ""
            llm = ChatOpenAI(
                model="gpt-4o", 
                openai_api_key=openai_api_key, 
                streaming=True,
                temperature=0.2,  # Lower temperature for more factual responses
                max_tokens=1500
            )
            
            lower_q = user_query.lower()

            # Enhanced intent matching
            intent_keys = {
                "founder": ["founder", "who is the founder", "co-founder", "ceo", "team leader", "management"],
                "team": ["team", "leadership", "management", "staff", "employees"],
                "valuation": ["valuation", "company worth", "value", "pre-money", "post-money"],
                "ask": ["ask", "funding ask", "how much funding", "raise", "investment needed"],
                "round": ["previous round", "funding round", "series a", "seed", "pre-seed"],
                "investor": ["investors", "vc", "backers", "existing investors", "previous investors"],
                "market": ["market", "market size", "tam", "opportunity"],
                "problem": ["problem", "pain point", "challenge"],
                "solution": ["solution", "product", "technology"],
                "traction": ["traction", "revenue", "customers", "growth"],
                "competition": ["competition", "competitors", "competitive"]
            }

            matched_key = next((k for k, v in intent_keys.items() if any(q in lower_q for q in v)), None)

            if matched_key:
                section_text = match_section(matched_key, st.session_state.sections, st.session_state.structured_data)
                if section_text == "Not mentioned in deck.":
                    web_result = search_serpapi(user_query)
                    context_msg = f"Deck Analysis: {section_text}\n\nWeb Search Results:\n{web_result}"
                else:
                    # Include structured data context
                    context_msg = f"Deck Section ({matched_key}):\n{section_text}\n\nStructured Data:\n{st.session_state.structured_data[:1000]}"
            elif any(x in lower_q for x in ["lawsuit", "legal", "controversy", "reputation", "news"]):
                web_result = search_serpapi(user_query)
                context_msg = f"Deck Context:\n{context[:1000]}\n\nWeb Research:\n{web_result}"
            else:
                context_msg = f"Full Deck Analysis:\n{context[:2500]}\n\nStructured Data:\n{st.session_state.structured_data[:1000]}"

            # Enhanced system message
            system_msg = """You are an expert VC analyst AI. Analyze pitch decks thoroughly and provide detailed, actionable insights.

            Guidelines:
            1. Be specific and cite exact information from the deck
            2. If information is missing, clearly state what's not mentioned
            3. For founders: Look for names, roles, backgrounds, experience, LinkedIn profiles
            4. For financials: Look for specific numbers, percentages, timelines
            5. Provide VC-relevant insights and red flags
            6. If web search was used, distinguish between deck info and external research
            """

            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=f"{context_msg}\n\nVC Question: {user_query}")
            ]

            # Chat output
            st.markdown(f"**🧑 You:** {user_query}")
            st.markdown("**🤖 Manna:**")

            def generate_response():
                for chunk in llm.stream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content

            full_output = st.write_stream(generate_response())

        except RateLimitError:
            st.error("❌ Rate limit exceeded. Please try again in a minute.")
            full_output = "Rate limit exceeded."
        except APIError as e:
            st.error(f"❌ OpenAI API error: {str(e)}")
            full_output = f"API error: {str(e)}"
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Streaming error: {str(e)}", exc_info=True)
            
            # Fallback to non-streaming
            try:
                response = llm.invoke(messages)
                full_output = response.content
                st.markdown(full_output)
            except Exception as fallback_e:
                st.error(f"❌ Fallback also failed: {str(fallback_e)}")
                full_output = "Error generating response."

        st.session_state.chat_history.append(
            (user_query, full_output.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        
        # Rerun to update sidebar
        st.rerun()
