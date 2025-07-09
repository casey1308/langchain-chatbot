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
import json
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
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections", "structured_data", "selected_chat_index"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        elif key == "file_uploaded":
            st.session_state[key] = False
        elif key == "selected_chat_index":
            st.session_state[key] = None
        else:
            st.session_state[key] = None

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
        Analyze this pitch deck text and extract the following information in a readable format.
        Look through ALL the text carefully and extract ANY information that might be relevant.
        Don't just look for obvious sections - information might be scattered throughout.
        
        Please provide a structured analysis in the following format:
        
        COMPANY OVERVIEW:
        • Company Name: [Name or Not mentioned]
        • Description: [Brief description]
        • Industry: [Industry/Sector]
        • Stage: [Seed/Series A/etc or Not mentioned]
        • Location: [Location if mentioned or Not mentioned]
        
        FOUNDERS & TEAM:
        • Founder 1: [Name] - [Role] - [Background/Experience]
        • Founder 2: [Name] - [Role] - [Background/Experience]
        • Team Size: [Number if mentioned or Not mentioned]
        
        MARKET & PROBLEM:
        • Market Size: [Size with specific numbers]
        • Problem: [Problem being solved]
        • Solution: [Solution provided]
        • Target Market: [Target market description]
        
        PRODUCT & TECHNOLOGY:
        • Product Description: [Product/service description]
        • Key Features: [Key features]
        • Technology Stack: [Technology if mentioned]
        • Differentiators: [What makes it unique]
        
        TRACTION & METRICS:
        • Revenue: [Current revenue or Not mentioned]
        • Customers: [Customer count/info]
        • Growth: [Growth metrics]
        • Users: [Active users if mentioned]
        • Partnerships: [Key partnerships or Not mentioned]
        
        FUNDING & FINANCIALS:
        • Funding Ask: [Amount seeking or Not mentioned]
        • Valuation: [Current valuation or Not mentioned]
        • Use of Funds: [How funds will be used or Not mentioned]
        • Previous Rounds: [Previous funding info or Not mentioned]
        • Revenue Model: [How they make money]
        • Pricing: [Pricing strategy if mentioned]
        
        COMPETITION & STRATEGY:
        • Competition: [Competitive landscape info if mentioned]
        • Go-to-Market: [Marketing/sales strategy]
        • Timeline: [Key milestones or timeline if mentioned]
        
        IMPORTANT: Look for information in ALL parts of the text, not just obvious sections. 
        Extract specific numbers, percentages, and concrete details whenever possible.
        If you find partial information, include it rather than saying "Not mentioned".
        
        Text to analyze:
        """
        
        messages = [
            SystemMessage(content="You are an expert at extracting structured data from pitch decks. Be thorough and look for information throughout the entire document, not just in obvious sections. Extract specific numbers and details whenever possible. Format the output in a clear, readable way with bullet points and sections."),
            HumanMessage(content=f"{extraction_prompt}\n\n{text[:6000]}")
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
            "num": 5  # Increased from 3 to get more results
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
                link = res.get("link", "")
                combined += f"Title: {title}\nSnippet: {snippet}\nSource: {link}\n\n"
            
            # Also check for people_also_ask results for founder queries
            if "people_also_ask" in data:
                combined += "\n--- Related Questions ---\n"
                for paa in data["people_also_ask"][:3]:  # Top 3 related questions
                    combined += f"Q: {paa.get('question', '')}\nA: {paa.get('snippet', '')}\n\n"
            
            # Check for knowledge graph results (good for person info)
            if "knowledge_graph" in data:
                kg = data["knowledge_graph"]
                combined += "\n--- Knowledge Graph ---\n"
                if "title" in kg:
                    combined += f"Title: {kg['title']}\n"
                if "description" in kg:
                    combined += f"Description: {kg['description']}\n"
                if "attributes" in kg:
                    for attr_name, attr_value in kg["attributes"].items():
                        combined += f"{attr_name}: {attr_value}\n"
                combined += "\n"
            
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
            
            # Enhanced prompt for founder/background searches
            if any(x in query.lower() for x in ["founder", "background", "education", "experience", "linkedin", "profile"]):
                search_prompt = f"""Analyze these search results and provide a comprehensive summary focusing on:
                1. Educational background (universities, degrees, graduation years)
                2. Professional experience (previous companies, roles, duration)
                3. Notable achievements or recognition
                4. LinkedIn profile information if available
                5. Any relevant industry expertise
                6. Entrepreneurial history
                
                Be specific about dates, companies, and roles. If information is limited, mention what's available.
                
                Search results:\n{combined}"""
            else:
                search_prompt = f"Summarize these search results for a VC investor:\n{combined}"
            
            return llm.invoke(search_prompt).content.strip()
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
    
    # Show structured data with expandable view
    if st.session_state.structured_data:
        st.header("🔍 Extracted Data")
        with st.expander("View Structured Data", expanded=False):
            st.markdown(st.session_state.structured_data)
        
        # Button to view full structured data
        if st.button("📄 View Full Structured Analysis"):
            st.session_state.show_structured_data = True
    
    # Chat History in Sidebar
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        
        # Add clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.selected_chat_index = None
            st.rerun()
        
        # Display chat history in reverse order (newest first)
        for i, (user, bot, timestamp) in enumerate(reversed(st.session_state.chat_history)):
            chat_index = len(st.session_state.chat_history) - 1 - i
            if st.button(f"Q{chat_index + 1}: {user[:30]}...", key=f"chat_{chat_index}"):
                st.session_state.selected_chat_index = chat_index
                st.rerun()

# Main content area

# Show full structured data if requested
if hasattr(st.session_state, 'show_structured_data') and st.session_state.show_structured_data:
    st.header("🔍 Complete Structured Analysis")
    st.markdown(st.session_state.structured_data)
    if st.button("❌ Close"):
        st.session_state.show_structured_data = False
        st.rerun()
    st.markdown("---")

# Show selected chat from history
if st.session_state.selected_chat_index is not None:
    chat_data = st.session_state.chat_history[st.session_state.selected_chat_index]
    user_q, bot_response, timestamp = chat_data
    
    st.header(f"💬 Chat #{st.session_state.selected_chat_index + 1}")
    st.markdown(f"**🧑 You:** {user_q}")
    st.markdown(f"**🤖 Manna:**")
    st.markdown(bot_response)
    st.markdown(f"*{timestamp}*")
    
    if st.button("❌ Close Chat View"):
        st.session_state.selected_chat_index = None
        st.rerun()
    
    st.markdown("---")

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
st.markdown("### 💬 Ask Questions")
st.markdown("**Example queries:**")
st.markdown("- `Tell me about the founders' background and education`")
st.markdown("- `Search for Himanshu Gupta's LinkedIn profile and experience`")
st.markdown("- `What is the founder's educational background?`")
st.markdown("- `Evaluate the complete pitch deck`")
st.markdown("- `What are the funding details?`")

user_query = st.chat_input("💬 Ask about founders, funding, valuation, team, etc.")

if user_query:
    # Clear any selected chat when asking new question
    st.session_state.selected_chat_index = None
    
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

            # Check for evaluation/analysis keywords
            evaluation_keywords = ["evaluate", "analysis", "analyze", "review", "assessment", "overall", "summary", "complete analysis"]
            is_evaluation = any(keyword in lower_q for keyword in evaluation_keywords)

            matched_key = next((k for k, v in intent_keys.items() if any(q in lower_q for q in v)), None)

            if is_evaluation:
                # For evaluation queries, provide comprehensive context with all sections
                all_sections_text = ""
                for section_name, section_content in st.session_state.sections.items():
                    all_sections_text += f"\n=== {section_name.upper()} ===\n{section_content}\n"
                
                context_msg = f"FULL PITCH DECK CONTENT:\n{all_sections_text}\n\nSTRUCTURED DATA ANALYSIS:\n{st.session_state.structured_data}\n\nORIGINAL DOCUMENT:\n{context[:1000]}"
            elif matched_key:
                section_text = match_section(matched_key, st.session_state.sections, st.session_state.structured_data)
                
                # Check if we should search for additional info
                should_search = (
                    section_text == "Not mentioned in deck." or
                    (matched_key in ["founder", "team"] and any(x in lower_q for x in ["background", "education", "experience", "linkedin", "profile", "bio", "career", "history"])) or
                    any(x in lower_q for x in ["search", "find", "look up", "research", "web search", "google"])
                )
                
                if should_search:
                    # Extract founder names from structured data for better search
                    founder_names = []
                    if st.session_state.structured_data:
                        try:
                            # Try to extract names from the structured text
                            lines = st.session_state.structured_data.split('\n')
                            for line in lines:
                                if 'Founder' in line and ':' in line:
                                    # Extract name from format like "• Founder 1: John Doe - CEO - Background"
                                    parts = line.split(':')[1].split('-')
                                    if parts[0].strip() and parts[0].strip() != "Not mentioned":
                                        founder_names.append(parts[0].strip())
                        except:
                            # If extraction fails, try to find common names
                            if "Himanshu Gupta" in st.session_state.structured_data:
                                founder_names.append("Himanshu Gupta")
                    
                    # Create better search query
                    if founder_names and matched_key in ["founder", "team"]:
                        search_query = f"{' '.join(founder_names)} founder background education experience linkedin"
                    else:
                        search_query = user_query
                    
                    web_result = search_serpapi(search_query)
                    context_msg = f"Deck Analysis: {section_text}\n\nWeb Search Results for '{search_query}':\n{web_result}"
                else:
                    # Include structured data context
                    context_msg = f"Deck Section ({matched_key}):\n{section_text}\n\nStructured Data:\n{st.session_state.structured_data[:1000]}"
            elif any(x in lower_q for x in ["lawsuit", "legal", "controversy", "reputation", "news", "background", "education", "linkedin", "profile", "search", "find", "look up", "research", "web search", "google"]):
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
            7. When doing a full evaluation/analysis, systematically go through each section provided
            8. Use the structured data analysis to cross-reference information
            9. Look for information across ALL sections, not just the most obvious ones
            10. If doing a comprehensive analysis, organize findings by: Company, Founders, Market, Product/Solution, Traction, Financials, and Investment Details
            """

            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=f"{context_msg}\n\nVC Question: {user_query}")
            ]

            # Display the conversation
            st.markdown(f"**🧑 You:** {user_query}")
            st.markdown("**🤖 Manna:**")

            # Create a placeholder for the streaming response
            response_placeholder = st.empty()
            full_response = ""

            # Stream the response
            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")
            
            # Remove the cursor and show final response
            response_placeholder.markdown(full_response)
            
            # Store in chat history
            st.session_state.chat_history.append(
                (user_query, full_response.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )

        except RateLimitError:
            st.error("❌ Rate limit exceeded. Please try again in a minute.")
            full_response = "Rate limit exceeded."
        except APIError as e:
            st.error(f"❌ OpenAI API error: {str(e)}")
            full_response = f"API error: {str(e)}"
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Streaming error: {str(e)}", exc_info=True)
            
            # Fallback to non-streaming
            try:
                response = llm.invoke(messages)
                full_response = response.content
                st.markdown(full_response)
                
                # Store in chat history
                st.session_state.chat_history.append(
                    (user_query, full_response.strip(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
            except Exception as fallback_e:
                st.error(f"❌ Fallback also failed: {str(fallback_e)}")
                full_response = "Error generating response."
