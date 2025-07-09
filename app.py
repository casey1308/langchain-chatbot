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
        Analyze this pitch deck text and extract the following information in key-value format.
        Look through ALL the text carefully and extract ANY information that might be relevant.
        Don't just look for obvious sections - information might be scattered throughout.
        
        Please provide the extracted data in this exact key-value format:
        
        company_name: [Company Name or Not mentioned]
        description: [Brief company description]
        industry: [Industry/Sector]
        stage: [Seed/Series A/etc or Not mentioned]
        location: [Location if mentioned or Not mentioned]
        
        founder_1_name: [Full Name or Not mentioned]
        founder_1_role: [CEO/CTO/etc or Not mentioned]
        founder_1_background: [Background/Experience or Not mentioned]
        founder_2_name: [Full Name or Not mentioned]
        founder_2_role: [CEO/CTO/etc or Not mentioned]
        founder_2_background: [Background/Experience or Not mentioned]
        team_size: [Number if mentioned or Not mentioned]
        
        market_size: [Market size with specific numbers or Not mentioned]
        problem: [Problem being solved or Not mentioned]
        solution: [Solution provided or Not mentioned]
        target_market: [Target market description or Not mentioned]
        
        product_description: [Product/service description or Not mentioned]
        key_features: [Key features or Not mentioned]
        technology_stack: [Technology if mentioned or Not mentioned]
        differentiators: [What makes it unique or Not mentioned]
        
        revenue: [Current revenue or Not mentioned]
        customers: [Customer count/info or Not mentioned]
        growth: [Growth metrics or Not mentioned]
        users: [Active users if mentioned or Not mentioned]
        partnerships: [Key partnerships or Not mentioned]
        
        ask: [Amount seeking or Not mentioned]
        valuation: [Current valuation or Not mentioned]
        funding: [Previous funding info or Not mentioned]
        use_of_funds: [How funds will be used or Not mentioned]
        revenue_model: [How they make money or Not mentioned]
        pricing: [Pricing strategy if mentioned or Not mentioned]
        
        competition: [Competitive landscape info if mentioned or Not mentioned]
        go_to_market: [Marketing/sales strategy or Not mentioned]
        timeline: [Key milestones or timeline if mentioned or Not mentioned]
        
        IMPORTANT: 
        1. Look for information in ALL parts of the text, not just obvious sections
        2. Extract specific numbers, percentages, and concrete details whenever possible
        3. If you find partial information, include it rather than saying "Not mentioned"
        4. Keep each value on a single line
        5. Use the exact key format shown above
        
        Text to analyze:
        """
        
        messages = [
            SystemMessage(content="You are an expert at extracting structured data from pitch decks. Be thorough and look for information throughout the entire document. Extract specific numbers and details whenever possible. Format the output EXACTLY as requested with key-value pairs."),
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
        
        # Parse and display key-value pairs in expandable format
        with st.expander("View Key-Value Data", expanded=False):
            if st.session_state.structured_data:
                lines = st.session_state.structured_data.split('\n')
                for line in lines:
                    line = line.strip()
                    if ':' in line and line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        if value and value != "Not mentioned":
                            st.write(f"**{key}:** {value}")
        
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
    
    # Display as formatted key-value pairs
    if st.session_state.structured_data:
        lines = st.session_state.structured_data.split('\n')
        
        # Group by categories
        current_category = "General"
        categories = {
            "company": "🏢 Company Information",
            "founder": "👥 Founders & Team", 
            "market": "📊 Market & Problem",
            "product": "🚀 Product & Technology",
            "revenue": "💰 Traction & Metrics",
            "customers": "💰 Traction & Metrics",
            "growth": "💰 Traction & Metrics",
            "users": "💰 Traction & Metrics",
            "partnerships": "💰 Traction & Metrics",
            "ask": "💵 Funding & Financials",
            "valuation": "💵 Funding & Financials",
            "funding": "💵 Funding & Financials",
            "use_of_funds": "💵 Funding & Financials",
            "revenue_model": "💵 Funding & Financials",
            "pricing": "💵 Funding & Financials",
            "competition": "⚔️ Competition & Strategy",
            "go_to_market": "⚔️ Competition & Strategy",
            "timeline": "⚔️ Competition & Strategy"
        }
        
        grouped_data = {}
        for line in lines:
            line = line.strip()
            if ':' in line and line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                # Find category
                category = "🔍 Other Information"
                for cat_key, cat_name in categories.items():
                    if cat_key in key:
                        category = cat_name
                        break
                
                if category not in grouped_data:
                    grouped_data[category] = []
                
                if value and value != "Not mentioned":
                    grouped_data[category].append((key.replace('_', ' ').title(), value))
        
        # Display grouped data
        for category, items in grouped_data.items():
            if items:
                st.subheader(category)
                for key, value in items:
                    st.write(f"**{key}:** {value}")
                st.markdown("---")
    
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
st.markdown("- `Search for founder's LinkedIn profile and experience`")
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
                    # Extract founder names and company name from structured data for better search
                    founder_names = []
                    company_name = ""
                    
                    if st.session_state.structured_data:
                        try:
                            # Parse the key-value structured data
                            lines = st.session_state.structured_data.split('\n')
                            for line in lines:
                                line = line.strip()
                                if ':' in line:
                                    key, value = line.split(':', 1)
                                    key = key.strip().lower()
                                    value = value.strip()
                                    
                                    if 'founder' in key and 'name' in key and value and value != "Not mentioned":
                                        founder_names.append(value)
                                    elif key == 'company_name' and value and value != "Not mentioned":
                                        company_name = value
                        except:
                            # Fallback extraction
                            if "Himanshu Gupta" in st.session_state.structured_data:
                                founder_names.append("Himanshu Gupta")
                    
                    # Create better search query with company context
                    if founder_names and matched_key in ["founder", "team"]:
                        if company_name:
                            # Search with founder name + company name for more specific results
                            search_query = f'"{founder_names[0]}" "{company_name}" founder CEO linkedin background education'
                        else:
                            # Search with just founder name + founder context
                            search_query = f'"{founder_names[0]}" founder linkedin background education experience'
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
