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
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections", "structured_data", "selected_chat_index", "crm_data"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        elif key == "file_uploaded":
            st.session_state[key] = False
        elif key == "selected_chat_index":
            st.session_state[key] = None
        else:
            st.session_state[key] = None

# Enhanced text cleaner with better parsing
def clean_text(text):
    """Advanced text cleaning for better parsing"""
    # Remove hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    
    # Fix common PDF artifacts
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # Add space between lowercase and uppercase
    text = re.sub(r"(\d+)([A-Za-z])", r"\1 \2", text)  # Add space between numbers and letters
    text = re.sub(r"([A-Za-z])(\d+)", r"\1 \2", text)  # Add space between letters and numbers
    
    # Clean up formatting
    text = re.sub(r"\n{3,}", "\n\n", text)  # Reduce multiple newlines
    text = re.sub(r"[•◦▪▫‣⁃]", "", text)  # Remove bullet points
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    
    # Fix common OCR errors
    text = re.sub(r"\b([A-Z]{2,})\b", lambda m: m.group(1).title(), text)  # Fix ALL CAPS
    text = re.sub(r"\s+([.,:;!?])", r"\1", text)  # Fix spacing before punctuation
    
    return text.strip()

# Enhanced PDF extraction with better text processing
def extract_pdf_text(file_bytes):
    """Extract text from PDF with advanced processing"""
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        full_text = ""
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():  # Only add non-empty pages
                    full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1}: {e}")
                continue
        
        cleaned_text = clean_text(full_text)
        logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF")
        return cleaned_text
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

# Enhanced section splitting with better pattern matching
def split_sections(text):
    """Split text into logical sections with improved detection"""
    sections = {"Executive Summary": []}
    current_section = "Executive Summary"
    
    # Comprehensive heading patterns
    heading_patterns = [
        # Company & Team
        r"(?i)^(about\s+(?:us|the\s+company)|company\s+overview|our\s+story|executive\s+summary)",
        r"(?i)^(founder|co-founder|founding\s+team|leadership|management\s+team|team|our\s+team)",
        
        # Problem & Solution
        r"(?i)^(problem|pain\s+point|market\s+problem|the\s+problem|challenge)",
        r"(?i)^(solution|our\s+solution|product|technology|platform|approach)",
        
        # Market & Business
        r"(?i)^(market|market\s+size|market\s+opportunity|addressable\s+market|tam|sam|som)",
        r"(?i)^(business\s+model|revenue\s+model|monetization|how\s+we\s+make\s+money)",
        r"(?i)^(competition|competitive\s+landscape|competitors|market\s+analysis)",
        
        # Traction & Growth
        r"(?i)^(traction|growth|metrics|key\s+metrics|performance|milestones|achievements)",
        r"(?i)^(customers|user\s+base|client\s+base|testimonials|case\s+studies)",
        
        # Financial & Funding
        r"(?i)^(financials|financial\s+projections|revenue|sales|unit\s+economics)",
        r"(?i)^(funding|investment|ask|series|round|capital|valuation|use\s+of\s+funds)",
        r"(?i)^(cap\s+table|equity|ownership|investor\s+relations)",
        
        # Strategy & Future
        r"(?i)^(roadmap|future\s+plans|strategy|vision|goals|objectives)",
        r"(?i)^(go[\s-]?to[\s-]?market|marketing|sales\s+strategy|distribution)",
        r"(?i)^(exit\s+strategy|acquisition|ipo|returns)"
    ]
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a heading
        is_heading = False
        for pattern in heading_patterns:
            if re.match(pattern, line) and len(line) < 150:  # Reasonable heading length
                # Clean up the heading
                current_section = re.sub(r"^[\d\.\-\s]+", "", line).strip()
                current_section = current_section.title()
                sections[current_section] = []
                is_heading = True
                break
        
        # Also check for numbered sections
        if not is_heading and re.match(r"^\d+[\.\)]\s+[A-Z]", line) and len(line) < 100:
            current_section = re.sub(r"^\d+[\.\)]\s+", "", line).strip()
            sections[current_section] = []
            is_heading = True
        
        if not is_heading:
            sections.setdefault(current_section, []).append(line)
    
    # Clean up sections
    cleaned_sections = {}
    for k, v in sections.items():
        if v:  # Only keep non-empty sections
            content = "\n".join(v).strip()
            if content:
                cleaned_sections[k] = content
    
    return cleaned_sections

# Enhanced CRM-focused data extraction with better prompting
def extract_crm_structured_data(text):
    """Extract CRM-specific structured data with improved accuracy"""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)
        
        extraction_prompt = """
        Extract the following information from this pitch deck text. Return ONLY the values in the exact format shown:
        
        company_name: [Company name]
        ask: [Funding amount requested, e.g., "$2M", "₹5Cr"]
        revenue: [Current revenue, e.g., "$100K ARR", "₹50L"]
        valuation: [Company valuation, e.g., "$10M pre-money"]
        sector: [Industry sector, e.g., "FinTech", "HealthTech"]
        stage: [Company stage, e.g., "Seed", "Series A"]
        prior_funding: [Previous funding, e.g., "₹2Cr Seed 2022"]
        source: Pitch Deck Upload
        assign: [Founder names and roles]
        description: [What the company does in 1-2 sentences]
        
        INSTRUCTIONS:
        - Extract exact values from the text
        - Use "Not found" if information is not present
        - Include currency symbols and units
        - Keep responses concise and factual
        - Look for financial amounts carefully
        
        Text to analyze:
        """
        
        messages = [
            SystemMessage(content="You are a data extraction expert. Extract only the requested information in the exact format specified."),
            HumanMessage(content=f"{extraction_prompt}\n\n{text}")
        ]
        
        response = llm.invoke(messages)
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"CRM extraction error: {e}")
        return None

# Improved parsing with better error handling
def parse_crm_data(structured_text):
    """Parse structured text into CRM dictionary"""
    crm_data = {}
    
    if not structured_text:
        return crm_data
    
    required_fields = ['company_name', 'ask', 'revenue', 'valuation', 'sector', 'stage', 'prior_funding', 'source', 'assign', 'description']
    
    lines = structured_text.split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key in required_fields:
                crm_data[key] = value if value and value.lower() not in ["not found", "not mentioned", "n/a"] else "Not found"
    
    # Ensure all required fields exist
    for field in required_fields:
        if field not in crm_data:
            crm_data[field] = "Not found"
    
    return crm_data

# Enhanced CRM extraction with fallback
def extract_crm_data_with_fallback(text):
    """Main CRM extraction with fallback mechanism"""
    try:
        structured_text = extract_crm_structured_data(text)
        if structured_text:
            crm_data = parse_crm_data(structured_text)
            if crm_data:
                return crm_data
        
        # Fallback to regex extraction
        return regex_fallback_extraction(text)
        
    except Exception as e:
        logger.error(f"CRM extraction failed: {e}")
        return default_crm_data()

def regex_fallback_extraction(text):
    """Fallback extraction using regex patterns"""
    crm_data = default_crm_data()
    
    # Enhanced patterns for better extraction
    patterns = {
        'ask': [
            r'(?:seeking|raising|ask|need|require).*?[₹$]?\s*(\d+(?:\.\d+)?)\s*([KMBCr]*)\s*(?:million|crore|lakh|k|m|b)?',
            r'funding.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*([KMBCr]*)',
            r'investment.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*([KMBCr]*)'
        ],
        'revenue': [
            r'revenue.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*([KMBCr]*)',
            r'sales.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*([KMBCr]*)',
            r'(\d+(?:\.\d+)?)\s*([KMBCr]*)\s*(?:ARR|MRR|annual|monthly)'
        ],
        'valuation': [
            r'valuation.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*([KMBCr]*)',
            r'valued.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*([KMBCr]*)',
            r'(?:pre|post)[-\s]money.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*([KMBCr]*)'
        ]
    }
    
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                unit = match.group(2) if len(match.groups()) > 1 else ""
                crm_data[field] = f"${value}{unit}" if unit else f"${value}"
                break
    
    return crm_data

def default_crm_data():
    """Return default CRM data structure"""
    return {
        'company_name': 'Not found',
        'ask': 'Not found',
        'revenue': 'Not found',
        'valuation': 'Not found',
        'sector': 'Not found',
        'stage': 'Not found',
        'prior_funding': 'Not found',
        'source': 'Pitch Deck Upload',
        'assign': 'Not found',
        'description': 'Not found'
    }

def extract_number_cr(value):
    """Convert value to number for Zoho integration"""
    if not value:
        return 0.0
    match = re.search(r'([\d.]+)\s*Cr', value)
    if match:
        return float(match.group(1))
    return 0.0


# Check for specific CRM queries
def is_specific_crm_query(query):
    """Identify specific CRM field queries"""
    query_lower = query.lower()
    
    specific_keywords = {
        'ask': ['ask', 'funding', 'investment', 'raise', 'capital', 'seeking', 'round'],
        'founder': ['founder', 'ceo', 'team', 'who founded', 'leadership', 'management'],
        'revenue': ['revenue', 'sales', 'income', 'earnings', 'arr', 'mrr'],
        'valuation': ['valuation', 'worth', 'valued', 'pre-money', 'post-money'],
        'company': ['company name', 'what company', 'name of company'],
        'description': ['what do they do', 'what does the company do', 'business', 'product', 'service']
    }
    
    for field, keywords in specific_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return field
    
    return None

def generate_crm_response(field, crm_data):
    """Generate focused response for CRM field queries"""
    if not crm_data:
        return "No CRM data available. Please upload a pitch deck first."
    
    field_mapping = {
        'company': 'company_name',
        'founder': 'assign',
        'ask': 'ask',
        'revenue': 'revenue',
        'valuation': 'valuation',
        'description': 'description'
    }
    
    crm_field = field_mapping.get(field, field)
    value = crm_data.get(crm_field, "Not found")
    
    if not value or value == "Not found":
        return f"**{field.title()}:** Not mentioned in the pitch deck."
    
    return f"**{field.title()}:** {value}"

# Enhanced comprehensive analysis
def extract_comprehensive_analysis(text):
    """Generate comprehensive VC analysis"""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.1)
        
        analysis_prompt = """
        Conduct a comprehensive VC analysis of this pitch deck. Structure your analysis as follows:
        
        ## EXECUTIVE SUMMARY
        - Investment opportunity overview
        - Key strengths and concerns
        - Investment recommendation
        
        ## COMPANY & TEAM ANALYSIS
        - Business model and value proposition
        - Founder backgrounds and team assessment
        - Competitive positioning
        
        ## MARKET OPPORTUNITY
        - Market size and growth potential
        - Market timing and competitive landscape
        - Go-to-market strategy
        
        ## FINANCIAL ANALYSIS
        - Revenue model and traction
        - Unit economics and scalability
        - Funding requirements and use of funds
        
        ## RISK ASSESSMENT
        - Market and competitive risks
        - Execution and team risks
        - Financial and regulatory risks
        
        ## INVESTMENT DECISION
        - Key questions for due diligence
        - Recommended next steps
        
        Be specific, cite exact information, and provide actionable insights.
        """
        
        messages = [
            SystemMessage(content="You are a senior VC analyst. Provide comprehensive, actionable investment analysis."),
            HumanMessage(content=f"{analysis_prompt}\n\n{text}")
        ]
        
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        logger.error(f"Analysis generation failed: {e}")
        return "Error generating analysis"

# Enhanced section matching
def match_section(key, sections, structured_data=None):
    """Match query to relevant sections"""
    key = key.lower()
    
    if structured_data and "founders" in key:
        return structured_data
    
    lookup = {
        "founder": ["founder", "team", "leadership", "management", "ceo"],
        "valuation": ["valuation", "worth", "pre-money", "post-money"],
        "ask": ["ask", "funding", "investment", "capital", "raise"],
        "market": ["market", "opportunity", "tam", "sam", "som"],
        "problem": ["problem", "pain point", "challenge"],
        "solution": ["solution", "product", "technology"],
        "traction": ["traction", "revenue", "growth", "metrics"],
        "competition": ["competition", "competitors", "competitive"]
    }
    
    best_match = None
    best_score = 0
    
    if key in lookup:
        for tag in lookup[key]:
            for section_key, section_content in sections.items():
                if tag in section_key.lower():
                    return section_content
                if tag in section_content.lower():
                    score = section_content.lower().count(tag)
                    if score > best_score:
                        best_match = section_content
                        best_score = score
    
    if best_match:
        return best_match
    
    # Fuzzy matching
    matches = get_close_matches(key, [k.lower() for k in sections.keys()], n=1, cutoff=0.3)
    if matches:
        for k in sections:
            if k.lower() == matches[0]:
                return sections[k]
    
    return "Not mentioned in deck."

# SERP API search
def search_serpapi(query):
    """Search using SERP API"""
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
            "num": 5
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
            
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
            
            if any(x in query.lower() for x in ["founder", "background", "education", "experience"]):
                search_prompt = f"""Analyze these search results for founder background:
                1. Educational background
                2. Professional experience
                3. Notable achievements
                4. Industry expertise
                
                Results:\n{combined}"""
            else:
                search_prompt = f"Summarize for VC analysis:\n{combined}"
            
            return llm.invoke(search_prompt).content.strip()
            
        return f"❌ SERP API error: {r.status_code}"
        
    except Exception as e:
        return f"❌ Web search failed: {str(e)}"

# Chat function
def chat_with_ai(user_input):
    """Handle AI chat interactions"""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.1)
        
        # Check for specific CRM queries
        crm_field = is_specific_crm_query(user_input)
        if crm_field and st.session_state.crm_data:
            return generate_crm_response(crm_field, st.session_state.crm_data)
        
        # Check for web search requests
        if any(term in user_input.lower() for term in ["search", "web", "google", "internet", "background", "linkedin"]):
            search_query = user_input.replace("search", "").replace("web", "").replace("google", "").strip()
            if st.session_state.crm_data and st.session_state.crm_data.get('company_name') != 'Not found':
                search_query = f"{st.session_state.crm_data['company_name']} {search_query}"
            
            search_result = search_serpapi(search_query)
            return f"🔍 **Web Search Results:**\n\n{search_result}"
        
        # General query processing
        if st.session_state.parsed_doc:
            relevant_content = match_section(user_input, st.session_state.sections, st.session_state.structured_data)
            
            context = f"""
            You are analyzing a pitch deck. Here's the relevant content:
            
            {relevant_content}
            
            Full document available for context.
            
            Answer the user's question based on this content. Be specific and cite information from the pitch deck.
            """
            
            messages = [
                SystemMessage(content="You are a VC analyst assistant. Provide helpful, specific answers based on the pitch deck content."),
                HumanMessage(content=f"{context}\n\nUser question: {user_input}")
            ]
            
            response = llm.invoke(messages)
            return response.content
        else:
            return "Please upload a pitch deck first to analyze."
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return "Sorry, I encountered an error. Please try again."

# UI Configuration
st.set_page_config(page_title="Augmento - Pitch Evaluator", page_icon="🚀", layout="wide")
st.title("🚀 Augmento - The Pitch Evaluator")

# Sidebar
with st.sidebar:
    st.header("📋 Document Overview")
    if st.session_state.sections:
        for section_name in st.session_state.sections.keys():
            if st.button(section_name, key=f"section_{section_name}"):
                st.session_state.selected_section = section_name
    
    if st.session_state.file_uploaded:
        st.header("📊 Quick Stats")
        st.write(f"**Sections:** {len(st.session_state.sections)}")
        st.write(f"**Text Length:** {len(st.session_state.parsed_doc):,} chars")
    
    # CRM Data Display
    if st.session_state.crm_data:
        st.header("🔗 CRM Data")
        key_fields = ['company_name', 'ask', 'revenue', 'valuation', 'sector', 'stage']
        
        for field in key_fields:
            if field in st.session_state.crm_data:
                value = st.session_state.crm_data[field]
                if value and value != "Not found":
                    display_value = value[:25] + "..." if len(value) > 25 else value
                    st.write(f"**{field.replace('_', ' ').title()}:** {display_value}")
    
    # Chat History
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.session_state.selected_chat_index = None
            st.rerun()
        
        for i, (user, bot, timestamp) in enumerate(reversed(st.session_state.chat_history)):
            chat_index = len(st.session_state.chat_history) - 1 - i
            if st.button(f"Q{chat_index + 1}: {user[:20]}...", key=f"chat_{chat_index}"):
                st.session_state.selected_chat_index = chat_index
                st.rerun()

# Main Content
if st.session_state.structured_data:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Comprehensive Analysis"):
            st.session_state.show_comprehensive_analysis = True
    
    with col2:
        if st.button("📊 CRM Summary"):
            st.session_state.show_crm_summary = True

# Show comprehensive analysis
if hasattr(st.session_state, 'show_comprehensive_analysis') and st.session_state.show_comprehensive_analysis:
    st.header("🔍 Comprehensive VC Analysis")
    
    if not hasattr(st.session_state, 'comprehensive_analysis'):
        with st.spinner("🔄 Generating analysis..."):
            st.session_state.comprehensive_analysis = extract_comprehensive_analysis(st.session_state.parsed_doc)
    
    st.markdown(st.session_state.comprehensive_analysis)
    
    if st.button("❌ Close Analysis"):
        st.session_state.show_comprehensive_analysis = False
        st.rerun()
    st.markdown("---")

# Show CRM summary
if hasattr(st.session_state, 'show_crm_summary') and st.session_state.show_crm_summary:
    st.header("📊 CRM Data Summary")
    
    if st.session_state.crm_data:
        col1, col2 = st.columns(2)
        
        with col1:
            for field in ['company_name', 'ask', 'revenue', 'valuation', 'sector']:
                value = st.session_state.crm_data.get(field, "Not found")
                st.write(f"**{field.replace('_', ' ').title()}:** {value}")
        
        with col2:
            for field in ['stage', 'prior_funding', 'assign', 'description']:
                value = st.session_state.crm_data.get(field, "Not found")
                if len(value) > 50:
                    value = value[:50] + "..."
                st.write(f"**{field.replace('_', ' ').title()}:** {value}")
        
        st.subheader("📤 Export Data")
        st.json(st.session_state.crm_data)
    
    if st.button("❌ Close Summary"):
        st.session_state.show_crm_summary = False
        st.rerun()
    st.markdown("---")

# Show selected chat
if st.session_state.selected_chat_index is not None:
    chat_data = st.session_state.chat_history[st.session_state.selected_chat_index]
    user_q, bot_response, timestamp = chat_data
    
    st.header(f"💬 Chat #{st.session_state.selected_chat_index + 1}")
    st.markdown(f"**🧑 You:** {user_q}")
    st.markdown(f"**🤖 AI:**")
    st.markdown(bot_response)
    st.markdown(f"*{timestamp}*")
    
    if st.button("❌ Close Chat"):
        st.session_state.selected_chat_index = None
        st.rerun()
    st.markdown("---")

# Show selected section
if hasattr(st.session_state, 'selected_section') and st.session_state.selected_section:
    st.subheader(f"📖 {st.session_state.selected_section}")
    st.text_area("Content", st.session_state.sections[st.session_state.selected_section], height=200)

# File Upload (continuation from where it was cut off)
file = st.file_uploader("📤 Upload your pitch deck (PDF only)", type=["pdf"])

if file:
    with st.spinner("🔄 Processing pitch deck..."):
        file_bytes = file.read()
        text = extract_pdf_text(file_bytes)
        
        if text:
            st.session_state.parsed_doc = text
            st.session_state.sections = split_sections(text)
            st.session_state.file_uploaded = True
            
            # Extract CRM data
            st.session_state.crm_data = extract_crm_data_with_fallback(text)
            st.session_state.crm_data['received_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            
            st.success("✅ Pitch deck processed successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to extract text from PDF")

# Chat Interface
if st.session_state.file_uploaded:
    st.header("💬 Chat with AI")
    
    # Sample questions
    st.subheader("🔍 Quick Questions")
    sample_questions = [
        "What is the funding ask?",
        "Who are the founders?",
        "What is the company's valuation?",
        "What problem does this solve?",
        "What is the business model?",
        "Search for founder background",
        "Show me the market opportunity",
        "What is their traction?"
    ]
    
    cols = st.columns(4)
    for i, question in enumerate(sample_questions):
        with cols[i % 4]:
            if st.button(question, key=f"sample_q_{i}"):
                # Process the sample question
                with st.spinner("🤖 Analyzing..."):
                    response = chat_with_ai(question)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.chat_history.append((question, response, timestamp))
                    st.rerun()
    
    # Chat input
    user_input = st.text_input("Ask a question about the pitch deck:", key="chat_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("Send", type="primary")
    with col2:
        if st.button("🎯 Get Investment Recommendation"):
            with st.spinner("🤖 Generating recommendation..."):
                recommendation_query = "Based on this pitch deck, provide a detailed investment recommendation including strengths, weaknesses, and overall assessment."
                response = chat_with_ai(recommendation_query)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append((recommendation_query, response, timestamp))
                st.rerun()
    
    if send_button and user_input:
        with st.spinner("🤖 Analyzing..."):
            response = chat_with_ai(user_input)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append((user_input, response, timestamp))
            st.rerun()
    
    # Display recent chat
    if st.session_state.chat_history:
        st.subheader("💬 Recent Conversation")
        
        # Show last 3 conversations
        recent_chats = st.session_state.chat_history[-3:]
        for i, (user_q, bot_response, timestamp) in enumerate(reversed(recent_chats)):
            with st.expander(f"Q: {user_q[:50]}{'...' if len(user_q) > 50 else ''}", expanded=(i == 0)):
                st.markdown(f"**🧑 You:** {user_q}")
                st.markdown(f"**🤖 AI:**")
                st.markdown(bot_response)
                st.markdown(f"*{timestamp}*")

# Document Sections Display
if st.session_state.sections:
    st.header("📑 Document Sections")
    
    # Create tabs for different sections
    section_names = list(st.session_state.sections.keys())
    if section_names:
        tabs = st.tabs(section_names[:6])  # Limit to first 6 sections to avoid too many tabs
        
        for i, (section_name, section_content) in enumerate(list(st.session_state.sections.items())[:6]):
            with tabs[i]:
                st.markdown(f"### {section_name}")
                st.text_area("Content", section_content, height=300, key=f"section_content_{i}")
                
                # Section-specific analysis
                if st.button(f"🔍 Analyze {section_name}", key=f"analyze_{i}"):
                    with st.spinner(f"Analyzing {section_name}..."):
                        analysis_query = f"Provide a detailed analysis of the {section_name} section"
                        response = chat_with_ai(analysis_query)
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.chat_history.append((analysis_query, response, timestamp))
                        st.rerun()

# Export and Download Options
if st.session_state.crm_data:
    st.header("📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Copy CRM Data"):
            crm_text = json.dumps(st.session_state.crm_data, indent=2)
            st.code(crm_text, language="json")
    
    with col2:
        if st.button("📊 Generate Report"):
            with st.spinner("Generating report..."):
                report_query = "Generate a comprehensive investment report based on this pitch deck"
                report = chat_with_ai(report_query)
                st.download_button(
                    label="⬇️ Download Report",
                    data=report,
                    file_name=f"investment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    with col3:
        if st.button("🔗 Send to CRM"):
            if zoho_webhook_url:
                send_to_zoho_webhook(st.session_state.crm_data)
                st.success("✅ Data sent to CRM!")
            else:
                st.warning("⚠️ CRM webhook URL not configured")

# Footer
st.markdown("---")
st.markdown("### 🚀 Augmento - Pitch Deck Analyzer")
st.markdown("Powered by OpenAI GPT-4 and advanced NLP processing")

# Debug information (only show in development)
if st.checkbox("🔧 Show Debug Info"):
    st.subheader("Debug Information")
    if st.session_state.parsed_doc:
        st.write(f"Document length: {len(st.session_state.parsed_doc)} characters")
        st.write(f"Number of sections: {len(st.session_state.sections) if st.session_state.sections else 0}")
        st.write(f"Chat history length: {len(st.session_state.chat_history)}")
        
        if st.session_state.crm_data:
            st.write("CRM Data:")
            st.json(st.session_state.crm_data)
        
        if st.button("Show Raw Text"):
            st.text_area("Raw Extracted Text", st.session_state.parsed_doc, height=300)

# Error handling for API limits
def handle_api_error(func, *args, **kwargs):
    """Wrapper to handle API errors gracefully"""
    try:
        return func(*args, **kwargs)
    except RateLimitError:
        st.error("❌ API rate limit exceeded. Please try again later.")
        return None
    except APIError as e:
        st.error(f"❌ API error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

# Add custom CSS for better styling
st.markdown("""
<style>
.stButton > button {
    width: 100%;
    margin-bottom: 5px;
}
.stTextInput > div > div > input {
    border-radius: 10px;
}
.stTextArea > div > div > textarea {
    border-radius: 10px;
}
.stExpander > div > div > div > div {
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)


