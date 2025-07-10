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
zoho_webhook_url = os.getenv("ZOHO_WEBHOOK_URL")

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

# Enhanced CRM-focused data extraction
# Enhanced CRM-focused data extraction with better parsing
def extract_crm_structured_data(text):
    """Extract CRM-specific structured data from pitch deck using LLM"""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)

        extraction_prompt = """
        Analyze this pitch deck text and extract CRM-specific information. 
        Return the data in this EXACT format (key: value pairs separated by newlines):
        
        company_name: [Extract exact company name or "Not mentioned"]
        ask: [Extract funding amount being sought, e.g., "$2M Series A" or "Not mentioned"]
        revenue: [Extract current revenue figures, e.g., "$500K ARR" or "Not mentioned"]
        valuation: [Extract current valuation, e.g., "$5M pre-money" or "Not mentioned"]
        sector: [Extract industry/sector, e.g., "FinTech", "HealthTech", "SaaS" or "Not mentioned"]
        stage: [Extract company stage, e.g., "Pre-Seed", "Seed", "Series A" or "Not mentioned"]
        prior_funding: [Extract previous funding rounds, e.g., "₹2Cr Seed 2023" or "Not mentioned"]
        source: Pitch Deck Upload
        assign: [Extract founder names and roles, e.g., "John Doe (CEO), Jane Smith (CTO)" or "Not mentioned"]
        description: [Brief 1-2 sentence description of what the company does or "Not mentioned"]
        
        CRITICAL INSTRUCTIONS:
        1. Return ONLY the key: value pairs in the exact format shown above
        2. Do NOT add any additional text, explanations, or formatting
        3. Look through the ENTIRE document for information
        4. Extract specific numbers and amounts wherever possible
        5. Include currency symbols and units (K, M, B, Cr)
        6. If information is not found, use "Not mentioned"
        7. Keep each value on a single line
        
        Text to analyze:
        """

        messages = [
            SystemMessage(content="You are a data extraction specialist. Return ONLY the requested key-value pairs in the exact format specified. Do not add any additional text or explanations."),
            HumanMessage(content=f"{extraction_prompt}\n\n{text}")
        ]

        response = llm.invoke(messages)
        return response.content.strip()

    except Exception as e:
        logger.error(f"Error extracting CRM structured data: {e}")
        return """company_name: Error extracting data
ask: Not mentioned
revenue: Not mentioned
valuation: Not mentioned
sector: Not mentioned
stage: Not mentioned
prior_funding: Not mentioned
source: Pitch Deck Upload
assign: Not mentioned
description: Error extracting data"""

# Improved parsing function with better error handling
def parse_crm_data(structured_text):
    """Parse the structured text into a dictionary for CRM integration"""
    crm_data = {}
    
    if not structured_text:
        logger.warning("No structured text provided for parsing")
        return crm_data
    
    logger.info(f"Parsing structured text: {structured_text[:200]}...")
    
    lines = structured_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if ':' in line:
            # Split only on the first colon to handle values with colons
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            # Only keep valid CRM fields
            valid_fields = ['company_name', 'ask', 'revenue', 'valuation', 'sector', 'stage', 'prior_funding', 'source', 'assign', 'description']
            
            if key in valid_fields:
                # Clean up the value
                if value and value.lower() not in ["not mentioned", "not found", "n/a", "na", ""]:
                    crm_data[key] = value
                else:
                    crm_data[key] = "Not found"
            else:
                logger.warning(f"Unknown field encountered: {key}")
    
    # Ensure all required fields are present
    required_fields = ['company_name', 'ask', 'revenue', 'valuation', 'sector', 'stage', 'prior_funding', 'source', 'assign', 'description']
    for field in required_fields:
        if field not in crm_data:
            crm_data[field] = "Not found"
    
    logger.info(f"Parsed CRM data: {crm_data}")
    return crm_data

# Enhanced debugging function to help troubleshoot
def debug_crm_extraction(text):
    """Debug function to help troubleshoot CRM extraction issues"""
    logger.info("Starting CRM extraction debug...")
    
    # Check if text is available
    if not text:
        logger.error("No text provided for extraction")
        return None
    
    logger.info(f"Text length: {len(text)} characters")
    logger.info(f"Text preview: {text[:500]}...")
    
    try:
        # Extract structured data
        structured_text = extract_crm_structured_data(text)
        logger.info(f"Structured text result: {structured_text}")
        
        # Parse the data
        crm_data = parse_crm_data(structured_text)
        logger.info(f"Final CRM data: {crm_data}")
        
        return crm_data
        
    except Exception as e:
        logger.error(f"Debug extraction failed: {e}")
        return None

# Alternative fallback extraction using regex patterns
def fallback_crm_extraction(text):
    """Fallback extraction using regex patterns when LLM fails"""
    logger.info("Using fallback CRM extraction...")
    
    crm_data = {
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
    
    # Common patterns for extraction
    patterns = {
        'ask': [
            r'seeking\s+[$₹]?(\d+(?:\.\d+)?)\s*([KMB]?)\s*(.*?)(?:funding|investment|capital)',
            r'raise\s+[$₹]?(\d+(?:\.\d+)?)\s*([KMB]?)',
            r'asking\s+for\s+[$₹]?(\d+(?:\.\d+)?)\s*([KMB]?)',
            r'funding\s+required?\s*:?\s*[$₹]?(\d+(?:\.\d+)?)\s*([KMB]?)'
        ],
        'revenue': [
            r'revenue\s+of\s+[$₹]?(\d+(?:\.\d+)?)\s*([KMB]?)',
            r'[$₹](\d+(?:\.\d+)?)\s*([KMB]?)\s+ARR',
            r'[$₹](\d+(?:\.\d+)?)\s*([KMB]?)\s+MRR',
            r'sales\s+of\s+[$₹]?(\d+(?:\.\d+)?)\s*([KMB]?)'
        ],
        'valuation': [
            r'valuation\s+of\s+[$₹]?(\d+(?:\.\d+)?)\s*([KMB]?)',
            r'valued\s+at\s+[$₹]?(\d+(?:\.\d+)?)\s*([KMB]?)',
            r'pre-money\s+[$₹]?(\d+(?:\.\d+)?)\s*([KMB]?)',
            r'post-money\s+[$₹]?(\d+(?:\.\d+)?)\s*([KMB]?)'
        ]
    }
    
    # Try to extract using patterns
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1)
                unit = match.group(2) if len(match.groups()) > 1 else ""
                crm_data[field] = f"${value}{unit}" if unit else f"${value}"
                break
    
    # Extract company name (usually appears early in the document)
    company_patterns = [
        r'(?:company|startup|firm)\s+name\s*:?\s*([A-Z][a-zA-Z\s&]+)',
        r'^([A-Z][a-zA-Z\s&]+)\s+(?:pitch|deck|presentation)',
        r'(?:founded|started)\s+([A-Z][a-zA-Z\s&]+)'
    ]
    
    for pattern in company_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            crm_data['company_name'] = match.group(1).strip()
            break
    
    return crm_data

# Modified main extraction function with fallback
def extract_crm_data_with_fallback(text):
    """Main CRM extraction function with fallback mechanism"""
    try:
        # Try primary extraction
        crm_data = debug_crm_extraction(text)
        
        # Check if extraction was successful
        if crm_data and any(value != "Not found" for value in crm_data.values() if value != "Pitch Deck Upload"):
            return crm_data
        else:
            logger.warning("Primary extraction failed, trying fallback...")
            return fallback_crm_extraction(text)
            
    except Exception as e:
        logger.error(f"All extraction methods failed: {e}")
        # Return default structure
        return {
            'company_name': 'Extraction failed',
            'ask': 'Not found',
            'revenue': 'Not found',
            'valuation': 'Not found',
            'sector': 'Not found',
            'stage': 'Not found',
            'prior_funding': 'Not found',
            'source': 'Pitch Deck Upload',
            'assign': 'Not found',
            'description': 'Extraction failed'
        }

def extract_number_cr(value):
    """Convert '₹ 12 Cr Pre-Series A' to 12.0 (float)"""
    if not value:
        return 0.0
    match = re.search(r'([\d.]+)\s*Cr', value)
    if match:
        return float(match.group(1))
    return 0.0

def send_to_zoho_webhook(crm_data):
    if not zoho_webhook_url:
        logger.warning("❌ ZOHO_WEBHOOK_URL not set in .env")
        return
    
    try:
        
        # Preprocess values for Zoho
        crm_payload = {
            "company_name": crm_data.get("company_name", ""),
            "ask": extract_number_cr(crm_data.get("ask", "")),
            "valuation": extract_number_cr(crm_data.get("valuation", "")),
            "revenue": crm_data.get("revenue", ""),
            "sector": crm_data.get("sector", ""),
            "stage": crm_data.get("stage", ""),
            "prior_funding": crm_data.get("prior_funding", ""),
            "description": crm_data.get("description", ""),
            "source": crm_data.get("source", ""),
            "assign": crm_data.get("assign", ""),
            "received_date": crm_data.get("received_date", "")
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(zoho_webhook_url, json=crm_payload, headers=headers)

        if response.status_code == 200:
            logger.info("✅ CRM data sent to Zoho Flow successfully")
        else:
            logger.warning(f"⚠️ Webhook error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Failed to send to Zoho webhook: {e}")

# Check if query is asking for specific CRM data
def is_specific_crm_query(query):
    """Check if the query is asking for specific CRM information"""
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

# Generate specific CRM response
def generate_crm_response(field, crm_data):
    """Generate a focused response for specific CRM field queries"""
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
    value = crm_data.get(crm_field, "Not mentioned")

    if not value or value == "Not mentioned":
        return f"**{field.title()}:** Not mentioned in the pitch deck."

    return f"**{field.title()}:** {value}"

# Enhanced comprehensive analysis
def extract_comprehensive_analysis(text):
    """Extract comprehensive analysis for deep insights"""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.1)

        analysis_prompt = """
        Conduct a COMPREHENSIVE VC analysis of this pitch deck. Provide detailed insights in the following structure:
        
        ## EXECUTIVE SUMMARY
        - One-paragraph overview of the investment opportunity
        - Key strengths and concerns
        - Overall recommendation (Strong/Moderate/Weak investment case)
        
        ## COMPANY OVERVIEW
        - Business model and value proposition
        - Stage of company and maturity
        - Competitive positioning
        
        ## FOUNDERS & TEAM ANALYSIS
        - Founder backgrounds and relevant experience
        - Team composition and key gaps
        - Leadership assessment
        
        ## MARKET OPPORTUNITY
        - Market size and growth potential (TAM/SAM/SOM)
        - Market timing and trends
        - Competitive landscape analysis
        
        ## PRODUCT & TECHNOLOGY
        - Product differentiation and unique value prop
        - Technology stack and IP considerations
        - Product-market fit evidence
        
        ## TRACTION & METRICS
        - Revenue metrics and growth trajectory
        - Customer acquisition and retention
        - Key performance indicators
        - Milestone achievements
        
        ## FINANCIAL ANALYSIS
        - Revenue model and pricing strategy
        - Unit economics and scalability
        - Funding history and use of funds
        - Financial projections assessment
        
        ## INVESTMENT DETAILS
        - Funding ask and valuation analysis
        - Deal terms and structure
        - Use of funds breakdown
        - Exit strategy considerations
        
        ## RISK ASSESSMENT
        - Market risks and competitive threats
        - Execution risks and team capabilities
        - Financial and scalability risks
        - Regulatory and compliance considerations
        
        ## RECOMMENDATIONS
        - Due diligence areas to focus on
        - Key questions for management
        - Suggested next steps
        
        IMPORTANT: 
        - Be specific and cite exact information from the deck
        - Highlight missing information that should be obtained
        - Provide actionable insights for investment decision
        - Use bullet points for readability
        - Include specific numbers, percentages, and metrics
        
        Pitch Deck Content:
        """

        messages = [
            SystemMessage(content="You are a senior VC analyst conducting comprehensive due diligence. Provide detailed, actionable insights that will help make investment decisions. Be thorough, specific, and highlight both opportunities and risks."),
            HumanMessage(content=f"{analysis_prompt}\n\n{text}")
        ]

        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        logger.error(f"Error generating comprehensive analysis: {e}")
        return "Error generating comprehensive analysis"

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

            if "people_also_ask" in data:
                combined += "\n--- Related Questions ---\n"
                for paa in data["people_also_ask"][:3]:
                    combined += f"Q: {paa.get('question', '')}\nA: {paa.get('snippet', '')}\n\n"

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
st.set_page_config(page_title="Augmento - The Pitch Evaluator", page_icon="🕵️")
st.title("🕵️ Augmento - The Pitch Evaluator")

# Sidebar content
with st.sidebar:
    st.header("📋 Overview")
    if st.session_state.sections:
        for section_name in st.session_state.sections.keys():
            if st.button(section_name, key=f"section_{section_name}"):
                st.session_state.selected_section = section_name

    # Show quick stats
    if st.session_state.file_uploaded:
        st.header("📊 Quick Stats")
        st.write(f"Total sections: {len(st.session_state.sections)}")
        st.write(f"Total text length: {len(st.session_state.parsed_doc)} chars")

    # Show CRM data
    if st.session_state.crm_data:
        st.header("🔗 CRM Integration Data")

        # Display key CRM fields
        crm_fields = ['company_name', 'ask', 'revenue', 'valuation', 'sector', 'stage', 'prior_funding', 'source', 'assign', 'description']
        for field in crm_fields:
            if field in st.session_state.crm_data and st.session_state.crm_data[field]:
                display_value = st.session_state.crm_data[field]
                if len(display_value) > 50:
                    display_value = display_value[:50] + "..."
                st.write(f"**{field.replace('_', ' ').title()}:** {display_value}")

        # Export CRM data button
        if st.button("📤 Export CRM Data"):
            st.json(st.session_state.crm_data)

    # Chat History in Sidebar
    if st.session_state.chat_history:
        st.header("💬 Chat History")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.selected_chat_index = None
            st.rerun()

        for i, (user, bot, timestamp) in enumerate(reversed(st.session_state.chat_history)):
            chat_index = len(st.session_state.chat_history) - 1 - i
            if st.button(f"Q{chat_index + 1}: {user[:30]}...", key=f"chat_{chat_index}"):
                st.session_state.selected_chat_index = chat_index
                st.rerun()

# Main content area

# Show comprehensive analysis if available
if st.session_state.structured_data:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔍 View Comprehensive Analysis"):
            st.session_state.show_comprehensive_analysis = True

    with col2:
        if st.button("📊 View CRM Data Summary"):
            st.session_state.show_crm_summary = True

# Show comprehensive analysis
if hasattr(st.session_state, 'show_comprehensive_analysis') and st.session_state.show_comprehensive_analysis:
    st.header("🔍 Comprehensive VC Analysis")

    if not hasattr(st.session_state, 'comprehensive_analysis'):
        with st.spinner("🔄 Generating comprehensive analysis..."):
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
        # Display CRM data in clean format
        st.subheader("🔑 CRM Fields")

        for field, value in st.session_state.crm_data.items():
            if value:
                st.write(f"**{field.replace('_', ' ').title()}:** {value}")

        # JSON export
        st.subheader("📤 Export Data")
        st.json(st.session_state.crm_data)

        # Copy to clipboard button
        if st.button("📋 Copy JSON to Clipboard"):
            st.code(json.dumps(st.session_state.crm_data, indent=2))

    if st.button("❌ Close CRM Summary"):
        st.session_state.show_crm_summary = False
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

        # Extract CRM-specific structured data with improved error handling
        with st.spinner("🔍 Extracting CRM data..."):
            try:
                # Use the improved extraction function
                st.session_state.crm_data = extract_crm_data_with_fallback(text)
                
                # Add received_date (upload date)
                st.session_state.crm_data["received_date"] = datetime.today().strftime("%Y-%m-%d")
                
                # Also store the raw structured text for debugging
                st.session_state.structured_data = extract_crm_structured_data(text)
                
                # Log the results for debugging
                logger.info(f"CRM extraction completed. Found {len([k for k, v in st.session_state.crm_data.items() if v != 'Not found'])} fields")
                
                # Optional: Send to Zoho webhook if data is valid
                if st.session_state.crm_data.get('company_name') != 'Not found':
                    send_to_zoho_webhook(st.session_state.crm_data)
                
            except Exception as e:
                logger.error(f"CRM extraction failed: {e}")
                # Set default values on failure
                st.session_state.crm_data = {
                    'company_name': 'Extraction failed',
                    'ask': 'Not found',
                    'revenue': 'Not found',
                    'valuation': 'Not found',
                    'sector': 'Not found',
                    'stage': 'Not found',
                    'prior_funding': 'Not found',
                    'source': 'Pitch Deck Upload',
                    'assign': 'Not found',
                    'description': 'Extraction failed',
                    'received_date': datetime.today().strftime("%Y-%m-%d")
                }
                st.error(f"❌ CRM extraction failed: {e}")

    st.success("✅ Pitch deck parsed and CRM data extracted!")
    
    # Add debugging information
    if st.checkbox("🔍 Show Debug Info"):
        st.subheader("Debug Information")
        st.write(f"**Text Length:** {len(st.session_state.parsed_doc)} characters")
        st.write(f"**Sections Found:** {len(st.session_state.sections)}")
        st.write(f"**CRM Fields Extracted:** {len([k for k, v in st.session_state.crm_data.items() if v != 'Not found'])}")
        
        with st.expander("Raw Structured Data"):
            st.text(st.session_state.structured_data)
            
        with st.expander("Parsed CRM Data"):
            st.json(st.session_state.crm_data)

    # Show CRM data preview
if st.session_state.crm_data:
    st.subheader("🔗 CRM Data Preview")
    
    # First row - main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        company_name = st.session_state.crm_data.get('company_name', 'Not found')
        display_name = company_name[:15] + "..." if len(company_name) > 15 else company_name
        st.metric("Company", display_name)

    with col2:
        ask_value = st.session_state.crm_data.get('ask', 'Not found')
        display_ask = ask_value[:12] + "..." if len(ask_value) > 12 else ask_value
        st.metric("Ask", display_ask)

    with col3:
        valuation_value = st.session_state.crm_data.get('valuation', 'Not found')
        display_valuation = valuation_value[:12] + "..." if len(valuation_value) > 12 else valuation_value
        st.metric("Valuation", display_valuation)
    
    # Second row - additional details
    col4, col5, col6 = st.columns(3)
    with col4:
        sector_value = st.session_state.crm_data.get('sector', 'Not found')
        display_sector = sector_value[:15] + "..." if len(sector_value) > 15 else sector_value
        st.metric("Sector", display_sector)

    with col5:
        stage_value = st.session_state.crm_data.get('stage', 'Not found')
        display_stage = stage_value[:15] + "..." if len(stage_value) > 15 else stage_value
        st.metric("Stage", display_stage)

    with col6:
        prior_funding = st.session_state.crm_data.get('prior_funding', 'Not found')
        display_prior = prior_funding[:12] + "..." if len(prior_funding) > 12 else prior_funding
        st.metric("Prior Funding", display_prior)
    
    # Third row - description
    if st.session_state.crm_data.get('description'):
        st.markdown("**📝 Brief Description:**")
        description = st.session_state.crm_data.get('description', '')
        st.markdown(f"<small>{description}</small>", unsafe_allow_html=True)
# Show selected section
if hasattr(st.session_state, 'selected_section') and st.session_state.selected_section:
    st.subheader(f"📖 {st.session_state.selected_section}")
    st.text_area("Content", st.session_state.sections[st.session_state.selected_section], height=200)

# Enhanced prompt input
st.markdown("### 💬 Ask Questions")
st.markdown("**Quick CRM Queries:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💰 What is their ask?"):
        st.session_state.auto_query = "What is their ask?"

with col2:
    if st.button("👥 Who are the founders?"):
        st.session_state.auto_query = "Who are the founders?"

with col3:
    if st.button("💵 What is their revenue?"):
        st.session_state.auto_query = "What is their revenue?"

col4, col5, col6 = st.columns(3)

with col4:
    if st.button("🏢 Company valuation?"):
        st.session_state.auto_query = "What is their valuation?"

with col5:
    if st.button("📋 Company description?"):
        st.session_state.auto_query = "What does the company do?"

with col6:
    if st.button("📊 Full Analysis"):
        st.session_state.auto_query = "Provide a comprehensive analysis of this pitch deck"

st.markdown("**Example specific queries:**")
st.markdown("- `What is their ask?` - Returns just the funding ask")
st.markdown("- `Who are the founders?` - Returns just founder information")
st.markdown("- `What is their revenue?` - Returns just revenue data")
st.markdown("- `What does the company do?` - Returns just company description")

# Handle auto queries
if hasattr(st.session_state, 'auto_query') and st.session_state.auto_query:
    user_query = st.session_state.auto_query
    st.session_state.auto_query = None
else:
    user_query = st.chat_input("💬 Ask about the pitch deck...")

if user_query:
    # Clear any selected chat when asking new question
    st.session_state.selected_chat_index = None

    # Check if this is a specific CRM query
    specific_field = is_specific_crm_query(user_query)

    if specific_field and st.session_state.crm_data:
        # Handle specific CRM queries with direct response
        response = generate_crm_response(specific_field, st.session_state.crm_data)

        st.markdown(f"**🧑 You:** {user_query}")
        st.markdown("**🤖 Manna:**")
        st.markdown(response)

        # Store in chat history
        st.session_state.chat_history.append(
            (user_query, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
    else:
        # Handle comprehensive queries
        with st.spinner("🤖 Analyzing..."):
            try:
                context = st.session_state.parsed_doc or ""
                llm = ChatOpenAI(
                    model="gpt-4o", 
                    openai_api_key=openai_api_key, 
                    streaming=True,
                    temperature=0.1,
                    max_tokens=2000
                )

                # Generate comprehensive analysis if not already done
                if not hasattr(st.session_state, 'comprehensive_analysis'):
                    st.session_state.comprehensive_analysis = extract_comprehensive_analysis(context)

                context_msg = f"COMPREHENSIVE ANALYSIS:\n{st.session_state.comprehensive_analysis}\n\nCRM DATA:\n{st.session_state.structured_data}\n\nFULL DOCUMENT:\n{context[:2000]}"

                # Enhanced system message for deeper analysis
                system_msg = """You are a senior VC analyst with 15+ years of experience. Provide DEEP, actionable insights for investment decisions.

                Analysis Guidelines:
                1. Be specific and cite exact information from the deck
                2. Provide concrete numbers, metrics, and percentages
                3. Highlight missing critical information
                4. Identify red flags and investment risks
                5. Suggest specific due diligence questions
                6. Compare against industry benchmarks when relevant
                7. Provide clear investment recommendation rationale
                8. Focus on scalability and market opportunity assessment
                9. Analyze team capability for execution
                10. Evaluate competitive moat and differentiation
                
                Response Format:
                - Use clear headers and bullet points
                - Include specific actionable insights
                - Highlight key metrics and financial data
                - Provide both opportunities and risks
                - End with specific next steps or questions
                """

                messages = [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=f"{context_msg}\n\nVC Analysis Request: {user_query}")
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
            except APIError as e:
                st.error(f"❌ OpenAI API error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                logger.error(f"Analysis error: {str(e)}", exc_info=True)

# Footer
st.markdown("---")
st.markdown("**💡 Pro Tip:** Use specific queries like 'What is their ask?' for quick CRM data, or ask comprehensive questions for detailed analysis.")
