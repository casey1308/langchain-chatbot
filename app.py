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

# ENVIRONMENT VARIABLE VALIDATION
def validate_environment():
    """Validate all required environment variables"""
    required_vars = {
        "OPENAI_API_KEY": openai_api_key,
        "SERPAPI_API_KEY": serpapi_key,
        "ZOHO_WEBHOOK_URL": zoho_webhook_url
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        st.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        st.info("Please add these variables to your .env file:")
        for var in missing_vars:
            st.code(f"{var}=your_value_here")
        return False
    
    return True

# Call validation
if not validate_environment():
    st.stop()

# Initialize session state
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections", "structured_data", "selected_chat_index", "crm_data", "show_comprehensive_analysis", "show_crm_summary", "selected_section"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        elif key == "file_uploaded":
            st.session_state[key] = False
        elif key == "selected_chat_index":
            st.session_state[key] = None
        elif key in ["show_comprehensive_analysis", "show_crm_summary"]:
            st.session_state[key] = False
        else:
            st.session_state[key] = None

# Enhanced text cleaner
def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[•◦▪▫‣⁃]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
    return clean_text(full_text)

def split_sections(text):
    sections = {"General": []}
    current_section = "General"
    heading_patterns = [
        r"(?i)(founder|co-founder|team|leadership|management|ceo|cto|cfo|about\s+us)",
        r"(?i)(problem|solution|market|business\s+model|revenue|traction|competition)",
        r"(?i)(financials|funding|valuation|ask|investment|series|round|cap\s+table)",
        r"(?i)(product|technology|demo|features|roadmap|development)",
        r"(?i)(market\s+size|addressable\s+market|tam|sam|som|opportunity)",
        r"(?i)(go-to-market|marketing|sales|strategy|growth|expansion)"
    ]
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        is_heading = False
        for pattern in heading_patterns:
            if re.match(pattern, line) and len(line) < 100:
                current_section = line
                sections[current_section] = []
                is_heading = True
                break
        if not is_heading:
            sections.setdefault(current_section, []).append(line)
    return {k: "\n".join(v) for k, v in sections.items() if v}

def parse_currency_amount(text):
    """Smart currency parser that extracts numerical values from text"""
    if not text or text.lower() in ["not mentioned", "not found", "n/a", ""]:
        return None
    
    # Remove common prefixes and clean the text
    text = re.sub(r'(ask|asking|seeking|raising|looking for|need|required|funding|investment|valuation|revenue|worth|valued at)[\s:]*', '', text, flags=re.IGNORECASE)
    text = text.strip()
    
    # Pattern to match currency amounts
    patterns = [
        r'[\$₹€£¥]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb]|million|billion|thousand|crore|lakh)',
        r'[\$₹€£¥]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb])',
        r'[\$₹€£¥]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
        r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb]|million|billion|thousand|crore|lakh)',
        r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*([kmb])'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                number = float(match.group(1).replace(',', ''))
                multiplier = match.group(2).lower() if len(match.groups()) > 1 else ''
                
                # Apply multipliers
                if multiplier in ['k', 'thousand']:
                    number *= 1000
                elif multiplier in ['m', 'million']:
                    number *= 1000000
                elif multiplier in ['b', 'billion']:
                    number *= 1000000000
                elif multiplier in ['crore']:
                    number *= 10000000
                elif multiplier in ['lakh']:
                    number *= 100000
                
                return number
            except (ValueError, IndexError):
                continue
    
    return None

def format_currency_display(amount, original_text=""):
    """Format currency amount for display"""
    if amount is None:
        return original_text if original_text else "Not mentioned"
    
    if amount >= 1000000000:
        return f"${amount/1000000000:.1f}B"
    elif amount >= 1000000:
        return f"${amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"${amount/1000:.0f}K"
    else:
        return f"${amount:,.0f}"

def validate_crm_data(crm_data):
    """Validate extracted CRM data"""
    validation_result = {"valid": True, "errors": [], "warnings": []}
    
    # Check for required fields
    required_fields = ["company_name"]
    for field in required_fields:
        if not crm_data.get(field) or crm_data[field].strip() == "":
            validation_result["valid"] = False
            validation_result["errors"].append(f"Missing required field: {field}")
    
    # Check for HTML content (indicates extraction issues)
    for field, value in crm_data.items():
        if value and isinstance(value, str):
            if '<' in value and '>' in value:
                validation_result["warnings"].append(f"HTML content detected in {field}")
            if 'div' in value.lower() or 'html' in value.lower():
                validation_result["warnings"].append(f"HTML tags detected in {field}")
    
    # Check financial fields
    financial_fields = ['ask_amount', 'revenue_amount', 'valuation_amount']
    for field in financial_fields:
        if field in crm_data and crm_data[field] is not None:
            try:
                float(crm_data[field])
            except (ValueError, TypeError):
                validation_result["warnings"].append(f"Invalid financial value in {field}")
    
    return validation_result

def extract_crm_structured_data(text):
    """Enhanced CRM data extraction with better HTML handling"""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)
        extraction_prompt = """
        You are a pitch deck analyzer. Extract ONLY the following information in plain text format.
        
        CRITICAL: Return ONLY plain text, no HTML, no markdown, no formatting.
        
        Extract these exact fields:
        company_name: [Company name]
        sector: [Industry sector]
        stage: [Funding stage]
        ask: [Funding amount requested]
        revenue: [Current revenue]
        valuation: [Company valuation]
        source: Pitch Deck Upload
        assign: [Founder names and roles]
        description: [Business description in 2-3 sentences]
        
        Rules:
        - If information is not found, write "Not mentioned"
        - Use exact format: "field_name: value"
        - No HTML tags, no <div>, no formatting
        - Extract numbers with currency symbols (e.g., $2M, ₹5 Cr)
        - Keep descriptions concise and factual
        
        Example output:
        company_name: TechCorp
        sector: Fintech
        stage: Series A
        ask: $2M
        revenue: $500K ARR
        valuation: $10M pre-money
        source: Pitch Deck Upload
        assign: John Doe (CEO), Jane Smith (CTO)
        description: AI-powered financial platform for small businesses.
        """
        
        messages = [
            SystemMessage(content="You are a data extraction specialist. Return only plain text in the exact format requested. No HTML, no markdown, no additional formatting."),
            HumanMessage(content=f"{extraction_prompt}\n\nPitch Deck Content:\n{text}")
        ]
        
        response = llm.invoke(messages)
        
        # Clean the response to remove any HTML tags
        clean_response = re.sub(r'<[^>]+>', '', response.content)
        clean_response = re.sub(r'```[^`]*```', '', clean_response)  # Remove code blocks
        clean_response = clean_response.strip()
        
        # Validate the response format
        if '<div' in clean_response or '<html' in clean_response:
            logger.warning("LLM returned HTML content, attempting to clean...")
            # Extract text from HTML if present
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(clean_response, 'html.parser')
                clean_response = soup.get_text()
            except ImportError:
                # If BeautifulSoup is not available, use regex to clean
                clean_response = re.sub(r'<[^>]*>', '', clean_response)
        
        return clean_response
        
    except Exception as e:
        logger.error(f"Error extracting CRM structured data: {e}")
        return """company_name: Not mentioned
sector: Not mentioned
stage: Not mentioned
ask: Not mentioned
revenue: Not mentioned
valuation: Not mentioned
source: Pitch Deck Upload
assign: Not mentioned
description: Not mentioned"""

def parse_crm_data(structured_text):
    """Enhanced CRM data parser with better validation"""
    crm_data = {}
    
    # Clean the input text first
    structured_text = re.sub(r'<[^>]+>', '', structured_text)  # Remove HTML tags
    structured_text = re.sub(r'```[^`]*```', '', structured_text)  # Remove code blocks
    
    lines = structured_text.split('\n')
    
    # Expected fields
    expected_fields = ['company_name', 'sector', 'stage', 'ask', 'revenue', 'valuation', 'source', 'assign', 'description']
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            # Only process expected fields
            if key in expected_fields:
                if value and value.lower() not in ["not mentioned", "not found", "n/a", "", "null"]:
                    crm_data[key] = value
                else:
                    crm_data[key] = ""
    
    # Ensure all expected fields exist
    for field in expected_fields:
        if field not in crm_data:
            crm_data[field] = ""
    
    # Smart parsing for financial fields
    financial_fields = ['ask', 'revenue', 'valuation']
    for field in financial_fields:
        if field in crm_data and crm_data[field]:
            # Store original text for display
            crm_data[f"{field}_display"] = crm_data[field]
            # Parse numerical value for CRM
            parsed_amount = parse_currency_amount(crm_data[field])
            crm_data[f"{field}_amount"] = parsed_amount
        else:
            crm_data[f"{field}_display"] = "Not mentioned"
            crm_data[f"{field}_amount"] = None
    
    return crm_data

def format_crm_data_for_zoho(crm_data):
    """Enhanced CRM data formatting for Zoho with validation"""
    
    # Validate required fields
    required_fields = ["company_name"]
    for field in required_fields:
        if not crm_data.get(field):
            logger.warning(f"Missing required field: {field}")
    
    # Clean and format data
    formatted_data = {
        "company_name": crm_data.get("company_name", "Unknown Company").strip(),
        "sector": crm_data.get("sector", "").strip(),
        "stage": crm_data.get("stage", "").strip(),
        "description": crm_data.get("description", "").strip(),
        "source": "Pitch Deck Upload",
        "assign": crm_data.get("assign", "").strip(),
        
        # Financial amounts (converted to float for Zoho)
        "ask_amount": float(crm_data.get("ask_amount", 0)) if crm_data.get("ask_amount") else 0,
        "revenue_amount": float(crm_data.get("revenue_amount", 0)) if crm_data.get("revenue_amount") else 0,
        "valuation_amount": float(crm_data.get("valuation_amount", 0)) if crm_data.get("valuation_amount") else 0,
        
        # Display versions
        "ask_display": crm_data.get("ask_display", "Not mentioned"),
        "revenue_display": crm_data.get("revenue_display", "Not mentioned"),
        "valuation_display": crm_data.get("valuation_display", "Not mentioned"),
    }
    
    return formatted_data

def send_to_zoho_webhook(crm_data):
    if not zoho_webhook_url:
        st.error("❌ ZOHO_WEBHOOK_URL not configured in .env file")
        return False

    try:
        # Remove the "data": [ ... ] wrapper
        zoho_payload = {
            "Company": crm_data.get("company_name", ""),
            "Industry": crm_data.get("sector", ""),
            "Stage": crm_data.get("stage", ""),
            "Description": crm_data.get("description", ""),
            "Lead_Source": crm_data.get("source", "Pitch Deck Upload"),
            "Owner": crm_data.get("assign", ""),
            "Funding_Ask": crm_data.get("ask_amount") or 0,
            "Current_Revenue": crm_data.get("revenue_amount") or 0,
            "Valuation": crm_data.get("valuation_amount") or 0,
            "Ask_Display": crm_data.get("ask_display", ""),
            "Revenue_Display": crm_data.get("revenue_display", ""),
            "Valuation_Display": crm_data.get("valuation_display", ""),
            "Upload_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Document_Type": "Pitch Deck",
            "Processing_Status": "Completed"
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        logger.info(f"Sending to Zoho Flow: {json.dumps(zoho_payload, indent=2)}")
        response = requests.post(
            zoho_webhook_url,
            json=zoho_payload,
            headers=headers,
            timeout=30
        )

        logger.info(f"Zoho response: {response.status_code} {response.text}")

        if response.status_code == 200:
            logger.info("✅ Successfully sent to Zoho Flow")
            return True
        else:
            logger.error(f"❌ Zoho webhook failed: {response.status_code} - {response.text}")
            st.error(f"❌ Zoho webhook failed: {response.status_code} - {response.text}")
            return False

    except requests.exceptions.Timeout:
        logger.error("❌ Zoho webhook timeout")
        st.error("❌ Zoho webhook timeout")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Zoho webhook connection error")
        st.error("❌ Zoho webhook connection error")
        return False
    except Exception as e:
        logger.error(f"❌ Zoho webhook error: {str(e)}")
        st.error(f"❌ Zoho webhook error: {str(e)}")
        return False

def display_zoho_status(webhook_success, crm_data):
    """Display Zoho integration status with details"""
    if webhook_success:
        st.success("✅ Data successfully sent to Zoho CRM!")
        
        # Show what was sent
        with st.expander("📋 View data sent to Zoho CRM"):
            st.json({
                "Company": crm_data.get("company_name", ""),
                "Industry": crm_data.get("sector", ""),
                "Stage": crm_data.get("stage", ""),
                "Funding_Ask": f"${crm_data.get('ask_amount', 0):,.0f}" if crm_data.get('ask_amount') else "Not specified",
                "Revenue": f"${crm_data.get('revenue_amount', 0):,.0f}" if crm_data.get('revenue_amount') else "Not specified",
                "Valuation": f"${crm_data.get('valuation_amount', 0):,.0f}" if crm_data.get('valuation_amount') else "Not specified",
                "Lead_Source": "Pitch Deck Upload",
                "Upload_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    else:
        st.error("❌ Failed to send data to Zoho CRM")
        st.warning("Check your webhook URL and internet connection")

def test_zoho_connection():
    """Test Zoho webhook connection"""
    test_data = {
        "company_name": "Test Company",
        "sector": "Technology",
        "stage": "Seed",
        "description": "Test webhook connection",
        "source": "Test",
        "assign": "Test User",
        "ask_amount": 1000000,
        "revenue_amount": 500000,
        "valuation_amount": 5000000,
        "ask_display": "$1M",
        "revenue_display": "$500K",
        "valuation_display": "$5M"
    }
    
    return send_to_zoho_webhook(test_data)

def is_specific_crm_query(query):
    """Enhanced query detection for CRM fields"""
    query_lower = query.lower()
    specific_keywords = {
        'ask': ['ask', 'funding', 'investment', 'raise', 'capital', 'seeking', 'round', 'money needed'],
        'founder': ['founder', 'ceo', 'team', 'who founded', 'leadership', 'management', 'founders'],
        'revenue': ['revenue', 'sales', 'income', 'earnings', 'arr', 'mrr', 'turnover', 'gmv'],
        'valuation': ['valuation', 'worth', 'valued', 'pre-money', 'post-money', 'company value'],
        'company': ['company name', 'what company', 'name of company', 'startup name'],
        'sector': ['sector', 'industry', 'vertical', 'domain', 'space'],
        'stage': ['stage', 'bootstrapped', 'seed', 'series', 'early', 'growth', 'phase'],
        'description': ['what do they do', 'what does the company do', 'business', 'product', 'service', 'about']
    }
    
    for field, keywords in specific_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return field
    return None

def generate_crm_response(field, crm_data):
    """Generate intelligent responses for CRM queries"""
    if not crm_data:
        return "No CRM data available. Please upload a pitch deck first."
    
    field_mapping = {
        'company': 'company_name',
        'founder': 'assign',
        'ask': 'ask_display',
        'revenue': 'revenue_display',
        'valuation': 'valuation_display',
        'sector': 'sector',
        'stage': 'stage',
        'description': 'description'
    }
    
    crm_field = field_mapping.get(field, field)
    value = crm_data.get(crm_field, "")
    
    if not value:
        return f"**{field.title()}:** Not mentioned in the pitch deck."
    
    # Add context for financial fields
    if field in ['ask', 'revenue', 'valuation']:
        amount = crm_data.get(f"{field}_amount")
        if amount:
            formatted = format_currency_display(amount, value)
            return f"**{field.title()}:** {formatted}"
    
    return f"**{field.title()}:** {value}"

def extract_comprehensive_analysis(text):
    """Enhanced comprehensive analysis with smarter insights"""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.1)
        analysis_prompt = """
        You are a top-tier VC analyst. Provide a comprehensive analysis of this pitch deck with actionable insights.

        ## EXECUTIVE SUMMARY
        - Investment thesis and opportunity overview
        - Key strengths and major concerns
        - Overall investment recommendation with rationale

        ## COMPANY OVERVIEW
        - Business model and value proposition analysis
        - Market positioning and competitive advantages
        - Stage assessment and readiness for investment

        ## FOUNDERS & TEAM ANALYSIS
        - Founder backgrounds and relevant experience
        - Team composition and capability assessment
        - Key hiring needs and talent gaps

        ## MARKET OPPORTUNITY
        - TAM/SAM/SOM analysis and market sizing
        - Market timing and growth trends
        - Competitive landscape and differentiation

        ## PRODUCT & TECHNOLOGY
        - Product differentiation and unique value proposition
        - Technology moat and IP considerations
        - Product-market fit evidence and validation

        ## TRACTION & METRICS
        - Revenue growth and key metrics analysis
        - Customer acquisition and retention analysis
        - Milestone achievements and future projections

        ## FINANCIAL ANALYSIS
        - Unit economics and scalability assessment
        - Burn rate and runway analysis
        - Financial projections and assumptions review

        ## INVESTMENT TERMS
        - Valuation analysis and justification
        - Deal structure and terms assessment
        - Use of funds and ROI projections

        ## RISK ASSESSMENT
        - Market and competitive risks
        - Execution and team risks
        - Financial and operational risks

        ## RECOMMENDATIONS
        - Investment decision rationale
        - Key due diligence questions
        - Next steps and action items

        Provide specific, actionable insights with exact numbers and metrics from the deck.
        """
        
        messages = [
            SystemMessage(content="You are a senior VC analyst with expertise in evaluating startup investments. Provide detailed, actionable insights for investment decisions."),
            HumanMessage(content=f"{analysis_prompt}\n\nPitch Deck Content:\n{text}")
        ]
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating comprehensive analysis: {e}")
        return "Error generating comprehensive analysis"

def search_serpapi(query):
    """Enhanced search functionality with better result processing"""
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
            
            # Process with LLM for better insights
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
            search_prompt = f"Provide a comprehensive summary of these search results for VC analysis:\n{combined}"
            
            return llm.invoke(search_prompt).content.strip()
        
        return f"❌ SERP API error: {r.status_code}"
    except Exception as e:
        return f"❌ Web search failed: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Perpendo — Smart VC Pitch Evaluator", page_icon="📊", layout="wide")
st.title("📊 Perpendo — Smart VC Pitch Evaluator")

# Enhanced sidebar
with st.sidebar:
    st.header("🔧 Zoho Integration")
    if st.button("🧪 Test Zoho Connection"):
        with st.spinner("Testing Zoho webhook..."):
            test_result = test_zoho_connection()
            if test_result:
                st.success("✅ Zoho connection successful!")
            else:
                st.error("❌ Zoho connection failed!")
    
    # Show webhook status
    if zoho_webhook_url:
        st.success("🔗 Zoho webhook configured")
    else:
        st.error("❌ Zoho webhook not configured")
    
    st.header("📋 Detected Sections")
    if st.session_state.sections:
        for section_name in st.session_state.sections.keys():
            if st.button(section_name, key=f"section_{section_name}"):
                st.session_state.selected_section = section_name
    
    if st.session_state.file_uploaded:
        st.header("📊 Document Stats")
        st.metric("Total sections", len(st.session_state.sections))
        st.metric("Document length", f"{len(st.session_state.parsed_doc):,} chars")
    
    if st.session_state.crm_data:
        st.header("🔗 CRM Quick View")
        crm_fields = ['company_name', 'sector', 'stage', 'ask_display', 'revenue_display', 'valuation_display']
        for field in crm_fields:
            if field in st.session_state.crm_data and st.session_state.crm_data[field]:
                display_value = st.session_state.crm_data[field]
                if len(display_value) > 40:
                    display_value = display_value[:40] + "..."
                st.write(f"**{field.replace('_', ' ').title()}:** {display_value}")
        
        if st.button("📤 Export CRM Data"):
            st.json(st.session_state.crm_data)
    
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.selected_chat_index = None
            st.rerun()
        
        for i, (user, bot, timestamp) in enumerate(reversed(st.session_state.chat_history)):
            chat_index = len(st.session_state.chat_history) - 1 - i
            if st.button(f"Q{chat_index + 1}: {user[:25]}...", key=f"chat_{chat_index}"):
                st.session_state.selected_chat_index = chat_index
                st.rerun()

# Main content area
if st.session_state.structured_data:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Comprehensive Analysis", use_container_width=True):
            st.session_state.show_comprehensive_analysis = True
    with col2:
        if st.button("📊 CRM Data Summary", use_container_width=True):
            st.session_state.show_crm_summary = True

# SINGLE File upload section
st.header("📄 Upload Pitch Deck")
uploaded_file = st.file_uploader("Drop your PDF pitch deck here", type=["pdf"])

if uploaded_file:
    with st.spinner("🔄 Processing pitch deck..."):
        file_bytes = uploaded_file.read()
        text = extract_pdf_text(file_bytes)
        st.session_state.parsed_doc = text
        st.session_state.sections = split_sections(text)
        st.session_state.file_uploaded = True

        with st.spinner("🧠 Extracting CRM data with AI..."):
            crm_structured_text = extract_crm_structured_data(text)
            st.session_state.structured_data = crm_structured_text
            st.session_state.crm_data = parse_crm_data(crm_structured_text)
            
            # Validate the extracted data
            validation = validate_crm_data(st.session_state.crm_data)
            
            if not validation["valid"]:
                st.error("⚠️ Data extraction issues detected")

                for error in validation["errors"]:
                    st.error(f"❌ {error}")
            
            if validation["warnings"]:
                st.warning("⚠️ Data quality warnings:")
                for warning in validation["warnings"]:
                    st.warning(f"⚠️ {warning}")

        # Auto-send to Zoho CRM
        if st.session_state.crm_data and st.session_state.crm_data.get("company_name"):
            with st.spinner("📤 Sending data to Zoho CRM..."):
                zoho_success = send_to_zoho_webhook(st.session_state.crm_data)
                display_zoho_status(zoho_success, st.session_state.crm_data)

    st.success("✅ Pitch deck processed successfully!")
    
    # Display extracted sections
    if st.session_state.sections:
        st.subheader("📋 Extracted Sections")
        for section_name, section_content in st.session_state.sections.items():
            if section_content.strip():
                with st.expander(f"📁 {section_name}"):
                    st.write(section_content[:500] + "..." if len(section_content) > 500 else section_content)

# CRM Data Summary Display
if st.session_state.show_crm_summary and st.session_state.crm_data:
    st.header("📊 CRM Data Summary")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Company", st.session_state.crm_data.get("company_name", "Not mentioned"))
        st.metric("Sector", st.session_state.crm_data.get("sector", "Not mentioned"))
        st.metric("Stage", st.session_state.crm_data.get("stage", "Not mentioned"))
    
    with col2:
        ask_amount = st.session_state.crm_data.get("ask_amount")
        ask_display = format_currency_display(ask_amount, st.session_state.crm_data.get("ask_display", ""))
        st.metric("Funding Ask", ask_display)
        
        revenue_amount = st.session_state.crm_data.get("revenue_amount")
        revenue_display = format_currency_display(revenue_amount, st.session_state.crm_data.get("revenue_display", ""))
        st.metric("Revenue", revenue_display)
    
    with col3:
        valuation_amount = st.session_state.crm_data.get("valuation_amount")
        valuation_display = format_currency_display(valuation_amount, st.session_state.crm_data.get("valuation_display", ""))
        st.metric("Valuation", valuation_display)
    
    # Team and Description
    st.subheader("👥 Team")
    st.write(st.session_state.crm_data.get("assign", "Not mentioned"))
    
    st.subheader("📝 Description")
    st.write(st.session_state.crm_data.get("description", "Not mentioned"))
    
    # Raw CRM data for debugging
    with st.expander("🔍 Raw CRM Data (Debug)"):
        st.json(st.session_state.crm_data)
    
    if st.button("❌ Close CRM Summary"):
        st.session_state.show_crm_summary = False
        st.rerun()

# Comprehensive Analysis Display
if st.session_state.show_comprehensive_analysis and st.session_state.parsed_doc:
    st.header("🔍 Comprehensive Analysis")
    
    with st.spinner("🧠 Generating comprehensive analysis..."):
        analysis = extract_comprehensive_analysis(st.session_state.parsed_doc)
        st.markdown(analysis)
    
    if st.button("❌ Close Analysis"):
        st.session_state.show_comprehensive_analysis = False
        st.rerun()

# Selected Section Display
if st.session_state.selected_section and st.session_state.sections:
    st.header(f"📁 {st.session_state.selected_section}")
    section_content = st.session_state.sections.get(st.session_state.selected_section, "")
    st.write(section_content)
    
    if st.button("❌ Close Section"):
        st.session_state.selected_section = None
        st.rerun()

# Chat History Display
if st.session_state.selected_chat_index is not None and st.session_state.chat_history:
    chat_index = st.session_state.selected_chat_index
    if 0 <= chat_index < len(st.session_state.chat_history):
        user_msg, bot_msg, timestamp = st.session_state.chat_history[chat_index]
        
        st.header(f"💬 Chat History - Q{chat_index + 1}")
        st.info(f"**Question:** {user_msg}")
        st.success(f"**Answer:** {bot_msg}")
        st.caption(f"Asked at: {timestamp}")
        
        if st.button("❌ Close Chat"):
            st.session_state.selected_chat_index = None
            st.rerun()

# Chat Interface
if st.session_state.file_uploaded:
    st.header("💬 Ask Questions About the Pitch Deck")
    
    # Enhanced quick questions
    st.subheader("🚀 Quick Questions")
    quick_questions = [
        "What is the funding ask amount?",
        "Who are the founders?",
        "What is the current revenue?",
        "What is the company valuation?",
        "What sector does the company operate in?",
        "What stage is the company in?",
        "What does the company do?",
        "What is the market size?",
        "Who are the competitors?",
        "What is the business model?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}"):
                # Process quick question
                crm_field = is_specific_crm_query(question)
                if crm_field and st.session_state.crm_data:
                    response = generate_crm_response(crm_field, st.session_state.crm_data)
                else:
                    # Use LLM for complex questions
                    try:
                        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
                        messages = [
                            SystemMessage(content="You are a helpful assistant analyzing a pitch deck. Provide clear, concise answers based on the document content."),
                            HumanMessage(content=f"Question: {question}\n\nDocument: {st.session_state.parsed_doc}")
                        ]
                        response = llm.invoke(messages).content
                    except Exception as e:
                        response = f"Error processing question: {str(e)}"
                
                # Add to chat history
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append((question, response, timestamp))
                st.rerun()

    # Custom question input
    user_question = st.text_input("💭 Ask a custom question:", placeholder="e.g., What are the key risks for this investment?")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔍 Get Answer", use_container_width=True):
            if user_question:
                with st.spinner("🧠 Analyzing..."):
                    # Check if it's a specific CRM query
                    crm_field = is_specific_crm_query(user_question)
                    if crm_field and st.session_state.crm_data:
                        response = generate_crm_response(crm_field, st.session_state.crm_data)
                    else:
                        # Use LLM for complex analysis
                        try:
                            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
                            messages = [
                                SystemMessage(content="You are a senior VC analyst. Provide detailed, actionable insights based on the pitch deck content."),
                                HumanMessage(content=f"Question: {user_question}\n\nPitch Deck Content: {st.session_state.parsed_doc}")
                            ]
                            response = llm.invoke(messages).content
                        except Exception as e:
                            response = f"Error processing question: {str(e)}"
                    
                    # Add to chat history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.chat_history.append((user_question, response, timestamp))
                    st.rerun()
    
    with col2:
        if st.button("🌐 Web Search", use_container_width=True):
            if user_question:
                with st.spinner("🔍 Searching web..."):
                    # Enhance query with company context
                    company_name = st.session_state.crm_data.get("company_name", "") if st.session_state.crm_data else ""
                    sector = st.session_state.crm_data.get("sector", "") if st.session_state.crm_data else ""
                    
                    search_query = user_question
                    if company_name:
                        search_query += f" {company_name}"
                    if sector:
                        search_query += f" {sector}"
                    
                    search_result = search_serpapi(search_query)
                    
                    # Add to chat history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.chat_history.append((f"🌐 {user_question}", search_result, timestamp))
                    st.rerun()

    # Display latest chat response
    if st.session_state.chat_history:
        latest_question, latest_response, latest_timestamp = st.session_state.chat_history[-1]
        st.subheader("💬 Latest Response")
        st.info(f"**Q:** {latest_question}")
        st.success(f"**A:** {latest_response}")
        st.caption(f"Asked at: {latest_timestamp}")

# Footer
st.markdown("---")
st.markdown("**Perpendo — Smart VC Pitch Evaluator** | Powered by OpenAI GPT-4o & Streamlit")
st.markdown("🔗 [GitHub](https://github.com/your-repo) | 📧 [Support](mailto:support@perpendo.com)")

# Add some styling
st.markdown("""
<style>
    .stMetric > div > div > div > div {
        font-size: 20px;
    }
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #FF5252;
    }
    .stExpander {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)
