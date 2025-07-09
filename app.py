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
    # Matches: $1M, $1.5M, $2 million, 1.5 million, $500K, $5,000,000, etc.
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
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(clean_response, 'html.parser')
            clean_response = soup.get_text()
        
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


# Updated file processing section
uploaded_file = st.file_uploader("Upload your pitch deck (PDF)", type=["pdf"])

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
                st.error("⚠️ Data extraction issues detected:")
                for error in validation["errors"]:
                    st.error(f"- {error}")
                with st.expander("🔍 Debug: Raw Extraction"):
                    st.code(crm_structured_text)
                st.warning("Retrying with simplified extraction...")
                retry_prompt = f"""
                Extract key information from this pitch deck in simple format:

                Company: [company name]
                Industry: [industry/sector]
                Stage: [funding stage]
                Funding: [amount requested]
                Revenue: [current revenue]
                Value: [valuation]
                Founders: [founder names]

                Text: {text[:5000]}
                """
                try:
                    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)
                    retry_response = llm.invoke(retry_prompt)
                    st.info("Retry extraction:")
                    st.code(retry_response.content)
                except Exception as e:
                    st.error(f"Retry failed: {e}")
            
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    st.warning(f"⚠️ {warning}")
        
        # Only proceed with Zoho if validation passed
        if validation["valid"]:
            with st.spinner("🔗 Sending to Zoho CRM..."):
                formatted_crm_data = format_crm_data_for_zoho(st.session_state.crm_data)
                webhook_success = send_to_zoho_webhook(formatted_crm_data)
                display_zoho_status(webhook_success, st.session_state.crm_data)
        
        st.success("✅ Pitch deck processed successfully!")


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
    """Enhanced Zoho webhook integration with better error handling"""
    if not zoho_webhook_url:
        st.error("❌ ZOHO_WEBHOOK_URL not configured in .env file")
        return False
    
    try:
        # Format data for Zoho CRM
        zoho_payload = {
            "data": [{
                "Company": crm_data.get("company_name", ""),
                "Industry": crm_data.get("sector", ""),
                "Stage": crm_data.get("stage", ""),
                "Description": crm_data.get("description", ""),
                "Lead_Source": crm_data.get("source", "Pitch Deck Upload"),
                "Owner": crm_data.get("assign", ""),
                
                # Financial fields (as numbers for Zoho)
                "Funding_Ask": crm_data.get("ask_amount") or 0,
                "Current_Revenue": crm_data.get("revenue_amount") or 0,
                "Valuation": crm_data.get("valuation_amount") or 0,
                
                # Display versions for notes
                "Ask_Display": crm_data.get("ask_display", ""),
                "Revenue_Display": crm_data.get("revenue_display", ""),
                "Valuation_Display": crm_data.get("valuation_display", ""),
                
                # Additional metadata
                "Upload_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Document_Type": "Pitch Deck",
                "Processing_Status": "Completed"
            }]
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        logger.info(f"Sending to Zoho Flow: {json.dumps(zoho_payload, indent=2)}")
        
        # Send with timeout
        response = requests.post(
            zoho_webhook_url, 
            json=zoho_payload, 
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            logger.info("✅ Successfully sent to Zoho Flow")
            return True
        else:
            logger.error(f"❌ Zoho webhook failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("❌ Zoho webhook timeout")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("❌ Zoho webhook connection error")
        return False
    except Exception as e:
        logger.error(f"❌ Zoho webhook error: {str(e)}")
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

# File upload section
st.header("📄 Upload Pitch Deck")
file = st.file_uploader("Drop your PDF pitch deck here", type=["pdf"])

if file:
    with st.spinner("🔄 Processing pitch deck..."):
        file_bytes = file.read()
        text = extract_pdf_text(file_bytes)
        st.session_state.parsed_doc = text
        st.session_state.sections = split_sections(text)
        st.session_state.file_uploaded = True

    with st.spinner("🧠 Extracting CRM data with AI..."):
        crm_structured_text = extract_crm_structured_data(text)
        st.session_state.structured_data = crm_structured_text
        st.session_state.crm_data = parse_crm_data(crm_structured_text)
    
    # Enhanced Zoho integration with better feedback
    with st.spinner("🔗 Sending to Zoho CRM..."):
        formatted_crm_data = format_crm_data_for_zoho(st.session_state.crm_data)
        webhook_success = send_to_zoho_webhook(formatted_crm_data)
        
        # Display integration status
        display_zoho_status(webhook_success, st.session_state.crm_data)
    
    st.success("✅ Pitch deck processed successfully!")

    # Enhanced CRM Data Preview Card
    if st.session_state.crm_data:
        st.markdown("### 🔗 CRM Integration Data")
        
        # Create enhanced preview card with better styling
        company_name = st.session_state.crm_data.get("company_name", "Unknown Company")
        sector = st.session_state.crm_data.get("sector", "Not specified")
        stage = st.session_state.crm_data.get("stage", "Not specified")
        ask_display = st.session_state.crm_data.get("ask_display", "Not mentioned")
        revenue_display = st.session_state.crm_data.get("revenue_display", "Not mentioned")
        valuation_display = st.session_state.crm_data.get("valuation_display", "Not mentioned")
        assign = st.session_state.crm_data.get("assign", "Not mentioned")
        description = st.session_state.crm_data.get("description", "No description available")
        source = st.session_state.crm_data.get("source", "Pitch Deck Upload")
        
        card_html = f"""
        <style>
        .crm-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .crm-card h2 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0 0 1rem 0;
        text-align: center;
    }}
    .crm-field {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.2);
    }}
    .crm-field:last-child {{
        border-bottom: none;
    }}
    .crm-label {{
        font-weight: 600;
        opacity: 0.9;
        flex: 1;
    }}
    .crm-value {{
        font-weight: 400;
        text-align: right;
        flex: 2;
        word-wrap: break-word;
    }}
    .crm-metrics {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }}
    .crm-metric {{
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }}
    .crm-metric-value {{
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    .crm-metric-label {{
        font-size: 0.9rem;
        opacity: 0.8;
    }}
</style>
<div class="crm-card">
    <h2>🏢 {company_name}</h2>
    <div class="crm-field">
        <div class="crm-label">Industry:</div>
        <div class="crm-value">{sector}</div>
    </div>
    <div class="crm-field">
        <div class="crm-label">Stage:</div>
        <div class="crm-value">{stage}</div>
    </div>
    <div class="crm-field">
        <div class="crm-label">Founders:</div>
        <div class="crm-value">{assign}</div>
    </div>
    <div class="crm-field">
        <div class="crm-label">Source:</div>
        <div class="crm-value">{source}</div>
    </div>
    
    <div class="crm-metrics">
        <div class="crm-metric">
            <div class="crm-metric-value">{ask_display}</div>
            <div class="crm-metric-label">Funding Ask</div>
        </div>
        <div class="crm-metric">
            <div class="crm-metric-value">{revenue_display}</div>
            <div class="crm-metric-label">Revenue</div>
        </div>
        <div class="crm-metric">
            <div class="crm-metric-value">{valuation_display}</div>
            <div class="crm-metric-label">Valuation</div>
        </div>
    </div>
    
    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
        <strong>Description:</strong><br>
        {description}
    </div>
</div>
"""

        st.markdown(card_html, unsafe_allow_html=True)

# Display comprehensive analysis if requested
if st.session_state.get('show_comprehensive_analysis', False):
    st.markdown("### 📊 Comprehensive VC Analysis")
    with st.spinner("🔍 Generating comprehensive analysis..."):
        analysis = extract_comprehensive_analysis(st.session_state.parsed_doc)
        st.markdown(analysis)
    st.session_state.show_comprehensive_analysis = False

# Display CRM summary if requested
if st.session_state.get('show_crm_summary', False):
    st.markdown("### 📋 CRM Data Summary")
    if st.session_state.crm_data:
        st.json(st.session_state.crm_data)
    else:
        st.warning("No CRM data available. Please upload a pitch deck first.")
    st.session_state.show_crm_summary = False

# Chat interface
if st.session_state.file_uploaded:
    st.markdown("### 💬 AI Chat Interface")
    
    # Display selected chat if any
    if st.session_state.selected_chat_index is not None:
        selected_chat = st.session_state.chat_history[st.session_state.selected_chat_index]
        st.markdown(f"**Previous Question:** {selected_chat[0]}")
        st.markdown(f"**Answer:** {selected_chat[1]}")
        st.markdown(f"**Time:** {selected_chat[2]}")
        if st.button("🔄 Ask New Question"):
            st.session_state.selected_chat_index = None
            st.rerun()
    
    # Chat input
  # Replace the problematic section around line 850-870 with this fixed version:

# Chat input using form (RECOMMENDED APPROACH)
if st.session_state.selected_chat_index is None:
    # Quick action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🚀 Quick Analysis", use_container_width=True):
            user_query = "Give me a quick analysis of this pitch deck"
            st.session_state.process_query = user_query
    with col2:
        if st.button("💰 Investment Potential", use_container_width=True):
            user_query = "What's the investment potential of this company?"
            st.session_state.process_query = user_query
    with col3:
        if st.button("⚠️ Risk Assessment", use_container_width=True):
            user_query = "What are the main risks with this investment?"
            st.session_state.process_query = user_query
    
    # Chat form
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("💭 Ask anything about this pitch deck:")
        submitted = st.form_submit_button("Send")
        
        if submitted and user_query:
            st.session_state.process_query = user_query
    
    # Process query if one exists
    if st.session_state.get('process_query'):
        user_query = st.session_state.process_query
        del st.session_state.process_query  # Clear it
        
        if user_query:
            with st.spinner("🤔 Analyzing your question..."):
                # Check if it's a specific CRM query
                crm_field = is_specific_crm_query(user_query)
                if crm_field and st.session_state.crm_data:
                    response = generate_crm_response(crm_field, st.session_state.crm_data)
                else:
                    # Check if web search is needed
                    search_keywords = ["market size", "competitors", "industry trends", "recent news", "company background"]
                    needs_search = any(keyword in user_query.lower() for keyword in search_keywords)
                    
                    if needs_search:
                        # Perform web search
                        company_name = st.session_state.crm_data.get("company_name", "")
                        sector = st.session_state.crm_data.get("sector", "")
                        search_query = f"{company_name} {sector} {user_query}"
                        search_results = search_serpapi(search_query)
                        
                        # Generate response with search results
                        try:
                            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
                            prompt = f"""
                            User Question: {user_query}
                            
                            Pitch Deck Content:
                            {st.session_state.parsed_doc}
                            
                            Web Search Results:
                            {search_results}
                            
                            Please provide a comprehensive answer combining insights from the pitch deck and web search results.
                            """
                            response = llm.invoke(prompt).content
                        except Exception as e:
                            response = f"Error generating response: {str(e)}"
                    else:
                        # Generate response from pitch deck only
                        try:
                            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
                            prompt = f"""
                            Based on this pitch deck content, please answer the user's question:
                            
                            Question: {user_query}
                            
                            Pitch Deck Content:
                            {st.session_state.parsed_doc}
                            
                            CRM Data:
                            {st.session_state.crm_data}
                            
                            Provide a detailed, insightful answer as a VC analyst would.
                            """
                            response = llm.invoke(prompt).content
                        except Exception as e:
                            response = f"Error generating response: {str(e)}"
                
                # Add to chat history
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.chat_history.append((user_query, response, timestamp))
                
                # Display response
                st.markdown(f"**Your Question:** {user_query}")
                st.markdown(f"**AI Response:** {response}")
                
                # Automatically rerun to clear the processed query
                st.rerun()
# Enhanced section viewer
if st.session_state.get('selected_section'):
    st.markdown(f"### 📑 Section: {st.session_state.selected_section}")
    section_content = st.session_state.sections[st.session_state.selected_section]
    st.markdown(section_content)
    
    # Add section-specific analysis
    if st.button("🔍 Analyze This Section"):
        with st.spinner("Analyzing section..."):
            try:
                llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
                analysis_prompt = f"""
                Provide a detailed analysis of this section from a VC perspective:
                
                Section: {st.session_state.selected_section}
                Content: {section_content}
                
                Focus on:
                - Key insights and takeaways
                - Strengths and weaknesses
                - Missing information
                - VC concerns or opportunities
                """
                analysis = llm.invoke(analysis_prompt).content
                st.markdown(f"**Analysis:** {analysis}")
            except Exception as e:
                st.error(f"Error analyzing section: {str(e)}")

# Footer with additional features
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.file_uploaded:
        if st.button("📊 Generate Investment Memo"):
            with st.spinner("Generating investment memo..."):
                try:
                    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
                    memo_prompt = f"""
                    Generate a professional investment memo based on this pitch deck:
                    
                    CRM Data: {st.session_state.crm_data}
                    Full Content: {st.session_state.parsed_doc}
                    
                    Format as a standard VC investment memo with:
                    - Executive Summary
                    - Investment Thesis
                    - Company Overview
                    - Market Analysis
                    - Team Assessment
                    - Financial Analysis
                    - Risk Assessment
                    - Recommendation
                    """
                    memo = llm.invoke(memo_prompt).content
                    st.markdown("### 📄 Investment Memo")
                    st.markdown(memo)
                except Exception as e:
                    st.error(f"Error generating memo: {str(e)}")

with col2:
    if st.session_state.file_uploaded:
        if st.button("📈 Generate Due Diligence Checklist"):
            with st.spinner("Generating due diligence checklist..."):
                try:
                    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
                    dd_prompt = f"""
                    Create a comprehensive due diligence checklist for this startup:
                    
                    Company: {st.session_state.crm_data.get('company_name', 'Unknown')}
                    Sector: {st.session_state.crm_data.get('sector', 'Unknown')}
                    Stage: {st.session_state.crm_data.get('stage', 'Unknown')}
                    
                    Include specific questions for:
                    - Legal and Corporate Structure
                    - Financial Due Diligence
                    - Market and Competition
                    - Technology and IP
                    - Team and HR
                    - Operational Due Diligence
                    - ESG and Compliance
                    """
                    checklist = llm.invoke(dd_prompt).content
                    st.markdown("### ✅ Due Diligence Checklist")
                    st.markdown(checklist)
                except Exception as e:
                    st.error(f"Error generating checklist: {str(e)}")

with col3:
    if st.session_state.file_uploaded:
        if st.button("📊 Export Full Report"):
            # Create comprehensive report
            report_data = {
                "company_info": st.session_state.crm_data,
                "sections": st.session_state.sections,
                "chat_history": st.session_state.chat_history,
                "timestamp": datetime.now().isoformat()
            }
            
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(report_data, indent=2),
                file_name=f"{st.session_state.crm_data.get('company_name', 'company')}_analysis_report.json",
                mime="application/json"
            )

# Initialize session state for new features
for key in ["show_comprehensive_analysis", "show_crm_summary", "selected_section"]:
    if key not in st.session_state:
        st.session_state[key] = False

# Application info
st.markdown("---")
st.markdown("**Perpendo** - Smart VC Pitch Evaluator | Powered by OpenAI GPT-4 & Zoho CRM Integration")
st.markdown("*Upload pitch decks, get instant AI analysis, and seamlessly integrate with your CRM workflow.*")
