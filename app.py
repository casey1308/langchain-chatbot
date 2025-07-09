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
    """Enhanced CRM data extraction with intelligent parsing"""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)
        extraction_prompt = """
        You are the world's most advanced pitch deck parser. Analyze this pitch deck and extract ONLY the following information in exact key-value format.
        Look through ALL the text meticulously and extract specific numbers, amounts, and concrete details.

        CRITICAL INSTRUCTIONS:
        1. Use these EXACT keys and provide specific values or "Not mentioned"
        2. For financial data (ask, revenue, valuation), extract the EXACT amount mentioned
        3. Include currency symbols and units (K, M, B) as they appear
        4. For ask: specify the round type if mentioned (Pre-seed, Seed, Series A, etc.)
        5. For revenue: specify the type (ARR, MRR, total revenue, GMV, etc.)
        6. For valuation: specify if it's pre-money, post-money, or current valuation
        7. For stage: infer from funding round, company maturity, or explicit mention
        8. For sector: be specific (e.g., "Fintech - Digital Payments", "Healthtech - Telemedicine")
        9. For assign: include ALL founder names with their roles
        10. For description: provide a clear, concise 2-3 sentence business description

        REQUIRED OUTPUT FORMAT:
        company_name: [Extract exact company name]
        sector: [Specific industry/sector with sub-category if applicable]
        stage: [Startup stage - Pre-seed/Seed/Series A/Series B/Growth/etc.]
        ask: [Funding amount with round type, e.g., "$2M Seed Round"]
        revenue: [Revenue figures with type, e.g., "$500K ARR" or "$2M GMV"]
        valuation: [Valuation with type, e.g., "$5M pre-money valuation"]
        source: [Always "Pitch Deck Upload"]
        assign: [Founder names with roles, e.g., "John Doe (CEO), Jane Smith (CTO)"]
        description: [Clear business description in 2-3 sentences]

        FINANCIAL DATA EXTRACTION RULES:
        - Extract EXACT numbers with currency symbols
        - Preserve units (K, M, B) as mentioned
        - For ask: include round stage and total amount
        - For revenue: include timeframe and type (monthly/annual)
        - For valuation: include pre/post money specification
        - If multiple numbers exist, prioritize the most recent/current

        EXAMPLES:
        ask: "$1.5M Seed Round"
        revenue: "$200K ARR"
        valuation: "$8M pre-money valuation"
        assign: "Sarah Johnson (CEO), Michael Chen (CTO), Lisa Rodriguez (CMO)"

        Pitch Deck Content:
        """
        
        messages = [
            SystemMessage(content="You are an expert at extracting CRM-specific structured data from pitch decks. You have analyzed thousands of pitch decks and can identify patterns, implicit information, and extract precise financial data. Focus on accuracy and completeness."),
            HumanMessage(content=f"{extraction_prompt}\n\n{text}")
        ]
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error extracting CRM structured data: {e}")
        return "Error extracting structured data"

def parse_crm_data(structured_text):
    """Enhanced CRM data parser with intelligent type conversion"""
    crm_data = {}
    lines = structured_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key in ['company_name', 'sector', 'stage', 'ask', 'revenue', 'valuation', 'source', 'assign', 'description']:
                if value and value.lower() not in ["not mentioned", "not found", "n/a", ""]:
                    crm_data[key] = value
                else:
                    crm_data[key] = ""
    
    # Smart parsing for financial fields
    financial_fields = ['ask', 'revenue', 'valuation']
    for field in financial_fields:
        if field in crm_data and crm_data[field]:
            # Store original text for display
            crm_data[f"{field}_display"] = crm_data[field]
            # Parse numerical value for CRM
            parsed_amount = parse_currency_amount(crm_data[field])
            crm_data[f"{field}_amount"] = parsed_amount
    
    return crm_data

def format_crm_data_for_zoho(crm_data):
    """Format CRM data specifically for Zoho CRM requirements"""
    zoho_payload = {
        "company_name": crm_data.get("company_name", ""),
        "sector": crm_data.get("sector", ""),
        "stage": crm_data.get("stage", ""),
        "description": crm_data.get("description", ""),
        "source": crm_data.get("source", "Pitch Deck Upload"),
        "assign": crm_data.get("assign", ""),
        
        # Financial fields as doubles for Zoho CRM
        "ask": crm_data.get("ask_amount"),  # Double value
        "revenue": crm_data.get("revenue_amount"),  # Double value  
        "valuation": crm_data.get("valuation_amount"),  # Double value
        
        # Display versions for reference
        "ask_display": crm_data.get("ask_display", ""),
        "revenue_display": crm_data.get("revenue_display", ""),
        "valuation_display": crm_data.get("valuation_display", ""),
    }
    
    # Clean up None values
    return {k: v for k, v in zoho_payload.items() if v is not None}

def send_to_zoho_webhook(crm_data):
    """Send formatted CRM data to Zoho webhook"""
    if not zoho_webhook_url:
        logger.warning("❌ ZOHO_WEBHOOK_URL not set in .env")
        return False
    
    try:
        zoho_payload = format_crm_data_for_zoho(crm_data)
        headers = {"Content-Type": "application/json"}
        
        logger.info(f"Sending payload to Zoho: {json.dumps(zoho_payload, indent=2)}")
        response = requests.post(zoho_webhook_url, json=zoho_payload, headers=headers)
        
        if response.status_code == 200:
            logger.info("✅ CRM data sent to Zoho Flow successfully")
            return True
        else:
            logger.warning(f"⚠️ Webhook error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to send to Zoho webhook: {e}")
        return False

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
        
        # Send to Zoho webhook
        webhook_success = send_to_zoho_webhook(st.session_state.crm_data)
        webhook_status = "✅ Sent to CRM" if webhook_success else "⚠️ CRM sync failed"
    
    st.success(f"✅ Pitch deck processed successfully! {webhook_status}")

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
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .crm-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }}
        .crm-field {{
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem;
            backdrop-filter: blur(10px);
        }}
        .crm-label {{
            font-size: 0.9rem;
            font-weight: 600;
            opacity: 0.8;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .crm-value {{
            font-size: 1.2rem;
            font-weight: 600;
            line-height: 1.4;
        }}
        .crm-highlight {{
            background: linear-gradient(90deg, #ffeaa7, #fab1a0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }}
        </style>
        <div class="crm-card">
            <h2 class="crm-highlight">{company_name}</h2>
            <div class="crm-grid">
                <div class="crm-field">
                    <div class="crm-label">Sector</div>
                    <div class="crm-value">{sector}</div>
                </div>
                <div class="crm-field">
                    <div class="crm-label">Stage</div>
                    <div class="crm-value">{stage}</div>
                </div>
                <div class="crm-field">
                    <div class="crm-label">Funding Ask</div>
                    <div class="crm-value">{ask_display}</div>
                </div>
                <div class="crm-field">
                    <div class="crm-label">Revenue</div>
                    <div class="crm-value">{revenue_display}</div>
                </div>
                <div class="crm-field">
                    <div class="crm-label">Valuation</div>
                    <div class="crm-value">{valuation_display}</div>
                </div>
                <div class="crm-field">
                    <div class="crm-label">Key Team</div>
                    <div class="crm-value">{assign}</div>
                </div>
                <div class="crm-field" style="grid-column: 1 / -1;">
                    <div class="crm-label">Description</div>
                    <div class="crm-value">{description}</div>
                </div>
                <div class="crm-field">
                    <div class="crm-label">Source</div>
                    <div class="crm-value">{source}</div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Show financial data parsing details
        if any(st.session_state.crm_data.get(f"{field}_amount") for field in ['ask', 'revenue', 'valuation']):
            st.markdown("#### 💰 Parsed Financial Data")
            fin_cols = st.columns(3)
            for i, field in enumerate(['ask', 'revenue', 'valuation']):
                amount = st.session_state.crm_data.get(f"{field}_amount")
                if amount:
                    with fin_cols[i]:
                        st.metric(
                            label=field.title(),
                            value=format_currency_display(amount),
                            delta=f"${amount:,.0f}" if amount >= 1000 else f"${amount:.2f}"
                        )

# Chat interface
if st.session_state.file_uploaded:
    st.markdown("### 💬 Intelligent Q&A")
    
    # Quick action buttons
    st.markdown("**Quick CRM Queries:**")
    quick_cols = st.columns(6)
    quick_queries = [
        ("💰 Funding Ask", "What is their funding ask?"),
        ("👥 Founders", "Who are the founders?"),
        ("💵 Revenue", "What is their revenue?"),
        ("🏢 Valuation", "What is their valuation?"),
        ("🏭 Sector", "What sector are they in?"),
        ("🚀 Stage", "What stage are they at?")
    ]
    
    for i, (label, query) in enumerate(quick_queries):
        with quick_cols[i]:
            if st.button(label, key=f"quick_{i}"):
                st.session_state.auto_query = query

    # Handle auto queries
    if hasattr(st.session_state, 'auto_query') and st.session_state.auto_query:
        user_query = st.session_state.auto_query
        st.session_state.auto_query = None
    else:
        user_query = st.chat_input("Ask anything about this pitch deck...")

    if user_query:
        st.session_state.selected_chat_index = None
        specific_field = is_specific_crm_query(user_query)
        
        if specific_field and st.session_state.crm_data:
           # Quick CRM response
           response = generate_crm_response(specific_field, st.session_state.crm_data)
       else:
           # General AI response
           with st.spinner("🧠 Analyzing..."):
               try:
                   llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.3)
                   
                   # Check if it's a web search query
                   web_indicators = ["latest", "recent", "current", "news", "market", "competitor", "funding", "valuation"]
                   if any(indicator in user_query.lower() for indicator in web_indicators):
                       search_results = search_serpapi(user_query)
                       context = f"Pitch Deck Content:\n{st.session_state.parsed_doc}\n\nWeb Search Results:\n{search_results}"
                   else:
                       context = st.session_state.parsed_doc
                   
                   messages = [
                       SystemMessage(content="You are a top-tier VC analyst. Provide detailed, actionable insights based on the pitch deck content. Use specific numbers and metrics when available."),
                       HumanMessage(content=f"Query: {user_query}\n\nContext: {context}")
                   ]
                   
                   response = llm.invoke(messages).content
               except Exception as e:
                   response = f"❌ Error processing query: {str(e)}"
       
       # Add to chat history
       timestamp = datetime.now().strftime("%H:%M:%S")
       st.session_state.chat_history.append((user_query, response, timestamp))
       
       # Display response
       st.markdown(f"**You:** {user_query}")
       st.markdown(f"**AI:** {response}")

   # Display selected chat from history
   if st.session_state.selected_chat_index is not None:
       idx = st.session_state.selected_chat_index
       if idx < len(st.session_state.chat_history):
           user_q, bot_r, ts = st.session_state.chat_history[idx]
           st.markdown(f"**Previous Query ({ts}):** {user_q}")
           st.markdown(f"**Response:** {bot_r}")

# Show comprehensive analysis
if hasattr(st.session_state, 'show_comprehensive_analysis') and st.session_state.show_comprehensive_analysis:
   st.markdown("### 📊 Comprehensive VC Analysis")
   with st.spinner("🔍 Generating comprehensive analysis..."):
       analysis = extract_comprehensive_analysis(st.session_state.parsed_doc)
       st.markdown(analysis)
   st.session_state.show_comprehensive_analysis = False

# Show CRM summary
if hasattr(st.session_state, 'show_crm_summary') and st.session_state.show_crm_summary:
   st.markdown("### 🔗 CRM Data Summary")
   if st.session_state.crm_data:
       st.json(st.session_state.crm_data)
   else:
       st.warning("No CRM data available. Please upload a pitch deck first.")
   st.session_state.show_crm_summary = False

# Footer
st.markdown("---")
st.markdown("**Perpendo** - Smart VC Pitch Evaluator | Built with ❤️ for VCs")
