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
import threading
import queue
import time

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
for key in ["chat_history", "parsed_doc", "file_uploaded", "sections", "structured_data", "selected_chat_index", "crm_data", "webhook_responses"]:
    if key not in st.session_state:
        if key == "chat_history":
            st.session_state[key] = []
        elif key == "webhook_responses":
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
def extract_crm_structured_data(text):
    """Extract CRM-specific structured data from pitch deck using LLM"""
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)
        
        extraction_prompt = """
        Analyze this pitch deck text and extract ONLY the following CRM-specific information in exact key-value format.
        Look through ALL the text carefully and extract specific numbers, amounts, and concrete details.
        
        CRITICAL: Use these EXACT keys and provide specific values or "Not mentioned":
        
        company_name: [Extract exact company name]
        ask: [Extract funding amount being sought, e.g., "$2M Series A" or "Not mentioned"]
        revenue: [Extract current revenue figures with specific numbers, e.g., "$500K ARR" or "Not mentioned"]
        valuation: [Extract current valuation with currency and amount, e.g., "$5M pre-money" or "Not mentioned"]
        source: [Always put "Pitch Deck Upload"]
        assign: [Extract founder names and key team members, e.g., "John Doe (CEO), Jane Smith (CTO)" or "Not mentioned"]
        description: [Brief 2-3 sentence description of what the company does]
        
        INSTRUCTIONS:
        1. Extract SPECIFIC numbers and amounts wherever possible
        2. Include currency symbols and units (K, M, B)
        3. Look for information across ALL sections of the document
        4. Be precise with valuation (pre-money/post-money distinction)
        5. For ask: include round type if mentioned (Seed, Series A, etc.)
        6. For revenue: include type if mentioned (ARR, MRR, total revenue)
        7. For assign: include founder names and key roles
        8. For description: keep it concise and business-focused
        9. Source should always be "Pitch Deck Upload"
        
        Text to analyze:
        """
        
        messages = [
            SystemMessage(content="You are an expert at extracting CRM-specific structured data from pitch decks. Focus on extracting exact numbers, amounts, and concrete details. Be thorough and look for information throughout the entire document."),
            HumanMessage(content=f"{extraction_prompt}\n\n{text}")
        ]
        
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        logger.error(f"Error extracting CRM structured data: {e}")
        return "Error extracting structured data"

# Parse CRM data into structured format
def parse_crm_data(structured_text):
    """Parse the structured text into a dictionary for CRM integration"""
    crm_data = {}
    lines = structured_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            # Map to CRM fields - only keep the core CRM fields
            if key in ['company_name', 'ask', 'revenue', 'valuation', 'source', 'assign', 'description']:
                crm_data[key] = value if value and value != "Not mentioned" else ""
    
    return crm_data
    
def extract_number_cr(text):
    """Extract numeric values from text for CRM (handles M, K, B suffixes)"""
    if not text or text == "Not mentioned":
        return ""
    
    import re
    
    # Remove common prefixes and clean the text
    text = re.sub(r'[$£€¥₹]', '', text)  # Remove currency symbols
    text = re.sub(r'[,\s]', '', text)    # Remove commas and spaces
    
    # Look for patterns like 2M, 500K, 1.5B, etc.
    pattern = r'(\d+(?:\.\d+)?)\s*([KMB])'
    match = re.search(pattern, text.upper())
    
    if match:
        number = float(match.group(1))
        suffix = match.group(2)
        
        # Convert to actual number
        if suffix == 'K':
            return str(int(number * 1000))
        elif suffix == 'M':
            return str(int(number * 1000000))
        elif suffix == 'B':
            return str(int(number * 1000000000))
    
    # If no suffix found, try to extract just the number
    number_match = re.search(r'(\d+(?:\.\d+)?)', text)
    if number_match:
        return str(int(float(number_match.group(1))))
    
    return text  # Return original if no number found

def send_to_zoho_webhook(crm_data):
    """Send data to Zoho webhook and handle webhook testing"""
    if not zoho_webhook_url:
        logger.warning("❌ ZOHO_WEBHOOK_URL not set in .env")
        return

    try:
        # Preprocess values for Zoho - convert ask and valuation to numbers
        crm_payload = {
            "company_name": crm_data.get("company_name", ""),
            "ask": extract_number_cr(crm_data.get("ask", "")),
            "valuation": extract_number_cr(crm_data.get("valuation", "")),
            "revenue": extract_number_cr(crm_data.get("revenue", "")),
            "description": crm_data.get("description", ""),
            "source": crm_data.get("source", ""),
            "assign": crm_data.get("assign", ""),
            "received_date": crm_data.get("received_date", ""),
            "timestamp": datetime.now().isoformat(),
            "test_mode": True  # Add this to indicate it's a test
        }

        # Debug: Show what's being sent
        logger.info(f"Sending to Zoho: {crm_payload}")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Manna-VC-Pitch-Evaluator/1.0"
        }
        
        # Send the webhook
        response = requests.post(
            zoho_webhook_url, 
            json=crm_payload, 
            headers=headers,
            timeout=30
        )

        if response.status_code in [200, 201, 202]:
            logger.info("✅ CRM data sent to Zoho Flow successfully")
            st.success("✅ Data sent to Zoho CRM successfully!")
            
            # Store the successful response
            st.session_state.webhook_responses.append({
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "status_code": response.status_code,
                "response": response.text[:500],  # Limit response text
                "payload": crm_payload
            })
            
        else:
            logger.warning(f"⚠️ Webhook error: {response.status_code} - {response.text}")
            st.warning(f"⚠️ Webhook error: {response.status_code}")
            
            # Store the error response
            st.session_state.webhook_responses.append({
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "status_code": response.status_code,
                "response": response.text[:500],
                "payload": crm_payload
            })

    except requests.exceptions.Timeout:
        logger.error("❌ Webhook timeout")
        st.error("❌ Webhook timeout - request took too long")
    except requests.exceptions.ConnectionError:
        logger.error("❌ Connection error to webhook")
        st.error("❌ Connection error to webhook")
    except Exception as e:
        logger.error(f"❌ Failed to send to Zoho webhook: {e}")
        st.error(f"❌ Failed to send to Zoho: {e}")

# Test webhook function
def test_webhook_connection():
    """Test webhook connection with sample data"""
    if not zoho_webhook_url:
        st.error("❌ ZOHO_WEBHOOK_URL not set in .env")
        return
    
    test_payload = {
        "company_name": "Test Company",
        "ask": "1000000",
        "valuation": "5000000",
        "revenue": "250000",
        "description": "Test company for webhook validation",
        "source": "Webhook Test",
        "assign": "Test User",
        "received_date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "test_mode": True
    }
    
    try:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Manna-VC-Pitch-Evaluator/1.0"
        }
        
        response = requests.post(
            zoho_webhook_url, 
            json=test_payload, 
            headers=headers,
            timeout=30
        )
        
        if response.status_code in [200, 201, 202]:
            st.success(f"✅ Webhook test successful! Status: {response.status_code}")
            st.json({"status": "success", "response": response.text[:200]})
        else:
            st.error(f"❌ Webhook test failed! Status: {response.status_code}")
            st.json({"status": "error", "response": response.text[:200]})
            
    except Exception as e:
        st.error(f"❌ Webhook test error: {str(e)}")

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
st.set_page_config(page_title="Manna — VC Pitch Evaluator", page_icon="📊")
st.title("📊 Manna — VC Pitch Evaluator")

# Add webhook testing section at the top
st.header("🔗 Webhook Testing")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🧪 Test Webhook Connection"):
        with st.spinner("Testing webhook..."):
            test_webhook_connection()

with col2:
    if st.button("📋 Show Webhook URL"):
        if zoho_webhook_url:
            st.success("✅ Webhook URL configured")
            st.code(zoho_webhook_url[:50] + "..." if len(zoho_webhook_url) > 50 else zoho_webhook_url)
        else:
            st.error("❌ No webhook URL configured")

with col3:
    if st.button("📊 Webhook History"):
        if st.session_state.webhook_responses:
            st.subheader("Recent Webhook Responses")
            for i, response in enumerate(reversed(st.session_state.webhook_responses[-5:])):
                status_icon = "✅" if response['status'] == 'success' else "❌"
                st.write(f"{status_icon} {response['timestamp'][:19]} - Status: {response['status_code']}")
        else:
            st.info("No webhook responses yet")

st.markdown("---")

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
    
    # Show CRM data
    if st.session_state.crm_data:
        st.header("🔗 CRM Integration Data")
        
        # Display key CRM fields
        crm_fields = ['company_name', 'ask', 'revenue', 'valuation', 'source', 'assign', 'description']
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

 # Extract CRM structured data
        with st.spinner("🔄 Extracting CRM data..."):
            structured_text = extract_crm_structured_data(text)
            st.session_state.structured_data = structured_text
            st.session_state.crm_data = parse_crm_data(structured_text)
            st.session_state.crm_data['received_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Send to Zoho webhook automatically
            send_to_zoho_webhook(st.session_state.crm_data)
        
        st.success("✅ Pitch deck processed successfully!")
        st.rerun()

# Chat interface
if st.session_state.file_uploaded:
    st.header("💬 Chat with your pitch deck")
    
    # Display current section if selected
    if hasattr(st.session_state, 'selected_section') and st.session_state.selected_section:
        st.subheader(f"📋 {st.session_state.selected_section}")
        st.markdown(st.session_state.sections[st.session_state.selected_section])
        
        if st.button("❌ Close Section View"):
            del st.session_state.selected_section
            st.rerun()
        
        st.markdown("---")
    
    # Chat input
    user_input = st.text_input("💬 Ask me anything about this pitch deck:", placeholder="e.g., What's the founder's background?")
    
    if user_input:
        # Add to chat history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with st.spinner("🤖 Generating response..."):
            # Check if it's a specific CRM query
            specific_field = is_specific_crm_query(user_input)
            
            if specific_field:
                # Generate focused CRM response
                response = generate_crm_response(specific_field, st.session_state.crm_data)
            else:
                # General chat response
                try:
                    llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.3)
                    
                    # Check if web search is needed
                    search_keywords = ["founder background", "ceo education", "linkedin", "previous company", "experience"]
                    needs_search = any(keyword in user_input.lower() for keyword in search_keywords)
                    
                    if needs_search:
                        # Extract company name for search
                        company_name = st.session_state.crm_data.get('company_name', '')
                        founder_name = st.session_state.crm_data.get('assign', '')
                        
                        search_query = f"{founder_name} {company_name} founder background"
                        search_results = search_serpapi(search_query)
                        
                        context = f"""
                        Pitch deck content: {st.session_state.parsed_doc}
                        
                        Web search results for founder background:
                        {search_results}
                        
                        Please provide a comprehensive answer combining both the pitch deck information and web search results.
                        """
                    else:
                        context = st.session_state.parsed_doc
                    
                    messages = [
                        SystemMessage(content="""You are Manna, a VC analyst AI assistant helping investors evaluate startup pitch decks. 
                        
                        Guidelines:
                        - Be professional and insightful
                        - Provide specific, actionable insights
                        - Reference exact information from the pitch deck
                        - Highlight missing information that would be important for due diligence
                        - Use formatting for readability
                        - Be direct and concise
                        - Focus on investment-relevant information
                        
                        When answering questions about founders or team members, provide:
                        - Educational background
                        - Professional experience
                        - Previous companies and roles
                        - Relevant industry expertise
                        - Any notable achievements
                        """),
                        HumanMessage(content=f"Based on this pitch deck content, answer this question: {user_input}\n\nContext:\n{context}")
                    ]
                    
                    response = llm.invoke(messages).content
                    
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    response = f"❌ Error generating response: {str(e)}"
        
        # Store in chat history
        st.session_state.chat_history.append((user_input, response, timestamp))
        
        # Display current response
        st.markdown(f"**🧑 You:** {user_input}")
        st.markdown(f"**🤖 Manna:**")
        st.markdown(response)
        st.markdown(f"*{timestamp}*")
        
        # Clear input by rerunning
        st.rerun()

# Quick action buttons
if st.session_state.file_uploaded:
    st.header("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💰 Funding Ask"):
            if st.session_state.crm_data and st.session_state.crm_data.get('ask'):
                st.info(f"**Funding Ask:** {st.session_state.crm_data['ask']}")
            else:
                st.warning("Funding ask not clearly mentioned in the deck")
    
    with col2:
        if st.button("💎 Valuation"):
            if st.session_state.crm_data and st.session_state.crm_data.get('valuation'):
                st.info(f"**Valuation:** {st.session_state.crm_data['valuation']}")
            else:
                st.warning("Valuation not clearly mentioned in the deck")
    
    with col3:
        if st.button("📈 Revenue"):
            if st.session_state.crm_data and st.session_state.crm_data.get('revenue'):
                st.info(f"**Revenue:** {st.session_state.crm_data['revenue']}")
            else:
                st.warning("Revenue not clearly mentioned in the deck")
    
    with col4:
        if st.button("👥 Founders"):
            if st.session_state.crm_data and st.session_state.crm_data.get('assign'):
                st.info(f"**Founders:** {st.session_state.crm_data['assign']}")
            else:
                st.warning("Founder information not clearly mentioned in the deck")

# Footer
st.markdown("---")
st.markdown("🚀 **Manna VC Pitch Evaluator** | Built for efficient startup evaluation")

# Debug information (only show in development)
if st.checkbox("🔧 Show Debug Info"):
    st.subheader("Debug Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Session State Keys:**")
        for key in st.session_state.keys():
            st.write(f"- {key}")
    
    with col2:
        st.write("**Environment Variables:**")
        st.write(f"- OpenAI API Key: {'✅ Set' if openai_api_key else '❌ Missing'}")
        st.write(f"- SERP API Key: {'✅ Set' if serpapi_key else '❌ Missing'}")
        st.write(f"- Zoho Webhook URL: {'✅ Set' if zoho_webhook_url else '❌ Missing'}")
    
    if st.session_state.crm_data:
        st.subheader("CRM Data (Raw)")
        st.json(st.session_state.crm_data)
    
    if st.session_state.webhook_responses:
        st.subheader("Webhook Responses")
        for i, response in enumerate(st.session_state.webhook_responses):
            st.write(f"**Response {i+1}:**")
            st.json(response)

# Auto-refresh for webhook responses
if st.session_state.webhook_responses:
    # Show latest webhook status in the main area
    latest_response = st.session_state.webhook_responses[-1]
    if latest_response['status'] == 'success':
        st.success(f"✅ Latest webhook sent successfully at {latest_response['timestamp'][:19]}")
    else:
        st.error(f"❌ Latest webhook failed at {latest_response['timestamp'][:19]} - Status: {latest_response['status_code']}")
        
