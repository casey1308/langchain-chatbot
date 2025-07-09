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
            "received_date": datetime.now().strftime("%d %m %Y"),
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
        "received_date": datetime.now().strftime("%d %m %Y"),
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

# Main upload section - this should be the first thing users see
st.header("📄 Upload Pitch Deck")
file = st.file_uploader("Upload a startup pitch deck (PDF)", type=["pdf"])

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
            if zoho_webhook_url:
                send_to_zoho_webhook(st.session_state.crm_data)
        
        st.success("✅ Pitch deck processed successfully!")
        st.rerun()

# Show main features only after file is uploaded
if st.session_state.file_uploaded:
    # CRM Data Summary Card
    if st.session_state.crm_data:
        st.header("💼 CRM Data Summary")
        
        # Display key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            company_name = st.session_state.crm_data.get('company_name', 'Not mentioned')
            st.metric("Company", company_name if company_name else "Not mentioned")
        
        with col2:
            ask = st.session_state.crm_data.get('ask', 'Not mentioned')
            st.metric("Funding Ask", ask if ask else "Not mentioned")
        
        with col3:
            valuation = st.session_state.crm_data.get('valuation', 'Not mentioned')
            st.metric("Valuation", valuation if valuation else "Not mentioned")
        
        with col4:
            revenue = st.session_state.crm_data.get('revenue', 'Not mentioned')
            st.metric("Revenue", revenue if revenue else "Not mentioned")
        
        # Founders and Description
        if st.session_state.crm_data.get('assign'):
            st.write(f"**👥 Founders:** {st.session_state.crm_data['assign']}")
        
        if st.session_state.crm_data.get('description'):
            st.write(f"**📝 Description:** {st.session_state.crm_data['description']}")
        
        st.markdown("---")
    
    # Action buttons
    st.header("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Full Analysis", use_container_width=True):
            st.session_state.show_comprehensive_analysis = True
    
    with col2:
        if st.button("📊 Export CRM Data", use_container_width=True):
            st.session_state.show_crm_export = True
    
    with col3:
        if st.button("🔗 CRM Integration", use_container_width=True):
            st.session_state.show_crm_integration = True
    
    st.markdown("---")
    
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
    
    # Show CRM export
    if hasattr(st.session_state, 'show_crm_export') and st.session_state.show_crm_export:
        st.header("📤 CRM Data Export")
        
        if st.session_state.crm_data:
            # Display CRM data in clean format
            st.subheader("🔑 Extracted CRM Fields")
            
            for field, value in st.session_state.crm_data.items():
                if value:
                    st.write(f"**{field.replace('_', ' ').title()}:** {value}")
            
            # JSON export
            st.subheader("📋 JSON Export")
            st.json(st.session_state.crm_data)
            
            # Download button
            json_str = json.dumps(st.session_state.crm_data, indent=2)
            st.download_button(
                label="💾 Download CRM Data",
                data=json_str,
                file_name=f"crm_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
      if st.button("❌ Close Integration"):
            st.session_state.show_crm_integration = False
            st.rerun()
        
        st.markdown("---")
    
    # Show CRM integration
    if hasattr(st.session_state, 'show_crm_integration') and st.session_state.show_crm_integration:
        st.header("🔗 CRM Integration")
        
        # Webhook testing section
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Test Webhook", use_container_width=True):
                with st.spinner("Testing webhook..."):
                    test_webhook_connection()
        
        with col2:
            if st.button("📋 Show Webhook Status", use_container_width=True):
                if zoho_webhook_url:
                    st.success("✅ Webhook URL configured")
                    st.code(zoho_webhook_url[:50] + "..." if len(zoho_webhook_url) > 50 else zoho_webhook_url)
                else:
                    st.error("❌ No webhook URL configured")
        
        # Show webhook history
        if st.session_state.webhook_responses:
            st.subheader("📊 Recent Webhook Responses")
            for i, response in enumerate(reversed(st.session_state.webhook_responses[-5:])):
                status_icon = "✅" if response['status'] == 'success' else "❌"
                with st.expander(f"{status_icon} {response['timestamp'][:19]} - Status: {response['status_code']}"):
                    st.json(response)
        
    if st.button("❌ Close Integration"):
            st.session_state.show_crm_integration = False
            st.rerun()
        st.markdown("---")
    
    # Chat interface
    st.header("💬 Ask Questions About This Pitch")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"💬 {q[:50]}..." if len(q) > 50 else f"💬 {q}"):
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {a}")
                if st.button(f"🗑️ Delete", key=f"del_{i}"):
                    st.session_state.chat_history.pop(i)
                    st.rerun()
        st.markdown("---")
    
    # New question input
    query = st.text_input("💭 Ask a question about this pitch deck:", placeholder="e.g., What is the funding ask? Who are the founders?")
    
    if query:
        with st.spinner("🤔 Analyzing..."):
            try:
                # Check if it's a specific CRM query
                crm_field = is_specific_crm_query(query)
                
                if crm_field and st.session_state.crm_data:
                    answer = generate_crm_response(crm_field, st.session_state.crm_data)
                else:
                    # Try to find relevant section
                    relevant_section = match_section(query, st.session_state.sections, st.session_state.structured_data)
                    
                    # If no relevant section found, use web search
                    if "Not mentioned" in relevant_section or len(relevant_section) < 50:
                        if any(x in query.lower() for x in ["founder", "ceo", "team", "background", "linkedin"]):
                            # Extract company name for founder search
                            company_name = st.session_state.crm_data.get('company_name', '') if st.session_state.crm_data else ''
                            search_query = f"{query} {company_name}".strip()
                            web_info = search_serpapi(search_query)
                            
                            if "❌" not in web_info:
                                answer = f"**From Web Search:**\n\n{web_info}"
                            else:
                                answer = f"**From Pitch Deck:**\n{relevant_section}\n\n**Web Search:** {web_info}"
                        else:
                            answer = f"**From Pitch Deck:**\n{relevant_section}"
                    else:
                        # Use LLM to generate contextual answer
                        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.1)
                        
                        context_prompt = f"""
                        Based on this pitch deck content, answer the following question clearly and concisely:
                        
                        Question: {query}
                        
                        Relevant Content:
                        {relevant_section}
                        
                        Additional CRM Data:
                        {st.session_state.structured_data if st.session_state.structured_data else 'None'}
                        
                        Instructions:
                        1. Provide a direct answer to the question
                        2. Use specific information from the pitch deck
                        3. If information is missing, clearly state that
                        4. Keep the response focused and professional
                        5. Format the response for easy reading
                        """
                        
                        answer = llm.invoke(context_prompt).content.strip()
                
                # Add to chat history
                st.session_state.chat_history.append((query, answer))
                
                # Display the answer
                st.markdown("### 💡 Answer:")
                st.markdown(answer)
                
            except Exception as e:
                error_msg = f"❌ Error processing question: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append((query, error_msg))

else:
    # Instructions when no file is uploaded
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Upload a pitch deck PDF** using the file uploader above
    2. **Review the CRM data** extracted automatically
    3. **Ask questions** about the pitch using natural language
    4. **Generate comprehensive analysis** for investment decisions
    5. **Export data** to your CRM system
    
    ### 📋 What You Can Ask
    
    - **Founder Information**: "Who are the founders?" or "What's the team background?"
    - **Financial Data**: "What's the funding ask?" or "What's the current revenue?"
    - **Market Analysis**: "What's the market size?" or "Who are the competitors?"
    - **Business Model**: "How do they make money?" or "What's the value proposition?"
    - **Traction**: "What traction do they have?" or "What are their key metrics?"
    
    ### 🔧 Features
    
    - **Smart PDF Processing**: Extracts and structures pitch deck content
    - **CRM Integration**: Automatically formats data for Zoho CRM
    - **Web Search**: Enriches founder and company information
    - **Comprehensive Analysis**: Generates detailed VC investment analysis
    - **Export Options**: Download data in JSON format
    """)

# Footer
st.markdown("---")
st.markdown("**Manna VC Pitch Evaluator** | Built with Streamlit, OpenAI, and LangChain")
