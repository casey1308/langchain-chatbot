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

def extract_crm_structured_data(text):
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)
        extraction_prompt = """
        Analyze this pitch deck and extract ONLY the following CRM-specific information in exact key-value format.
        Look through ALL the text carefully and extract specific numbers, amounts, and concrete details.

        CRITICAL: Use these EXACT keys and provide specific values or "Not mentioned":

        company_name: [Extract exact company name]
        sector: [Industry/sector, e.g., "Fintech", "Healthtech", "SaaS", or "Not mentioned"]
        stage: [Startup stage, e.g., "Bootstrapped", "Pre-seed", "Seed", "Early", "Growth", "Series A", etc., or "Not mentioned"]
        ask: [Funding amount being sought, e.g., "$2M Series A" or "Not mentioned"]
        revenue: [Current revenue figures with specific numbers, e.g., "$500K ARR" or "Not mentioned"]
        valuation: [Current valuation with currency and amount, e.g., "$5M pre-money" or "Not mentioned"]
        source: [Always put "Pitch Deck Upload"]
        assign: [Founder names and key team members, e.g., "John Doe (CEO), Jane Smith (CTO)" or "Not mentioned"]
        description: [Brief 2-3 sentence description of what the company does]

        INSTRUCTIONS:
        1. Extract SPECIFIC numbers and amounts wherever possible.
        2. Include currency symbols and units (K, M, B).
        3. Look for information across ALL sections of the document.
        4. For sector and stage, infer from context if not explicitly mentioned.
        5. For ask: include round type if mentioned (Seed, Series A, etc.).
        6. For revenue: include type if mentioned (ARR, MRR, total revenue).
        7. For assign: include founder names and key roles.
        8. For description: keep it concise and business-focused.
        9. Source should always be "Pitch Deck Upload".

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

def parse_crm_data(structured_text):
    crm_data = {}
    lines = structured_text.split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            if key in ['company_name', 'sector', 'stage', 'ask', 'revenue', 'valuation', 'source', 'assign', 'description']:
                crm_data[key] = value if value and value != "Not mentioned" else ""
    return crm_data

st.warning(f"Sending data to webhook: {zoho_webhook_url}")

def send_to_zoho_webhook(crm_data):
    if not zoho_webhook_url:
        logger.warning("❌ ZOHO_WEBHOOK_URL not set in .env")
        return
    try:
        crm_payload = {
            "company_name": crm_data.get("company_name", ""),
            "sector": crm_data.get("sector", ""),
            "stage": crm_data.get("stage", ""),
            "ask": crm_data.get("ask", ""),
            "valuation": crm_data.get("valuation", ""),
            "revenue": crm_data.get("revenue", ""),
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

def is_specific_crm_query(query):
    query_lower = query.lower()
    specific_keywords = {
        'ask': ['ask', 'funding', 'investment', 'raise', 'capital', 'seeking', 'round'],
        'founder': ['founder', 'ceo', 'team', 'who founded', 'leadership', 'management'],
        'revenue': ['revenue', 'sales', 'income', 'earnings', 'arr', 'mrr'],
        'valuation': ['valuation', 'worth', 'valued', 'pre-money', 'post-money'],
        'company': ['company name', 'what company', 'name of company'],
        'sector': ['sector', 'industry', 'vertical'],
        'stage': ['stage', 'bootstrapped', 'seed', 'series', 'early', 'growth'],
        'description': ['what do they do', 'what does the company do', 'business', 'product', 'service']
    }
    for field, keywords in specific_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return field
    return None

def generate_crm_response(field, crm_data):
    if not crm_data:
        return "No CRM data available. Please upload a pitch deck first."
    field_mapping = {
        'company': 'company_name',
        'founder': 'assign',
        'ask': 'ask',
        'revenue': 'revenue',
        'valuation': 'valuation',
        'sector': 'sector',
        'stage': 'stage',
        'description': 'description'
    }
    crm_field = field_mapping.get(field, field)
    value = crm_data.get(crm_field, "Not mentioned")
    if not value or value == "Not mentioned":
        return f"**{field.title()}:** Not mentioned in the pitch deck."
    return f"**{field.title()}:** {value}"

def extract_comprehensive_analysis(text):
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

def match_section(key, sections, structured_data=None):
    key = key.lower()
    if structured_data and "founders" in key:
        return structured_data
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
    matches = get_close_matches(key, [k.lower() for k in sections.keys()], n=1, cutoff=0.3)
    if matches:
        for k in sections:
            if k.lower() == matches[0]:
                return sections[k]
    return "Not mentioned in deck."

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

st.set_page_config(page_title="Perpendo — VC Pitch Evaluator", page_icon="📊")
st.title("📊 Perpendo — VC Pitch Evaluator")

with st.sidebar:
    st.header("📋 Detected Sections")
    if st.session_state.sections:
        for section_name in st.session_state.sections.keys():
            if st.button(section_name, key=f"section_{section_name}"):
                st.session_state.selected_section = section_name
    if st.session_state.file_uploaded:
        st.header("📊 Quick Stats")
        st.write(f"Total sections: {len(st.session_state.sections)}")
        st.write(f"Total text length: {len(st.session_state.parsed_doc)} chars")
    if st.session_state.crm_data:
        st.header("🔗 CRM Integration Data")
        crm_fields = [
            'company_name', 'sector', 'stage', 'ask', 'revenue', 'valuation', 'source', 'assign', 'description'
        ]
        for field in crm_fields:
            if field in st.session_state.crm_data and st.session_state.crm_data[field]:
                display_value = st.session_state.crm_data[field]
                if len(display_value) > 50:
                    display_value = display_value[:50] + "..."
                st.write(f"**{field.replace('_', ' ').title()}:** {display_value}")
        if st.button("📤 Export CRM Data"):
            st.json(st.session_state.crm_data)
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

if st.session_state.structured_data:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 View Comprehensive Analysis"):
            st.session_state.show_comprehensive_analysis = True
    with col2:
        if st.button("📊 View CRM Data Summary"):
            st.session_state.show_crm_summary = True

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

if hasattr(st.session_state, 'show_crm_summary') and st.session_state.show_crm_summary:
    st.header("📊 CRM Data Summary")
    if st.session_state.crm_data:
        st.subheader("🔑 CRM Fields")
        for field, value in st.session_state.crm_data.items():
            if value:
                st.write(f"**{field.replace('_', ' ').title()}:** {value}")
        st.subheader("📤 Export Data")
        st.json(st.session_state.crm_data)
        if st.button("📋 Copy JSON to Clipboard"):
            st.code(json.dumps(st.session_state.crm_data, indent=2))
    if st.button("❌ Close CRM Summary"):
        st.session_state.show_crm_summary = False
        st.rerun()
    st.markdown("---")

if st.session_state.selected_chat_index is not None:
    chat_data = st.session_state.chat_history[st.session_state.selected_chat_index]
    user_q, bot_response, timestamp = chat_data
    st.header(f"💬 Chat #{st.session_state.selected_chat_index + 1}")
    st.markdown(f"**🧑 You:** {user_q}")
    st.markdown(f"**🤖 Perpendo:**")
    st.markdown(bot_response)
    st.markdown(f"*{timestamp}*")
    if st.button("❌ Close Chat View"):
        st.session_state.selected_chat_index = None
        st.rerun()
    st.markdown("---")

file = st.file_uploader("📄 Upload a startup pitch deck (PDF)", type=["pdf"])

if file:
    with st.spinner("📄 Parsing pitch deck..."):
        file_bytes = file.read()
        text = extract_pdf_text(file_bytes)
        st.session_state.parsed_doc = text
        st.session_state.sections = split_sections(text)
        st.session_state.file_uploaded = True

    with st.spinner("🔍 Extracting CRM data..."):
        crm_structured_text = extract_crm_structured_data(text)
        st.session_state.structured_data = crm_structured_text
        st.session_state.crm_data = parse_crm_data(crm_structured_text)
        st.session_state.crm_data["received_date"] = datetime.today().strftime("%Y-%m-%d")
        send_to_zoho_webhook(st.session_state.crm_data)
    st.success("✅ Pitch deck parsed and CRM data extracted!")
    
  if st.session_state.crm_data:
    st.subheader("🔗 CRM Data Preview")

    card_html = """
    <style>
    .crm-card {
        background: linear-gradient(120deg, #f8fafc 0%, #e0e7ef 100%);
        border-radius: 18px;
        box-shadow: 0 6px 32px 0 rgba(30, 41, 59, 0.12);
        padding: 2.5rem 2rem 2rem 2rem;
        margin-bottom: 2rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
        color: #23272f;
    }
    .crm-card h2 {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #1e293b;
        letter-spacing: -1px;
    }
    .crm-field {
        margin-bottom: 1.1rem;
    }
    .crm-label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #64748b;
        margin-bottom: 0.18rem;
        letter-spacing: 0.5px;
    }
    .crm-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #23272f;
        background: #f1f5f9;
        border-radius: 6px;
        padding: 0.45rem 0.8rem;
        display: inline-block;
    }
    </style>
    <div class="crm-card">
        <h2>{company_name}</h2>
        <div class="crm-field">
            <span class="crm-label">Sector:</span><br>
            <span class="crm-value">{sector}</span>
        </div>
        <div class="crm-field">
            <span class="crm-label">Stage:</span><br>
            <span class="crm-value">{stage}</span>
        </div>
        <div class="crm-field">
            <span class="crm-label">Ask:</span><br>
            <span class="crm-value">{ask}</span>
        </div>
        <div class="crm-field">
            <span class="crm-label">Valuation:</span><br>
            <span class="crm-value">{valuation}</span>
        </div>
        <div class="crm-field">
            <span class="crm-label">Revenue:</span><br>
            <span class="crm-value">{revenue}</span>
        </div>
        <div class="crm-field">
            <span class="crm-label">Founders / Key Team:</span><br>
            <span class="crm-value">{assign}</span>
        </div>
        <div class="crm-field">
            <span class="crm-label">Description:</span><br>
            <span class="crm-value">{description}</span>
        </div>
        <div class="crm-field">
            <span class="crm-label">Source:</span><br>
            <span class="crm-value">{source}</span>
        </div>
        <div class="crm-field">
            <span class="crm-label">Received Date:</span><br>
            <span class="crm-value">{received_date}</span>
        </div>
    </div>
    """.format(
        company_name=st.session_state.crm_data.get("company_name", "Not found"),
        sector=st.session_state.crm_data.get("sector", "Not found"),
        stage=st.session_state.crm_data.get("stage", "Not found"),
        ask=st.session_state.crm_data.get("ask", "Not found"),
        valuation=st.session_state.crm_data.get("valuation", "Not found"),
        revenue=st.session_state.crm_data.get("revenue", "Not found"),
        assign=st.session_state.crm_data.get("assign", "Not found"),
        description=st.session_state.crm_data.get("description", "Not found"),
        source=st.session_state.crm_data.get("source", "Pitch Deck Upload"),
        received_date=st.session_state.crm_data.get("received_date", ""),
    )

    st.markdown(card_html, unsafe_allow_html=True)

if hasattr(st.session_state, 'selected_section') and st.session_state.selected_section:
    st.subheader(f"📖 {st.session_state.selected_section}")
    st.text_area("Content", st.session_state.sections[st.session_state.selected_section], height=200)

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
    if st.button("🏭 Sector?"):
        st.session_state.auto_query = "What is their sector?"
with col6:
    if st.button("🚀 Stage?"):
        st.session_state.auto_query = "What is their stage?"

st.markdown("**Example specific queries:**")
st.markdown("- `What is their ask?` - Returns just the funding ask")
st.markdown("- `Who are the founders?` - Returns just founder information")
st.markdown("- `What is their revenue?` - Returns just revenue data")
st.markdown("- `What does the company do?` - Returns just company description")
st.markdown("- `What is their sector?` - Returns the sector/industry")
st.markdown("- `What is their stage?` - Returns the startup stage")

if hasattr(st.session_state, 'auto_query') and st.session_state.auto_query:
    user_query = st.session_state.auto_query
    st.session_state.auto_query = None
else:
    user_query = st.chat_input("💬 Ask about the pitch deck...")

if user_query:
    st.session_state.selected_chat_index = None
    specific_field = is_specific_crm_query(user_query)
    if specific_field and st.session_state.crm_data:
        response = generate_crm_response(specific_field, st.session_state.crm_data)
        st.markdown(f"**🧑 You:** {user_query}")
        st.markdown("**🤖 Perpendo:**")
        st.markdown(response)
        st.session_state.chat_history.append(
            (user_query, response, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
    else:
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
                if not hasattr(st.session_state, 'comprehensive_analysis'):
                    st.session_state.comprehensive_analysis = extract_comprehensive_analysis(context)
                context_msg = f"COMPREHENSIVE ANALYSIS:\n{st.session_state.comprehensive_analysis}\n\nCRM DATA:\n{st.session_state.structured_data}\n\nFULL DOCUMENT:\n{context[:2000]}"
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
                st.markdown(f"**🧑 You:** {user_query}")
                st.markdown("**🤖 Perpendo:**")
                response_placeholder = st.empty()
                full_response = ""
                for chunk in llm.stream(messages):
                    if hasattr(chunk, 'content') and chunk.content:
                        full_response += chunk.content
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
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

st.markdown("---")
st.markdown("**💡 Pro Tip:** Use specific queries like 'What is their ask?' for quick CRM data, or ask comprehensive questions for detailed analysis.")
