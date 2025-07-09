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
        valuation: [Extract current valuation with currency and amount, e.g., "$5M pre-money" or "Not mentioned"]
        revenue: [Extract current revenue figures with specific numbers, e.g., "$500K ARR" or "Not mentioned"]
        ask: [Extract funding amount being sought, e.g., "$2M Series A" or "Not mentioned"]
        source: [Always put "Pitch Deck Upload"]
        
        ADDITIONAL CONTEXT (for analysis only):
        industry: [Industry/sector]
        stage: [Funding stage]
        founder_name: [Primary founder name]
        founder_role: [Primary founder role]
        location: [Company location]
        employees: [Team size/employee count]
        customers: [Customer metrics]
        market_size: [Addressable market size]
        
        INSTRUCTIONS:
        1. Extract SPECIFIC numbers and amounts wherever possible
        2. Include currency symbols and units (K, M, B)
        3. Look for information across ALL sections of the document
        4. Be precise with valuation (pre-money/post-money distinction)
        5. For ask: include round type if mentioned (Seed, Series A, etc.)
        6. For revenue: include type if mentioned (ARR, MRR, total revenue)
        7. Source should always be "Pitch Deck Upload"
        
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
            
            # Map to CRM fields
            if key in ['company_name', 'valuation', 'revenue', 'ask', 'source']:
                crm_data[key] = value if value and value != "Not mentioned" else ""
            elif key in ['industry', 'stage', 'founder_name', 'founder_role', 'location', 'employees', 'customers', 'market_size']:
                crm_data[key] = value if value and value != "Not mentioned" else ""
    
    return crm_data

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
        crm_fields = ['company_name', 'valuation', 'revenue', 'ask', 'source']
        for field in crm_fields:
            if field in st.session_state.crm_data and st.session_state.crm_data[field]:
                st.write(f"**{field.replace('_', ' ').title()}:** {st.session_state.crm_data[field]}")
        
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
        # Primary CRM fields
        st.subheader("🔑 Primary CRM Fields")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Company", st.session_state.crm_data.get('company_name', 'Not specified'))
            st.metric("Valuation", st.session_state.crm_data.get('valuation', 'Not specified'))
            st.metric("Revenue", st.session_state.crm_data.get('revenue', 'Not specified'))
        
        with col2:
            st.metric("Funding Ask", st.session_state.crm_data.get('ask', 'Not specified'))
            st.metric("Source", st.session_state.crm_data.get('source', 'Not specified'))
        
        # Additional context
        st.subheader("📋 Additional Context")
        additional_fields = ['industry', 'stage', 'founder_name', 'founder_role', 'location', 'employees', 'customers', 'market_size']
        
        for field in additional_fields:
            if field in st.session_state.crm_data and st.session_state.crm_data[field]:
                st.write(f"**{field.replace('_', ' ').title()}:** {st.session_state.crm_data[field]}")
        
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
        
        # Extract CRM-specific structured data
        with st.spinner("🔍 Extracting CRM data..."):
            crm_structured_text = extract_crm_structured_data(text)
            st.session_state.structured_data = crm_structured_text
            st.session_state.crm_data = parse_crm_data(crm_structured_text)
        
    st.success("✅ Pitch deck parsed and CRM data extracted!")
    
    # Show CRM data preview
    if st.session_state.crm_data:
        st.subheader("🔗 CRM Data Preview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Company", st.session_state.crm_data.get('company_name', 'Not found')[:20] + "..." if len(st.session_state.crm_data.get('company_name', '')) > 20 else st.session_state.crm_data.get('company_name', 'Not found'))
        
        with col2:
            st.metric("Valuation", st.session_state.crm_data.get('valuation', 'Not found'))
        
        with col3:
            st.metric("Ask", st.session_state.crm_data.get('ask', 'Not found'))

# Show selected section
if hasattr(st.session_state, 'selected_section') and st.session_state.selected_section:
    st.subheader(f"📖 {st.session_state.selected_section}")
    st.text_area("Content", st.session_state.sections[st.session_state.selected_section], height=200)

# Enhanced prompt input
st.markdown("### 💬 Ask Questions")
st.markdown("**Quick Actions:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Full Analysis"):
        st.session_state.auto_query = "Provide a comprehensive analysis of this pitch deck"

with col2:
    if st.button("👥 Founder Analysis"):
        st.session_state.auto_query = "Analyze the founders' backgrounds, experience, and team composition"

with col3:
    if st.button("💰 Financial Analysis"):
        st.session_state.auto_query = "Analyze the financial aspects including valuation, revenue, and funding ask"

st.markdown("**Example queries:**")
st.markdown("- `Provide a comprehensive VC analysis of this pitch deck`")
st.markdown("- `What are the key risks and opportunities?`")
st.markdown("- `Analyze the competitive landscape and market opportunity`")
st.markdown("- `What due diligence questions should I ask?`")

# Handle auto queries
if hasattr(st.session_state, 'auto_query') and st.session_state.auto_query:
    user_query = st.session_state.auto_query
    st.session_state.auto_query = None
else:
    user_query = st.chat_input("💬 Ask about the pitch deck...")

if user_query:
    # Clear any selected chat when asking new question
    st.session_state.selected_chat_index = None
    
    with st.spinner("🤖 Analyzing..."):
        try:
            context = st.session_state.parsed_doc or ""
            llm = ChatOpenAI(
                model="gpt-4o", 
                openai_api_key=openai_api_key, 
                streaming=True,
                temperature=0.1,  # Lower temperature for more analytical responses
                max_tokens=2000
            )
            
            lower_q = user_query.lower()

            # Enhanced intent matching for deeper analysis
            intent_keys = {
                "comprehensive": ["comprehensive", "full analysis", "complete analysis", "overall", "summary", "evaluate", "assessment"],
                "founder": ["founder", "team", "leadership", "management", "background", "experience"],
                "financial": ["financial", "valuation", "revenue", "funding", "ask", "investment", "money"],
                "market": ["market", "competition", "competitive", "industry", "opportunity"],
                "product": ["product", "technology", "solution", "features"],
                "traction": ["traction", "metrics", "growth", "customers", "users"],
                "risks": ["risks", "challenges", "concerns", "red flags", "weaknesses"],
                "due_diligence": ["due diligence", "questions", "investigation", "research"]
            }

            # Check for comprehensive analysis
            is_comprehensive = any(keyword in lower_q for keyword in intent_keys.get("comprehensive", []))
            
            if is_comprehensive:
                # Generate comprehensive analysis if not already done
                if not hasattr(st.session_state, 'comprehensive_analysis'):
                    st.session_state.comprehensive_analysis = extract_comprehensive_analysis(context)
                
                context_msg = f"COMPREHENSIVE ANALYSIS:\n{st.session_state.comprehensive_analysis}\n\nCRM DATA:\n{st.session_state.structured_data}\n\nFULL DOCUMENT:\n{context[:2000]}"
            else:
                # Find matching intent
                matched_intent = None
                for intent, keywords in intent_keys.items():
                    if any(keyword in lower_q for keyword in keywords):
                        matched_intent = intent
                        break
                
                if matched_intent:
                    section_text = match_section(matched_intent, st.session_state.sections, st.session_state.structured_data)
                    context_msg = f"FOCUSED ANALYSIS ({matched_intent}):\n{section_text}\n\nCRM DATA:\n{st.session_state.structured_data}\n\nFULL CONTEXT:\n{context[:2000]}"
                else:
                    context_msg = f"FULL ANALYSIS:\n{context[:3000]}\n\nCRM DATA:\n{st.session_state.structured_data}"

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
st.markdown("**💡 Pro Tip:** Use the CRM data export feature to integrate directly with your Zoho CRM dealflow module.")
