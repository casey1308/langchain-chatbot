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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY") # Optional: used for web search

# --- Environment Variable Checks ---
if not openai_api_key:
    st.error("❌ **OPENAI_API_KEY** not found in your environment variables or .env file.")
    st.error("Please add it to your `.env` file or set it as an environment variable to use Augmento.")
    st.stop() # Halts the Streamlit app if the key is missing

if not serpapi_key:
    st.warning("⚠️ **SERPAPI_API_KEY** not found. Web search functionality will be disabled.")

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes or resets all necessary session state variables."""
    defaults = {
        "chat_history": [],
        "parsed_doc": None,
        "file_uploaded": False,
        "sections": {},
        "crm_data": None,
        "comprehensive_analysis": None, # Stores the generated comprehensive analysis
        "uploaded_file_name": None,
        "active_chat_index": None, # To highlight specific chat in history
        "page": "Dashboard", # Control sidebar navigation: 'Dashboard', 'CRM Management', 'AI Chat', 'Document Viewer'
        "editable_crm_data": None # To allow editing CRM data in UI
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Utility Functions ---

def clean_text(text):
    """Advanced text cleaning for better parsing"""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text) # Remove hyphenated line breaks
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text) # Add space between lowercase and uppercase
    text = re.sub(r"(\d+)([A-Za-z])", r"\1 \2", text) # Add space between numbers and letters
    text = re.sub(r"([A-Za-z])(\d+)", r"\1 \2", text) # Add space between letters and numbers
    text = re.sub(r"\n{3,}", "\n\n", text) # Reduce multiple newlines
    text = re.sub(r"[•◦▪▫‣⁃●■►]", "", text) # Remove common bullet points and special characters
    text = re.sub(r"\s+", " ", text) # Normalize whitespace
    text = re.sub(r"\b([A-Z]{2,})\b", lambda m: m.group(1).title(), text) # Fix ALL CAPS (heuristic)
    text = re.sub(r"\s+([.,:;!?])", r"\1", text) # Fix spacing before punctuation
    return text.strip()

def extract_pdf_text(file_bytes):
    """Extract text from PDF with advanced processing"""
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1}: {e}")
                continue
        
        cleaned_text = clean_text(full_text)
        logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF")
        return cleaned_text
    except PyPDF2.errors.PdfReadError:
        logger.error("PDF extraction failed: File might be corrupted or not a valid PDF.")
        return ""
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

def split_sections(text):
    """Split text into logical sections with improved detection"""
    sections = {"Executive Summary": []}
    current_section = "Executive Summary"
    
    # Updated Comprehensive heading patterns - ordered from more specific to general
    heading_patterns = [
        r"(?i)^(executive\s+summary|overview)",
        r"(?i)^(company\s+overview|about\s+(?:us|the\s+company)|our\s+story|vision|mission)",
        r"(?i)^(team|founder|co-founder|founding\s+team|leadership|management\s+team|our\s+team)",
        r"(?i)^(problem|pain\s+point|market\s+problem|the\s+problem|challenge|customer\s+pain)",
        r"(?i)^(solution|our\s+solution|product|technology|platform|approach|how\s+it\s+works|product\s+overview)",
        r"(?i)^(market|market\s+size|market\s+opportunity|addressable\s+market|tam|sam|som|industry\s+analysis)",
        r"(?i)^(business\s+model|revenue\s+model|monetization|how\s+we\s+make\s+money|pricing)",
        r"(?i)^(competition|competitive\s+landscape|competitors|market\s+analysis|competitive\s+advantage)",
        r"(?i)^(traction|growth|metrics|key\s+metrics|performance|milestones|achievements|progress)",
        r"(?i)^(customers|user\s+base|client\s+base|testimonials|case\s+studies|partnerships)",
        r"(?i)^(financials|financial\s+projections|revenue|sales|unit\s+economics|burn\s+rate|profitability)",
        r"(?i)^(funding|investment|ask|series\s+[a-z0-9]|round|capital|valuation|use\s+of\s+funds)",
        r"(?i)^(cap\s+table|equity|ownership|investor\s+relations|deal\s+terms)",
        r"(?i)^(roadmap|future\s+plans|strategy|vision|goals|objectives)",
        r"(?i)^(go[\s-]?to[\s-]?market|marketing|sales\s+strategy|distribution|acquisition\s+strategy)",
        r"(?i)^(exit\s+strategy|acquisition|ipo|returns|liquidity)",
        r"(?i)^(appendix|disclaimer|contact\s+us|terms\s+and\s+conditions)"
    ]
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        is_heading = False
        for pattern in heading_patterns:
            # Check for patterns and reasonable length to avoid false positives
            if re.match(pattern, line) and len(line) < 70 and len(line.split()) < 8:
                potential_section_name = re.sub(r"^\d+[\.\-\s]*", "", line).strip().title()
                if potential_section_name: # Ensure it's not empty after cleaning
                    current_section = potential_section_name
                    sections[current_section] = []
                    is_heading = True
                    break
        
        # Check for numbered sections (e.g., "1. Introduction", "2. Our Solution") as a separate, strong signal
        if not is_heading and re.match(r"^\d+\.?\s+[\w\s]+", line) and len(line) < 70 and len(line.split()) < 8:
            potential_section_name = re.sub(r"^\d+\.?\s+", "", line).strip().title()
            if potential_section_name:
                current_section = potential_section_name
                sections[current_section] = []
                is_heading = True
        
        if not is_heading:
            sections.setdefault(current_section, []).append(line)
    
    # Post-processing: remove empty sections and join content
    cleaned_sections = {}
    for k, v in sections.items():
        content = "\n".join(v).strip()
        if content:
            cleaned_sections[k] = content
            
    # Ensure "Executive Summary" has content or move content from the first real section
    if "Executive Summary" in cleaned_sections and not cleaned_sections["Executive Summary"]:
        first_meaningful_section = next((name for name in cleaned_sections if name != "Executive Summary" and cleaned_sections[name]), None)
        if first_meaningful_section:
            cleaned_sections["Executive Summary"] = cleaned_sections[first_meaningful_section]
            logger.info(f"Executive Summary was empty; populated it with content from '{first_meaningful_section}'.")
            
    return cleaned_sections


def default_crm_data():
    """Returns a dictionary with default 'Not found' values for CRM fields."""
    return {
        'company_name': 'Not found',
        'ask': 'Not found',
        'revenue': 'Not found',
        'valuation': 'Not found',
        'sector': 'Not found',
        'stage': 'Not found',
        'prior_funding': 'Not found',
        'source': 'Pitch Deck Upload',
        'assign': 'Not found', # For founder/team names
        'description': 'Not found',
        'received_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def normalize_currency_value(value_str):
    """
    Normalizes a currency string (e.g., "$6.7M", "₹5Cr", "100K ARR") into a more consistent format.
    Keeps it as a string for display, but tries to make it cleaner.
    """
    if not isinstance(value_str, str):
        return str(value_str) # Convert numbers to string
    
    value_str = value_str.strip().replace(",", "") # Remove commas
    
    # Handle currency symbols
    currency_symbol = ""
    if '$' in value_str: currency_symbol = '$'
    elif '₹' in value_str: currency_symbol = '₹'
    elif '€' in value_str: currency_symbol = '€'
    elif '£' in value_str: currency_symbol = '£'
    value_str = value_str.replace('$', '').replace('₹', '').replace('€', '').replace('£', '').strip()

    # Handle common abbreviations (case-insensitive)
    value_str = re.sub(r'k(?:ilo)?', 'K', value_str, flags=re.IGNORECASE)
    value_str = re.sub(r'm(?:illion)?', 'M', value_str, flags=re.IGNORECASE)
    value_str = re.sub(r'b(?:illion)?', 'B', value_str, flags=re.IGNORECASE)
    value_str = re.sub(r'cr(?:ore)?', 'Cr', value_str, flags=re.IGNORECASE)
    value_str = re.sub(r'l(?:akh)?', 'L', value_str, flags=re.IGNORECASE)
    
    # Extract numerical part
    num_match = re.match(r'(\d+(?:\.\d+)?)', value_str)
    if num_match:
        number = num_match.group(1)
        # Extract remaining text (units like K, M, B, Cr, ARR, etc.)
        unit = value_str[num_match.end():].strip()
        return f"{currency_symbol}{number}{unit}"
    
    return f"{currency_symbol}{value_str}" # Return with symbol if no number found, or original

def extract_crm_structured_data(text):
    """
    Extracts structured CRM data using the OpenAI LLM.
    Returns raw LLM output string or None on error.
    """
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)
        
        extraction_prompt = f"""
        Extract the following information from this pitch deck text.
        Return ONLY the key-value pairs in the exact format shown below.
        
        company_name: [Exact Company Name. If not explicit, extract from common headers or context.]
        ask: [Total funding amount requested, e.g., "$2M", "₹5Cr", "Not found". Include currency and units.]
        revenue: [Current or latest reported revenue. Specify type if possible (e.g., "$100K ARR", "₹50L Monthly Revenue"). Use "Not found" if not present.]
        valuation: [Company valuation mentioned, e.g., "$10M pre-money", "₹75Cr post-money". Use "Not found" if not present.]
        sector: [Industry sector, e.g., "FinTech", "HealthTech", "SaaS", "E-commerce", "AI". Use "Not found" if not clearly stated.]
        stage: [Company's current stage, e.g., "Seed", "Series A", "Growth", "Pre-Seed", "Product-Market Fit", "Not found".]
        prior_funding: [Previous funding details, e.g., "₹2Cr Seed in 2022", "$500K Pre-Seed". Use "Not found" if no prior funding or details given.]
        source: Pitch Deck Upload
        assign: [Names of Founders/Key Leadership with their primary roles, e.g., "John Doe (CEO), Jane Smith (CTO)". Use "Not found" if not explicit.]
        description: [A concise, factual 1-2 sentence description of what the company does or its core offering. Avoid marketing fluff. Use "Not found" if unclear.]
        
        INSTRUCTIONS:
        - Read the entire text carefully to find the most accurate and precise values.
        - **If a piece of information is NOT found, you MUST output "Not found" for that specific field.**
        - For financial amounts (ask, revenue, valuation, prior_funding), include all currency symbols ($ ₹ € £) and magnitude units (K, M, B, Cr, L).
        - Ensure numerical values are exact from the text, including decimals.
        - For 'description', focus on the core business, not aspirations or future plans.
        - Be extremely literal with the provided text for extraction.
        
        Text to analyze:
        """
        
        messages = [
            SystemMessage(content="You are an expert data extraction bot for VC pitch decks. Extract exactly the requested information in the specified format."),
            HumanMessage(content=f"{extraction_prompt}\n\n{text}")
        ]
        
        response = llm.invoke(messages)
        return response.content.strip()
    
    except RateLimitError:
        logger.warning("OpenAI Rate Limit Exceeded during CRM extraction.")
        st.error("Too many requests for CRM extraction. Please wait a moment and try again.")
        return None
    except APIError as e:
        logger.error(f"OpenAI API Error during CRM extraction: {e}")
        st.error(f"An API error occurred during CRM extraction: {e.args[0]}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during CRM extraction: {e}")
        st.error(f"An unexpected error occurred during CRM extraction: {e}")
        return None

def parse_crm_data(structured_text):
    """
    Parses the structured text output from extract_crm_structured_data into a dictionary.
    Handles missing fields by assigning 'Not found'.
    Also applies normalization to financial values.
    """
    crm_data = default_crm_data()
    if not structured_text:
        return crm_data

    lines = structured_text.split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            # Use get_close_matches for robustness against slight LLM variations in key names
            close_match_key = get_close_matches(key, crm_data.keys(), n=1, cutoff=0.8)
            if close_match_key:
                crm_data_key = close_match_key[0]
                
                # Apply normalization for specific financial fields
                if crm_data_key in ['ask', 'revenue', 'valuation', 'prior_funding']:
                    processed_value = normalize_currency_value(value)
                else:
                    processed_value = value
                
                crm_data[crm_data_key] = processed_value if processed_value and processed_value.lower() not in ["not found", "not mentioned", "n/a", "no information"] else "Not found"
    
    return crm_data

def regex_fallback_extraction(text):
    """
    Fallback extraction using robust regex patterns for key financial figures.
    This is less reliable than LLM but provides a safety net.
    """
    fallback_data = default_crm_data()
    text_lower = text.lower() # Work with lowercase for regex

    # Company Name: Often in the first 1000 characters, capitalized, could be a common entity
    # This is a very basic heuristic, LLM is better for company name.
    company_name_match = re.search(r'^[A-Z][a-zA-Z0-9\s&.,-]{3,60}(?:\n|$)', text, re.MULTILINE)
    if company_name_match:
        fallback_data['company_name'] = company_name_match.group(0).strip()
    
    # Pattern for Ask (Funding)
    ask_patterns = [
        r'(?:seeking|raising|target|funding|investment|ask|we are seeking)\s+(?:of)?\s*([₹$€£]?\s*\d[\d,\.]*\s*(?:[kmbcrt]|\s*thousand|\s*million|\s*billion|\s*crore|\s*lakh)?(?:[a-z]{0,2})?)', # Handles "$10M", "10 million", "₹5Cr"
        r'(?:a|our)\s+(?:seed|series\s+[a-z])\s*(?:round)?\s*(?:of|for)\s*([₹$€£]?\s*\d[\d,\.]*\s*(?:[kmbcrt]|\s*thousand|\s*million|\s*billion|\s*crore|\s*lakh)?)',
        r'(?:looking to raise|require)\s*(?:up to)?\s*([₹$€£]?\s*\d[\d,\.]*\s*(?:[kmbcrt]|\s*thousand|\s*million|\s*billion|\s*crore|\s*lakh)?)'
    ]
    for pattern in ask_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fallback_data['ask'] = normalize_currency_value(match.group(1))
            break
            
    # Pattern for Revenue
    revenue_patterns = [
        r'(?:current|annual|monthly|projected)\s*(?:revenue|sales|income|earnings|arr|mrr|grr|nrr)\s*(?:of)?\s*([₹$€£]?\s*\d[\d,\.]*\s*(?:[kmbcrt]|\s*thousand|\s*million|\s*billion|\s*crore|\s*lakh)?(?:[\s]*arr|mrr)?)',
        r'([₹$€£]?\s*\d[\d,\.]*\s*(?:[kmbcrt]|\s*thousand|\s*million|\s*billion|\s*crore|\s*lakh)?)\s*(?:annually|monthly)\s*(?:revenue|sales)?',
        r'last year revenue\s*([₹$€£]?\s*\d[\d,\.]*\s*(?:[kmbcrt]|\s*thousand|\s*million|\s*billion|\s*crore|\s*lakh)?)'
    ]
    for pattern in revenue_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fallback_data['revenue'] = normalize_currency_value(match.group(1))
            break

    # Pattern for Valuation
    valuation_patterns = [
        r'(?:company\s*)?valuation\s*(?:of)?\s*([₹$€£]?\s*\d[\d,\.]*\s*(?:[kmbcrt]|\s*thousand|\s*million|\s*billion|\s*crore|\s*lakh)?(?:[\s]*(?:pre|post)[\s-]?money)?)',
        r'(?:is\s*)?valued\s+at\s*([₹$€£]?\s*\d[\d,\.]*\s*(?:[kmbcrt]|\s*thousand|\s*million|\s*billion|\s*crore|\s*lakh)?)',
        r'(?:pre|post)[\s-]?money\s*(?:valuation)?\s*(?:of)?\s*([₹$€£]?\s*\d[\d,\.]*\s*(?:[kmbcrt]|\s*thousand|\s*million|\s*billion|\s*crore|\s*lakh)?)'
    ]
    for pattern in valuation_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fallback_data['valuation'] = normalize_currency_value(match.group(1))
            break
            
    return fallback_data

def extract_crm_data_with_fallback(text):
    """
    Orchestrates CRM data extraction, prioritizing LLM and falling back to regex.
    """
    crm_data = default_crm_data()
    
    # Try LLM extraction first
    structured_text_llm = extract_crm_structured_data(text)
    if structured_text_llm:
        crm_data_llm = parse_crm_data(structured_text_llm)
        # Only update if LLM provides a non-empty result for a field that wasn't 'Not found' from LLM
        for key, value in crm_data_llm.items():
            if value != 'Not found':
                crm_data[key] = value
    
    # Fallback to regex for specific fields if LLM returned "Not found" or no info
    fallback_candidates = ['company_name', 'ask', 'revenue', 'valuation']
    if any(crm_data.get(field) == 'Not found' for field in fallback_candidates):
        logger.info("Some CRM fields were 'Not found' by LLM, attempting regex fallback.")
        regex_data = regex_fallback_extraction(text)
        for field in fallback_candidates:
            if crm_data.get(field) == 'Not found' and regex_data.get(field) != 'Not found':
                crm_data[field] = regex_data[field]
                logger.info(f"Filled '{field}' with regex fallback: {crm_data[field]}")
                
    return crm_data

def is_specific_crm_query(query):
    """Identifies if a user query is asking for a specific CRM field."""
    query_lower = query.lower()
    specific_keywords = {
        'company_name': ['company name', 'what company', 'name of company', 'about the company'],
        'ask': ['ask', 'funding', 'investment', 'raise', 'capital', 'seeking', 'round', 'how much money'],
        'revenue': ['revenue', 'sales', 'income', 'earnings', 'arr', 'mrr'],
        'valuation': ['valuation', 'worth', 'valued', 'pre-money', 'post-money'],
        'sector': ['sector', 'industry', 'domain'],
        'stage': ['stage', 'development stage', 'current stage'],
        'prior_funding': ['prior funding', 'previous funding', 'past investments', 'funding history'],
        'assign': ['founder', 'ceo', 'team', 'who founded', 'leadership', 'management', 'team members', 'executive team'],
        'description': ['what do they do', 'what does the company do', 'business', 'product', 'service', 'company description']
    }
    for field, keywords in specific_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return field
    return None

def generate_crm_response(field, crm_data):
    """Generates a focused response for a specific CRM field query."""
    if not crm_data:
        return "No CRM data available. Please upload a pitch deck first."
    
    value = crm_data.get(field, "Not found")
    display_field_name = field.replace('_', ' ').title() # Make it human-readable
    
    if not value or value == "Not found":
        return f"**{display_field_name}:** Not mentioned explicitly in the pitch deck or could not be extracted by AI."
    
    return f"**{display_field_name}:** {value}"

def extract_comprehensive_analysis(text):
    """
    Generates a comprehensive VC analysis using the OpenAI LLM.
    Caches the result in session state.
    """
    try:
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.2) # Slightly higher temperature for more creative analysis
        
        analysis_prompt = """
        Conduct a comprehensive VC analysis of this pitch deck. Focus on providing actionable insights for an investor.
        Structure your analysis using Markdown headings and bullet points as follows:
        
        ## EXECUTIVE SUMMARY
        - Briefly summarize the investment opportunity in 2-3 sentences.
        - List 3-5 key strengths of the company/pitch.
        - List 3-5 key concerns or weaknesses that an investor should be aware of.
        - Provide a clear Investment Recommendation: **"Strong Buy"**, **"Consider"**, or **"Pass"**. Justify it briefly (1-2 sentences).

        ## COMPANY & TEAM ANALYSIS
        - **Business Model & Value Proposition:** Clearly explain what they do, how they plan to generate revenue, and what unique value they offer.
        - **Founders & Team:** Assess the team's relevant experience, expertise, and their suitability to execute the plan. Highlight any gaps.
        - **Competitive Positioning:** How do they differentiate from current and potential competitors? What are their sustainable competitive advantages?

        ## MARKET OPPORTUNITY
        - **Market Size & Growth:** Describe the target market, its current size (e.g., TAM, SAM, SOM if mentioned), and projected growth potential.
        - **Market Timing & Trends:** Is this the right time for this solution? What macro or micro trends are they leveraging or impacted by?
        - **Go-to-Market Strategy:** Detail their strategy for acquiring customers and scaling distribution. Is it effective and cost-efficient?

        ## FINANCIAL ANALYSIS (if data is present)
        - **Revenue & Traction:** Summarize current revenue, growth metrics (e.g., ARR, MRR, user growth), and key historical milestones achieved.
        - **Unit Economics & Scalability:** Discuss the profitability per unit/customer. Is the business model scalable? What are the key cost drivers?
        - **Funding & Use of Funds:** What specific amount are they asking for, and how will the capital be utilized (e.g., hiring, marketing, R&D)? Is the ask justified?

        ## RISK ASSESSMENT
        - **Market & Competitive Risks:** External threats (e.g., market saturation, new disruptive entrants, changing customer preferences).
        - **Execution & Team Risks:** Internal challenges (e.g., ability to deliver on roadmap, team cohesion, key person dependency).
        - **Financial & Regulatory Risks:** Dependency on future funding, potential cash burn issues, legal/compliance hurdles.

        ## INVESTMENT DECISION & NEXT STEPS
        - **Key Questions for Due Diligence:** List 3-5 critical, probing questions an investor should ask in a follow-up.
        - **Recommended Next Steps:** What specific action should the investor take next (e.g., "Schedule follow-up call with founders to discuss financials", "Request detailed financial model and cap table", "Pass for now due to market saturation")?
        
        Be specific, reference information from the pitch deck where possible, and maintain a professional VC tone.
        """
        
        messages = [
            SystemMessage(content="You are an expert Venture Capital Analyst. Provide thorough, insightful, and actionable investment analysis based solely on the provided pitch deck text. Use clear markdown formatting."),
            HumanMessage(content=f"{analysis_prompt}\n\nPitch Deck Text:\n{text}")
        ]
        
        response = llm.invoke(messages)
        return response.content
    
    except RateLimitError:
        logger.warning("OpenAI Rate Limit Exceeded during comprehensive analysis.")
        return "Error: Rate limit exceeded. Please try again in a few minutes."
    except APIError as e:
        logger.error(f"OpenAI API Error during comprehensive analysis: {e}")
        return f"Error: Failed to generate analysis due to an OpenAI API issue: {e.args[0]}. Please check your API key."
    except Exception as e:
        logger.error(f"Unexpected error during comprehensive analysis: {e}")
        return f"Error: Could not generate comprehensive analysis due to an unexpected issue: {str(e)}"

def match_section_content(query, sections):
    """
    Finds the most relevant section content for a given query.
    Prioritizes direct matches, then fuzzy matches on section names, then content search.
    """
    query_lower = query.lower()
    
    # Try direct match with section names first
    for section_name, content in sections.items():
        if query_lower in section_name.lower():
            return content

    # Use a predefined mapping for common queries to section types
    keyword_to_section_map = {
        "founder": ["founder", "team", "leadership", "management"],
        "valuation": ["valuation", "funding", "investment", "financials"],
        "ask": ["ask", "funding", "investment"],
        "market": ["market", "opportunity"],
        "problem": ["problem"],
        "solution": ["solution", "product", "technology"],
        "traction": ["traction", "revenue", "growth", "metrics", "financials"],
        "competition": ["competition"],
        "business model": ["business model", "revenue model"],
        "go-to-market": ["go-to-market", "marketing", "strategy"]
    }
    
    # Check if query contains keywords directly mapping to known section types
    for key_phrase, target_section_keywords in keyword_to_section_map.items():
        if key_phrase in query_lower:
            for target_keyword in target_section_keywords:
                for section_name, content in sections.items():
                    if target_keyword in section_name.lower():
                        return content # Return the first good section match

    # Fuzzy matching against all section names as a fallback
    section_names_lower = [k.lower() for k in sections.keys()]
    matches = get_close_matches(query_lower, section_names_lower, n=1, cutoff=0.6)
    if matches:
        matched_section_name = next(k for k in sections.keys() if k.lower() == matches[0])
        return sections[matched_section_name]
    
    # If no section name matches, perform a simple content search (last resort)
    for section_name, content in sections.items():
        if query_lower in content.lower():
            snippet_start = max(0, content.lower().find(query_lower) - 200)
            snippet_end = min(len(content), snippet_start + len(query_lower) + 300)
            return f"**Found relevant information in the '{section_name}' section:**\n\n...{content[snippet_start:snippet_end]}..."

    return "The requested information could not be found in the pitch deck sections."

def search_serpapi(query):
    """
    Performs a web search using SERP API and summarizes the results with LLM.
    """
    if not serpapi_key:
        return "Web search is disabled. Please configure your SERPAPI_API_KEY in the `.env` file."
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": serpapi_key,
            "num": 5
        }
        r = requests.get("https://serpapi.com/search", params=params)
        r.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = r.json()
        
        if "organic_results" not in data or not data["organic_results"]:
            return "No relevant web search results found."
        
        combined_snippets = []
        for res in data["organic_results"]:
            title = res.get("title", "No Title")
            snippet = res.get("snippet", "No Snippet")
            link = res.get("link", "#")
            combined_snippets.append(f"Title: {title}\nSnippet: {snippet}\nSource: {link}")
        
        full_search_context = "\n\n".join(combined_snippets[:3])
        
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)
        
        if any(x in query.lower() for x in ["founder", "background", "education", "experience", "linkedin", "team"]):
            search_prompt = f"""Summarize the following search results to provide a comprehensive background for a founder/team, covering:
            - Educational background
            - Professional experience
            - Notable achievements/ventures
            - Industry expertise
            
            Search Results:\n{full_search_context}"""
        else:
            search_prompt = f"""Summarize the following web search results concisely for a VC analysis context.
            Focus on key insights directly relevant to the query '{query}'.
            
            Search Results:\n{full_search_context}"""
        
        response = llm.invoke(search_prompt)
        return f"🔍 **Web Search Results for '{query}':**\n\n{response.content.strip()}"
    
    except requests.exceptions.RequestException as req_err:
        logger.error(f"SERP API request failed: {req_err}")
        return f"❌ Web search request failed: {req_err}. Please check your internet connection or SERP API key."
    except Exception as e:
        logger.error(f"Web search failed unexpectedly: {str(e)}")
        return f"❌ Web search failed unexpectedly: {str(e)}"

# --- Main Chat Orchestration Function ---
def chat_with_ai(user_input):
    """
    Orchestrates AI responses based on user input, leveraging document analysis,
    CRM data, and external web search.
    """
    if not st.session_state.file_uploaded:
        return "Please upload a pitch deck first to start analyzing."

    user_input_lower = user_input.lower()

    # 1. Handle specific CRM data queries
    crm_field = is_specific_crm_query(user_input_lower)
    if crm_field:
        return generate_crm_response(crm_field, st.session_state.crm_data)
    
    # 2. Handle Comprehensive Analysis request
    if "comprehensive analysis" in user_input_lower or "investment recommendation" in user_input_lower or "full report" in user_input_lower:
        if st.session_state.comprehensive_analysis:
            return "Here is the comprehensive VC analysis already generated:\n\n" + st.session_state.comprehensive_analysis
        else:
            # If not yet generated, trigger it
            with st.spinner("Generating comprehensive analysis..."):
                st.session_state.comprehensive_analysis = extract_comprehensive_analysis(st.session_state.parsed_doc)
                if st.session_state.comprehensive_analysis.startswith("Error"):
                     return st.session_state.comprehensive_analysis
                return "Analysis generated. Please see the 'Comprehensive Analysis' page for details.\n\n" + st.session_state.comprehensive_analysis[:500] + "..." # Return snippet in chat
            

    # 3. Handle Web Search requests
    if any(term in user_input_lower for term in ["search web for", "google for", "find info on", "external search", "linkedin for", "background of", "web search"]):
        search_query_base = user_input_lower.replace("search web for", "").replace("google for", "").replace("find info on", "").replace("external search", "").replace("linkedin for", "").replace("background of", "").replace("web search", "").strip()
        
        if not search_query_base: # If user just said "web search", prompt for more info
            return "What would you like me to search the web for? Please be specific (e.g., 'web search for founder's linkedin')."

        # Augment search query with company name if available
        if st.session_state.crm_data and st.session_state.crm_data.get('company_name') != 'Not found':
            search_query = f"{st.session_state.crm_data['company_name']} {search_query_base}"
        else:
            search_query = search_query_base
        
        return search_serpapi(search_query)

    # 4. General document-based questions
    try:
        llm = ChatOpenai(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.1)
        
        relevant_content = match_section_content(user_input, st.session_state.sections)
        
        system_message_content = """
        You are Augmento AI, a specialized assistant for Venture Capitalists, analyzing pitch decks.
        Answer the user's question precisely and concisely, using information found *only* in the pitch deck.
        If the information is not explicitly present in the provided document sections or the general document text, clearly state that you couldn't find it in the deck.
        Avoid making up information. Be professional and to the point.
        """
        
        human_message_content = f"""
        User Question: {user_input}
        
        Relevant Pitch Deck Content:
        ---
        {relevant_content}
        ---
        
        Based on the "Relevant Pitch Deck Content" provided above, and the full document context, answer the user's question.
        """
        
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=human_message_content)
        ]
        
        response = llm.invoke(messages)
        return response.content
        
    except RateLimitError:
        logger.warning("OpenAI Rate Limit Exceeded during chat.")
        return "❌ I'm currently experiencing high demand. Please try asking your question again in a moment."
    except APIError as e:
        logger.error(f"OpenAI API error in chat: {e}")
        return f"❌ An OpenAI API error occurred: {e.args[0]}. Please ensure your API key is valid and not expired."
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        return f"❌ Sorry, I couldn't process that request. An unexpected error occurred: {str(e)}"

# --- Streamlit UI Configuration ---

st.set_page_config(
    page_title="Augmento - Smart Pitch Evaluator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded" # Start with sidebar expanded
)

# Custom CSS for a more polished look
st.markdown("""
<style>
    .stApp {
        background-color: #0c1015; /* Darker background */
        color: #e0e0e0; /* Lighter text */
    }
    .stButton > button {
        background-color: #2e7d32; /* Green for primary actions */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #388e3c;
        transform: translateY(-2px);
    }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div {
        background-color: #1a1e24; /* Darker input fields */
        color: #e0e0e0;
        border-radius: 8px;
        border: 1px solid #333;
    }
    .stExpander {
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #1a1e24;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #76ff03; /* Bright green for headings */
    }
    /* Style for the sidebar radio buttons */
    .stSidebar .stRadio div[role="radiogroup"] > label {
        padding: 8px 10px;
        border-radius: 8px;
        margin-bottom: 5px;
        transition: background-color 0.2s ease-in-out;
    }
    .stSidebar .stRadio div[role="radiogroup"] > label:hover {
        background-color: #1a1e24; /* Darker hover */
    }
    .stSidebar .stRadio div[role="radiogroup"] > label[data-baseweb="radio"] > div {
        color: #e0e0e0; /* Radio button text color */
    }
    /* Highlight selected radio button */
    .stSidebar .stRadio div[role="radiogroup"] > label[aria-checked="true"] {
        background-color: #333; /* Darker highlight */
        border: 1px solid #76ff03; /* Green border for selected */
    }
    /* Specific style for success/error/warning messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 8px;
        padding: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    /* Hide default Streamlit footer/header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Augmento - Smart Pitch Evaluator")
st.markdown("---")

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=AugmentoLogo", use_container_width=True) # Placeholder logo
    st.subheader("Navigation")
    
    # Use radio buttons for page selection
    page_selection = st.radio(
        "Go to",
        ('Dashboard', 'CRM Management', 'AI Chat', 'Document Viewer', 'Comprehensive Analysis'),
        key="main_page_selector"
    )
    st.session_state.page = page_selection

    st.markdown("---")
    st.subheader("Pitch Deck Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed", key="sidebar_pdf_uploader")

    if uploaded_file and not st.session_state.file_uploaded:
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.file_uploaded = True  # Prevent future reprocessing

    with st.spinner(f"🔄 Processing '{uploaded_file.name}'... This may take a moment."):
        file_bytes = uploaded_file.read()
        text = extract_pdf_text(file_bytes)

        if text:
            st.session_state.parsed_doc = text
            st.session_state.sections = split_sections(text)
            st.session_state.crm_data = extract_crm_data_with_fallback(text)
            st.session_state.editable_crm_data = dict(st.session_state.crm_data)
            st.success("✅ Pitch deck processed! Navigate using the options above.")
        else:
            st.session_state.file_uploaded = False
            st.error("❌ Failed to extract text from PDF. Please ensure it's a searchable PDF (not just an image scan).")

        # Reset states for a new upload
        st.session_state.file_uploaded = False
        st.session_state.parsed_doc = None
        st.session_state.sections = {}
        st.session_state.crm_data = None
        st.session_state.comprehensive_analysis = None
        st.session_state.chat_history = []
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.page = "Dashboard" # Redirect to dashboard on new upload

        with st.spinner(f"🔄 Processing '{uploaded_file.name}'... This may take a moment."):
            file_bytes = uploaded_file.read()
            text = extract_pdf_text(file_bytes)
            
            if text:
                st.session_state.parsed_doc = text
                st.session_state.sections = split_sections(text)
                st.session_state.file_uploaded = True
                
                # CRM data extraction happens immediately upon successful text extraction
                st.session_state.crm_data = extract_crm_data_with_fallback(text)
                st.session_state.editable_crm_data = dict(st.session_state.crm_data) # Populate editable copy
                
                st.success("✅ Pitch deck processed! Navigate using the options above.")
                
            else:
                st.error("❌ Failed to extract text from PDF. Please ensure it's a searchable PDF (not just an image scan).")
                st.session_state.file_uploaded = False
    
    if st.session_state.file_uploaded:
       st.success(f"Loaded: **{st.session_state.uploaded_file_name}**")
       st.write(f"Document Length: {len(st.session_state.parsed_doc):,} chars")
       st.write(f"Sections found: {len(st.session_state.sections)}")
    else:
       st.info("No pitch deck loaded.")
    
    st.markdown("---")
    st.caption("Powered by OpenAI and Streamlit")

# --- Main Content Area based on page selection ---
if not st.session_state.file_uploaded and st.session_state.page != "Dashboard":
    st.warning("Please upload a pitch deck PDF in the sidebar to access analysis features.")
    st.session_state.page = "Dashboard" # Force user back to dashboard if no file is uploaded


if st.session_state.page == "Dashboard":
    st.header("🏠 Dashboard Overview")
    if st.session_state.file_uploaded:
        st.markdown(f"### Currently Analyzing: **{st.session_state.uploaded_file_name}**")
        
        col_metrics, col_description = st.columns(2)
        
        with col_metrics:
            st.subheader("Key Investment Metrics")
            if st.session_state.crm_data:
                st.metric("Company Name", st.session_state.crm_data.get('company_name', 'Not found'))
                st.metric("Funding Ask", st.session_state.crm_data.get('ask', 'Not found'))
                st.metric("Current Revenue", st.session_state.crm_data.get('revenue', 'Not found'))
                st.metric("Valuation", st.session_state.crm_data.get('valuation', 'Not found'))
                st.metric("Sector", st.session_state.crm_data.get('sector', 'Not found'))
                st.metric("Stage", st.session_state.crm_data.get('stage', 'Not found'))
            else:
                st.info("CRM data not available. Ensure API key is valid and document is processed.")
        
        with col_description:
            st.subheader("Company Description")
            if st.session_state.crm_data and st.session_state.crm_data.get('description') != 'Not found':
                st.write(st.session_state.crm_data['description'])
            else:
                st.info("Company description not found in the pitch deck.")
            
            st.subheader("Quick Actions")
            if st.button("Start Chat with AI", use_container_width=True):
                st.session_state.page = "AI Chat"
        
            if st.button("Generate Comprehensive Analysis", use_container_width=True):
                st.session_state.page = "Comprehensive Analysis"
                st.session_state.comprehensive_analysis = None # Clear to force generation

            if st.button("View Raw Document Sections", use_container_width=True):
                st.session_state.page = "Document Viewer"
    else:
        st.info("Welcome to Augmento! Please upload a pitch deck PDF using the uploader in the sidebar to get started.")
        st.image("https://via.placeholder.com/700x350?text=Upload+Your+Pitch+Deck+to+Begin", use_container_width=True, caption="Augmento: Your Smart Pitch Evaluator")

elif st.session_state.page == "CRM Management":
    st.header("📋 CRM Data Management")
    if st.session_state.file_uploaded and st.session_state.crm_data:
        st.write("Review and edit the extracted CRM data below. This data will be used for exporting.")

        # Ensure editable_crm_data is initialized from crm_data
        if st.session_state.editable_crm_data is None:
            st.session_state.editable_crm_data = dict(st.session_state.crm_data)

        with st.form("crm_edit_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Key Information")
                st.session_state.editable_crm_data['company_name'] = st.text_input(
                    "Company Name", st.session_state.editable_crm_data.get('company_name', ''))
                st.session_state.editable_crm_data['ask'] = st.text_input(
                    "Funding Ask", st.session_state.editable_crm_data.get('ask', ''))
                st.session_state.editable_crm_data['revenue'] = st.text_input(
                    "Revenue", st.session_state.editable_crm_data.get('revenue', ''))
                st.session_state.editable_crm_data['valuation'] = st.text_input(
                    "Valuation", st.session_state.editable_crm_data.get('valuation', ''))
                st.session_state.editable_crm_data['sector'] = st.text_input(
                    "Sector", st.session_state.editable_crm_data.get('sector', ''))
                st.session_state.editable_crm_data['stage'] = st.text_input(
                    "Stage", st.session_state.editable_crm_data.get('stage', ''))
            
            with col2:
                st.subheader("Additional Details")
                st.session_state.editable_crm_data['prior_funding'] = st.text_input(
                    "Prior Funding", st.session_state.editable_crm_data.get('prior_funding', ''))
                st.session_state.editable_crm_data['assign'] = st.text_area(
                    "Assign (Founders/Team)", st.session_state.editable_crm_data.get('assign', ''), height=80)
                st.session_state.editable_crm_data['description'] = st.text_area(
                    "Description", st.session_state.editable_crm_data.get('description', ''), height=120)
                st.session_state.editable_crm_data['source'] = st.text_input(
                    "Source", st.session_state.editable_crm_data.get('source', 'Pitch Deck Upload'), disabled=True)
                st.session_state.editable_crm_data['received_date'] = st.text_input(
                    "Received Date", st.session_state.editable_crm_data.get('received_date', ''), disabled=True)

            submitted = st.form_submit_button("Update CRM Data", type="primary", use_container_width=True)
            if submitted:
                st.session_state.crm_data = dict(st.session_state.editable_crm_data) # Save changes to main CRM data
                st.success("CRM Data Updated Successfully!")
        
        st.markdown("---")
        st.subheader("⬇️ Export & Integration")
        col_export_json, col_export_zoho = st.columns(2)
        with col_export_json:
            st.download_button(
                label="Download CRM Data (JSON)",
                data=json.dumps(st.session_state.crm_data, indent=2),
                file_name=f"crm_data_{st.session_state.crm_data.get('company_name', 'pitch_deck').replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )
        with col_export_zoho:
            # Zoho webhook integration (optional) - requires a server-side component for real use
            if os.getenv("ZOHO_WEBHOOK_URL"):
                if st.button("🔗 Send to Zoho CRM (WIP)", use_container_width=True):
                    st.info("Sending data to Zoho is a work in progress and requires proper API integration.")
                    # Placeholder for actual Zoho integration
                    # try:
                    #     send_to_zoho_webhook(st.session_state.crm_data, os.getenv("ZOHO_WEBHOOK_URL"))
                    #     st.success("Data sent to Zoho CRM!")
                    # except Exception as e:
                    #     st.error(f"Failed to send to Zoho: {e}")
            else:
                st.info("💡 Tip: Set `ZOHO_WEBHOOK_URL` in your `.env` file to enable direct CRM export.")

    else:
        st.info("Upload a pitch deck PDF in the sidebar to extract and manage CRM data.")

elif st.session_state.page == "AI Chat":
    st.header("💬 Chat with Augmento AI")
    if st.session_state.file_uploaded:
        st.markdown("Ask specific questions about the pitch deck or perform web searches.")

        # Quick Prompts / Suggestions
        st.subheader("💡 Quick Prompts:")
        quick_prompts = [
            "What is the funding ask?", "Who are the founders?",
            "What is the company's valuation?", "What problem does this solve?",
            "What is the business model?", "Search web for founder background",
            "Show me the market opportunity", "What is their traction?",
            "Give me an investment recommendation."
        ]
        
        cols = st.columns(3)
        for i, question in enumerate(quick_prompts):
            with cols[i % 3]:
                if st.button(question, key=f"quick_prompt_{i}"):
                    with st.spinner("🤖 Augmento is thinking..."):
                        response = chat_with_ai(question)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.chat_history.append({"user": question, "ai": response, "time": timestamp})
                        st.session_state.active_chat_index = len(st.session_state.chat_history) - 1
                        # st.rerun() # Removed rerun here to allow typing without immediate refresh, will rerurn on send

        # Main Chat Input
        user_input = st.text_input("Your question:", key="chat_input_main", placeholder="e.g., 'What is their go-to-market strategy?' or 'Web search for competitors of [Company Name]'", on_change=None)
        send_button = st.button("Send Query", type="primary", use_container_width=False) # Keep send button simple

        if send_button and user_input:
            with st.spinner("🤖 Augmento is analyzing..."):
                response = chat_with_ai(user_input)
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append({"user": user_input, "ai": response, "time": timestamp})
                st.session_state.active_chat_index = len(st.session_state.chat_history) - 1
                st.experimental_rerun() # Force rerun to clear input and update chat history
        
        # Display Conversation History
        st.markdown("---")
        st.subheader("Your Conversation History:")
        if not st.session_state.chat_history:
            st.info("No conversations yet. Start by asking a question!")
        else:
            if st.button("🗑️ Clear Chat History", key="clear_chat_history", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.active_chat_index = None
                st.experimental_rerun() # Use experimental_rerun for clearing

            # Display chat history, most recent at top
            for i in range(len(st.session_state.chat_history) -1, -1, -1):
                chat_entry = st.session_state.chat_history[i]
                is_expanded = (i == st.session_state.active_chat_index)
                with st.expander(f"Q: {chat_entry['user'][:70]}{'...' if len(chat_entry['user']) > 70 else ''} ({chat_entry['time']})", expanded=is_expanded):
                    st.markdown(f"**🧑 You:** {chat_entry['user']}")
                    st.markdown(f"**🤖 Augmento AI:**")
                    st.markdown(chat_entry['ai'])
                    st.markdown(f"<small><em>{chat_entry['time']}</em></small>", unsafe_allow_html=True)
    else:
        st.info("Upload a pitch deck PDF in the sidebar to start chatting with Augmento AI.")

elif st.session_state.page == "Comprehensive Analysis":
    st.header("🔍 Comprehensive VC Analysis")
    if st.session_state.file_uploaded:
        st.info("This analysis provides an in-depth assessment from an investor's perspective. It may take up to a minute to generate.")
        
        if st.button("Generate/Refresh Comprehensive Analysis", key="generate_comp_analysis", use_container_width=True):
            st.session_state.comprehensive_analysis = None # Clear existing to force regeneration
            with st.spinner("🔄 Generating comprehensive analysis..."):
                st.session_state.comprehensive_analysis = extract_comprehensive_analysis(st.session_state.parsed_doc)
                if st.session_state.comprehensive_analysis.startswith("Error"):
                    st.error(f"Analysis generation failed. {st.session_state.comprehensive_analysis}")
        
        if st.session_state.comprehensive_analysis:
            st.markdown(st.session_state.comprehensive_analysis)
            st.markdown("---")
            st.download_button(
                label="⬇️ Download Analysis Report (Markdown)",
                data=st.session_state.comprehensive_analysis,
                file_name=f"vc_analysis_report_{st.session_state.crm_data.get('company_name', 'pitch_deck').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
             st.info("Click 'Generate/Refresh Comprehensive Analysis' above to generate the full report.")
    else:
        st.info("Upload a pitch deck PDF in the sidebar to generate a comprehensive VC analysis.")

elif st.session_state.page == "Document Viewer":
    st.header("📑 Pitch Deck Document Sections")
    if st.session_state.file_uploaded:
        st.info("Browse the raw extracted text, split into logical sections. Useful for verifying content and finding specific details.")
        if st.session_state.sections:
            section_options = list(st.session_state.sections.keys())
            
            # Selectbox to choose a section
            selected_section_name = st.selectbox(
                "Select a section to view its content:",
                options=['-- Select a Section --'] + section_options,
                key="section_selector_viewer"
            )

            if selected_section_name and selected_section_name != '-- Select a Section --':
                st.subheader(f"Content of: {selected_section_name}")
                st.text_area(
                    f"Section: {selected_section_name}",
                    st.session_state.sections[selected_section_name],
                    height=400,
                    key=f"section_content_display_{selected_section_name}"
                )
                if st.button(f"Ask AI to analyze this '{selected_section_name}' section", use_container_width=True):
                    with st.spinner("🤖 Analyzing section..."):
                        analysis_query = f"Provide a detailed analysis of the '{selected_section_name}' section."
                        response = chat_with_ai(analysis_query)
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.chat_history.append({"user": analysis_query, "ai": response, "time": timestamp})
                        st.session_state.active_chat_index = len(st.session_state.chat_history) - 1
                        st.session_state.page = "AI Chat" # Navigate to chat to show response
                        
            else:
                st.markdown("Please select a section from the dropdown to view its content.")
        else:
            st.info("No sections available. Please upload a PDF.")
    else:
        st.info("Upload a pitch deck PDF in the sidebar to view its extracted sections.")


st.markdown("---")
st.markdown("Augmento - Your Smart Pitch Evaluator")
st.caption("Version 2.0 | Developed for efficient VC analysis.")


# --- Debug Information (Optional) ---
if st.checkbox("🔧 Show Debug Info (for developers)"):
    st.subheader("Current Session State")
    st.json({k: (v if not (isinstance(v, str) and len(v) > 200) else f"{v[:200]}...") for k, v in st.session_state.items()})

    if st.session_state.parsed_doc:
        st.subheader("Raw Extracted Text Sample (first 2000 characters)")
        st.text_area("Full Raw Text", st.session_state.parsed_doc[:2000], height=300)
