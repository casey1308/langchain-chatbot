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
import hashlib
from typing import Dict, List, Optional, Tuple
from openai import OpenAIError, RateLimitError, APIError

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Configure logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_key = os.getenv("SERPAPI_API_KEY")

if not openai_api_key or not serpapi_key:
    st.error("❌ Add your OPENAI_API_KEY and SERPAPI_API_KEY in .env")
    st.stop()

# Initialize session state with better structure
def initialize_session_state():
    """Initialize session state variables with default values"""
    defaults = {
        "chat_history": [],
        "parsed_documents": {},  # Store multiple parsed documents
        "current_doc_id": None,
        "file_uploaded": False,
        "selected_chat_index": None,
        "show_comprehensive_analysis": False,
        "show_crm_summary": False,
        "selected_section": None,
        "processing_error": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Document data structure
class DocumentData:
    def __init__(self, file_name: str, file_hash: str):
        self.file_name = file_name
        self.file_hash = file_hash
        self.raw_text = ""
        self.sections = {}
        self.crm_data = {}
        self.comprehensive_analysis = None
        self.processing_status = "pending"
        self.error_message = None
        self.created_at = datetime.now().isoformat()
        self.char_count = 0

    def to_dict(self):
        return {
            'file_name': self.file_name,
            'file_hash': self.file_hash,
            'raw_text': self.raw_text,
            'sections': self.sections,
            'crm_data': self.crm_data,
            'comprehensive_analysis': self.comprehensive_analysis,
            'processing_status': self.processing_status,
            'error_message': self.error_message,
            'created_at': self.created_at,
            'char_count': self.char_count
        }

# Enhanced text cleaning with better error handling
def clean_text(text: str) -> str:
    """Advanced text cleaning for better parsing with error handling"""
    if not text or not isinstance(text, str):
        logger.warning("Empty or invalid text provided for cleaning")
        return ""
    
    try:
        # Remove hyphenated line breaks
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        
        # Fix common PDF artifacts
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"(\d+)([A-Za-z])", r"\1 \2", text)
        text = re.sub(r"([A-Za-z])(\d+)", r"\1 \2", text)
        
        # Clean up formatting
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[•◦▪▫‣⁃]", "", text)
        text = re.sub(r"\s+", " ", text)
        
        # Fix common OCR errors
        text = re.sub(r"\b([A-Z]{2,})\b", lambda m: m.group(1).title(), text)
        text = re.sub(r"\s+([.,:;!?])", r"\1", text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text.strip()
    
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text.strip() if text else ""

# Robust PDF extraction with multiple fallback methods
def extract_pdf_text(file_bytes: bytes) -> Tuple[str, bool]:
    """Extract text from PDF with multiple methods and error handling"""
    if not file_bytes:
        logger.error("No file bytes provided")
        return "", False
    
    extraction_methods = [
        _extract_with_pypdf2,
        _extract_with_fallback_method
    ]
    
    for method in extraction_methods:
        try:
            text, success = method(file_bytes)
            if success and text.strip():
                cleaned_text = clean_text(text)
                if len(cleaned_text) > 100:  # Minimum viable text length
                    logger.info(f"Successfully extracted {len(cleaned_text)} characters using {method.__name__}")
                    return cleaned_text, True
                else:
                    logger.warning(f"Text too short from {method.__name__}: {len(cleaned_text)} chars")
        except Exception as e:
            logger.warning(f"Method {method.__name__} failed: {e}")
            continue
    
    logger.error("All extraction methods failed")
    return "", False

def _extract_with_pypdf2(file_bytes: bytes) -> Tuple[str, bool]:
    """Primary extraction method using PyPDF2"""
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        
        if len(reader.pages) == 0:
            return "", False
        
        full_text = ""
        successful_pages = 0
        
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                    successful_pages += 1
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                continue
        
        if successful_pages == 0:
            return "", False
        
        return full_text, True
        
    except Exception as e:
        logger.error(f"PyPDF2 extraction failed: {e}")
        return "", False

def _extract_with_fallback_method(file_bytes: bytes) -> Tuple[str, bool]:
    """Fallback extraction method"""
    try:
        # Simple text extraction attempt
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text_parts = []
        
        for page in reader.pages:
            try:
                # Try different extraction methods
                if hasattr(page, 'extractText'):
                    text = page.extractText()
                else:
                    text = page.extract_text()
                
                if text:
                    text_parts.append(text)
            except:
                continue
        
        combined_text = "\n".join(text_parts)
        return combined_text, len(combined_text) > 50
        
    except Exception as e:
        logger.error(f"Fallback extraction failed: {e}")
        return "", False

# Enhanced section splitting with better pattern detection
def split_sections(text: str) -> Dict[str, str]:
    """Split text into logical sections with improved detection"""
    if not text:
        return {"Executive Summary": "No content available"}
    
    sections = {"Executive Summary": []}
    current_section = "Executive Summary"
    
    # Comprehensive heading patterns with priority order
    heading_patterns = [
        # High-priority patterns (specific startup sections)
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:executive\s+summary|summary)", "Executive Summary"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:about\s+(?:us|the\s+company)|company\s+overview|our\s+story)", "Company Overview"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:founder|co-founder|founding\s+team|leadership|management\s+team|team|our\s+team)", "Team & Founders"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:problem|pain\s+point|market\s+problem|the\s+problem|challenge)", "Problem Statement"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:solution|our\s+solution|product|technology|platform|approach)", "Solution & Product"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:market|market\s+size|market\s+opportunity|addressable\s+market|tam|sam|som)", "Market Opportunity"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:business\s+model|revenue\s+model|monetization|how\s+we\s+make\s+money)", "Business Model"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:traction|growth|metrics|key\s+metrics|performance|milestones|achievements)", "Traction & Growth"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:competition|competitive\s+landscape|competitors|market\s+analysis)", "Competitive Analysis"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:financials|financial\s+projections|revenue|sales|unit\s+economics)", "Financial Projections"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:funding|investment|ask|series|round|capital|valuation|use\s+of\s+funds)", "Funding Request"),
        (r"(?i)^(?:\d+[\.\)]\s*)?(?:roadmap|future\s+plans|strategy|vision|goals|objectives)", "Future Strategy"),
    ]
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for heading patterns
        section_found = False
        for pattern, section_name in heading_patterns:
            if re.match(pattern, line) and len(line) < 150:
                current_section = section_name
                sections[current_section] = []
                section_found = True
                break
        
        # Generic numbered sections
        if not section_found:
            numbered_match = re.match(r"^\d+[\.\)]\s+([A-Z][^.!?]*)", line)
            if numbered_match and len(line) < 100:
                section_title = numbered_match.group(1).strip()
                current_section = section_title
                sections[current_section] = []
                section_found = True
        
        if not section_found:
            sections.setdefault(current_section, []).append(line)
    
    # Clean up sections and remove empty ones
    cleaned_sections = {}
    for section_name, content_lines in sections.items():
        if content_lines:
            content = "\n".join(content_lines).strip()
            if content and len(content) > 20:  # Minimum content length
                cleaned_sections[section_name] = content
    
    # Ensure we have at least one section
    if not cleaned_sections:
        cleaned_sections["Content"] = text[:2000] + "..." if len(text) > 2000 else text
    
    return cleaned_sections

# Enhanced CRM data extraction with better error handling
def extract_crm_structured_data(text: str) -> Optional[str]:
    """Extract CRM-specific structured data with improved accuracy"""
    if not text or len(text) < 100:
        logger.warning("Text too short for meaningful CRM extraction")
        return None
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=openai_api_key, 
            temperature=0,
            timeout=30,
            max_retries=2
        )
        
        # Truncate text if too long to avoid token limits
        if len(text) > 15000:
            text = text[:15000] + "\n[... content truncated for analysis ...]"
        
        extraction_prompt = """
        Extract the following information from this pitch deck text. Return ONLY the values in the exact format shown.
        Be precise and extract exact values. If not found, use "Not found".
        
        company_name: [Company name]
        ask: [Funding amount requested with currency, e.g., "$2M", "₹5Cr"]
        revenue: [Current revenue with currency, e.g., "$100K ARR", "₹50L"]
        valuation: [Company valuation with currency, e.g., "$10M pre-money"]
        sector: [Industry sector, e.g., "FinTech", "HealthTech", "EdTech"]
        stage: [Company stage, e.g., "Seed", "Series A", "Pre-Series A"]
        prior_funding: [Previous funding with details, e.g., "₹2Cr Seed 2022"]
        source: Pitch Deck Upload
        assign: [Founder names and roles, comma-separated]
        description: [What the company does in 1-2 sentences]
        
        CRITICAL INSTRUCTIONS:
        - Extract exact amounts with currency symbols
        - Look for keywords like "asking", "raising", "seeking", "need"
        - Check for revenue mentions with ARR, MRR, monthly, annual
        - Include founder names and their roles
        - Keep description factual and concise
        
        Text to analyze:
        """
        
        messages = [
            SystemMessage(content="You are a precise data extraction expert. Extract only the requested information in the exact format specified."),
            HumanMessage(content=f"{extraction_prompt}\n\n{text}")
        ]
        
        response = llm.invoke(messages)
        return response.content.strip()
        
    except RateLimitError:
        logger.error("OpenAI API rate limit exceeded")
        return None
    except Exception as e:
        logger.error(f"CRM extraction error: {e}")
        return None

# Improved CRM data parsing
def parse_crm_data(structured_text: str) -> Dict[str, str]:
    """Parse structured text into CRM dictionary with validation"""
    crm_data = {}
    
    if not structured_text:
        return default_crm_data()
    
    required_fields = [
        'company_name', 'ask', 'revenue', 'valuation', 'sector', 
        'stage', 'prior_funding', 'source', 'assign', 'description'
    ]
    
    lines = structured_text.split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key in required_fields and value:
                # Clean up common "not found" variations
                if value.lower() not in ["not found", "not mentioned", "n/a", "none", "null", ""]:
                    crm_data[key] = value
                else:
                    crm_data[key] = "Not found"
    
    # Ensure all required fields exist
    for field in required_fields:
        if field not in crm_data:
            crm_data[field] = "Not found"
    
    # Add metadata
    crm_data['received_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return crm_data

def default_crm_data() -> Dict[str, str]:
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
        'description': 'Not found',
        'received_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# Enhanced CRM extraction with fallback
def extract_crm_data_with_fallback(text: str) -> Dict[str, str]:
    """Main CRM extraction with fallback mechanism"""
    try:
        structured_text = extract_crm_structured_data(text)
        if structured_text:
            crm_data = parse_crm_data(structured_text)
            if any(v != "Not found" for k, v in crm_data.items() if k != 'received_date'):
                return crm_data
        
        # Fallback to regex extraction
        logger.info("Falling back to regex extraction")
        return regex_fallback_extraction(text)
        
    except Exception as e:
        logger.error(f"CRM extraction failed: {e}")
        return default_crm_data()

def regex_fallback_extraction(text: str) -> Dict[str, str]:
    """Fallback extraction using regex patterns"""
    crm_data = default_crm_data()
    
    if not text:
        return crm_data
    
    # Enhanced patterns for better extraction
    patterns = {
        'ask': [
            r'(?:seeking|raising|ask|asking|need|require|looking\s+for).*?[₹$]?\s*(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+)',
            r'funding.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+)',
            r'investment.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+)',
            r'[₹$]\s*(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+).*?(?:funding|investment|ask)'
        ],
        'revenue': [
            r'revenue.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+)',
            r'sales.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+)',
            r'(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+)\s*(?:ARR|MRR|annual|monthly|revenue)',
            r'[₹$]\s*(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+).*?revenue'
        ],
        'valuation': [
            r'valuation.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+)',
            r'valued\s+at.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+)',
            r'(?:pre|post)[-\s]money.*?[₹$]?\s*(\d+(?:\.\d+)?)\s*(million|crore|lakh|[KMBCr]+)'
        ],
        'company_name': [
            r'(?:company|startup|firm).*?(?:name|called)\s+(?:is\s+)?([A-Z][A-Za-z\s]+)',
            r'^([A-Z][A-Za-z\s]{2,30})\s+(?:is\s+a|provides|offers|specializes)',
            r'introducing\s+([A-Z][A-Za-z\s]{2,30})'
        ]
    }
    
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    if field == 'company_name':
                        crm_data[field] = match.group(1).strip()
                    else:
                        value = match.group(1)
                        unit = match.group(2) if len(match.groups()) > 1 else ""
                        crm_data[field] = f"${value}{unit}" if unit else f"${value}"
                    break
            except Exception as e:
                logger.warning(f"Regex pattern failed for {field}: {e}")
                continue
    
    return crm_data

# File hash generation for caching
def generate_file_hash(file_bytes: bytes) -> str:
    """Generate hash for file to enable caching"""
    return hashlib.md5(file_bytes).hexdigest()

# Main document processing function
def process_document(file_bytes: bytes, file_name: str) -> Optional[str]:
    """Process document with comprehensive error handling and caching"""
    file_hash = generate_file_hash(file_bytes)
    
    # Check if already processed
    if file_hash in st.session_state.parsed_documents:
        logger.info(f"Document {file_name} already processed (hash: {file_hash})")
        st.session_state.current_doc_id = file_hash
        st.session_state.file_uploaded = True
        return file_hash
    
    # Create new document data
    doc_data = DocumentData(file_name, file_hash)
    
    try:
        # Extract text
        with st.spinner("🔄 Extracting text from PDF..."):
            raw_text, success = extract_pdf_text(file_bytes)
            
            if not success or len(raw_text) < 100:
                doc_data.processing_status = "failed"
                doc_data.error_message = "Failed to extract meaningful text from PDF"
                st.session_state.parsed_documents[file_hash] = doc_data
                return None
            
            doc_data.raw_text = raw_text
            doc_data.char_count = len(raw_text)
        
        # Split into sections
        with st.spinner("🔄 Analyzing document structure..."):
            doc_data.sections = split_sections(raw_text)
            
            if not doc_data.sections:
                doc_data.sections = {"Content": raw_text[:2000]}
        
        # Extract CRM data
        with st.spinner("🔄 Extracting CRM data..."):
            doc_data.crm_data = extract_crm_data_with_fallback(raw_text)
        
        doc_data.processing_status = "completed"
        st.session_state.parsed_documents[file_hash] = doc_data
        st.session_state.current_doc_id = file_hash
        st.session_state.file_uploaded = True
        
        logger.info(f"Successfully processed {file_name} ({doc_data.char_count} chars, {len(doc_data.sections)} sections)")
        return file_hash
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        doc_data.processing_status = "failed"
        doc_data.error_message = str(e)
        st.session_state.parsed_documents[file_hash] = doc_data
        return None

# Get current document data
def get_current_doc() -> Optional[DocumentData]:
    """Get current document data"""
    if st.session_state.current_doc_id and st.session_state.current_doc_id in st.session_state.parsed_documents:
        return st.session_state.parsed_documents[st.session_state.current_doc_id]
    return None

# Enhanced query processing
def is_specific_crm_query(query: str) -> Optional[str]:
    """Identify specific CRM field queries"""
    query_lower = query.lower()
    
    specific_keywords = {
        'ask': ['ask', 'funding', 'investment', 'raise', 'capital', 'seeking', 'round', 'money'],
        'founder': ['founder', 'ceo', 'team', 'who founded', 'leadership', 'management', 'started'],
        'revenue': ['revenue', 'sales', 'income', 'earnings', 'arr', 'mrr', 'money made'],
        'valuation': ['valuation', 'worth', 'valued', 'pre-money', 'post-money', 'value'],
        'company': ['company name', 'what company', 'name of company', 'startup name'],
        'description': ['what do they do', 'what does the company do', 'business', 'product', 'service'],
        'sector': ['sector', 'industry', 'vertical', 'market', 'space'],
        'stage': ['stage', 'series', 'round', 'funding stage']
    }
    
    for field, keywords in specific_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            return field
    
    return None

def generate_crm_response(field: str, crm_data: Dict[str, str]) -> str:
    """Generate focused response for CRM field queries"""
    if not crm_data:
        return "No CRM data available. Please upload a pitch deck first."
    
    field_mapping = {
        'company': 'company_name',
        'founder': 'assign',
        'ask': 'ask',
        'revenue': 'revenue',
        'valuation': 'valuation',
        'description': 'description',
        'sector': 'sector',
        'stage': 'stage'
    }
    
    crm_field = field_mapping.get(field, field)
    value = crm_data.get(crm_field, "Not found")
    
    if not value or value == "Not found":
        return f"**{field.title()}:** Not mentioned in the pitch deck."
    
    return f"**{field.title()}:** {value}"

# Enhanced section matching
def match_section(query: str, sections: Dict[str, str], crm_data: Optional[Dict] = None) -> str:
    """Match query to relevant sections with improved accuracy"""
    query_lower = query.lower()
    
    # First check if it's a CRM-specific query
    if crm_data:
        crm_field = is_specific_crm_query(query)
        if crm_field:
            return generate_crm_response(crm_field, crm_data)
    
    # Section keyword mapping
    section_keywords = {
        "founder": ["founder", "team", "leadership", "management", "ceo", "started"],
        "problem": ["problem", "pain", "challenge", "issue", "difficulty"],
        "solution": ["solution", "product", "technology", "platform", "approach"],
        "market": ["market", "opportunity", "size", "tam", "sam", "som", "addressable"],
        "traction": ["traction", "growth", "metrics", "performance", "achievement"],
        "financial": ["financial", "revenue", "sales", "projection", "economics"],
        "competition": ["competition", "competitor", "competitive", "landscape"],
        "funding": ["funding", "investment", "ask", "capital", "valuation", "money"],
        "business": ["business", "model", "monetization", "revenue model"],
        "strategy": ["strategy", "roadmap", "future", "plan", "vision", "goal"]
    }
    
    # Find best matching section
    best_match = ""
    best_score = 0
    
    for category, keywords in section_keywords.items():
        for keyword in keywords:
            if keyword in query_lower:
                # Look for sections containing this keyword
                for section_name, section_content in sections.items():
                    if keyword in section_name.lower() or keyword in section_content.lower():
                        score = section_content.lower().count(keyword)
                        if score > best_score:
                            best_match = section_content
                            best_score = score
    
    if best_match:
        return best_match
    
    # Fallback: fuzzy matching on section names
    section_names = list(sections.keys())
    matches = get_close_matches(query_lower, [name.lower() for name in section_names], n=1, cutoff=0.3)
    
    if matches:
        for name in section_names:
            if name.lower() == matches[0]:
                return sections[name]
    
    # Last resort: return all sections combined (truncated)
    all_content = "\n\n".join([f"**{name}:**\n{content}" for name, content in sections.items()])
    if len(all_content) > 3000:
        all_content = all_content[:3000] + "\n\n[Content truncated...]"
    
    return all_content if all_content else "No relevant information found in the pitch deck."

# Enhanced chat function with better error handling
def chat_with_ai(user_input: str) -> str:
    """Handle AI chat interactions with comprehensive error handling"""
    try:
        doc_data = get_current_doc()
        if not doc_data or doc_data.processing_status != "completed":
            return "Please upload and process a pitch deck first to start the conversation."
        
        # Handle web search requests
        if any(term in user_input.lower() for term in ["search", "web", "google", "internet", "background", "linkedin"]):
            if not serpapi_key:
                return "Web search is not available. SERPAPI_KEY not configured."
            
            search_query = re.sub(r'\b(search|web|google|internet)\b', '', user_input, flags=re.IGNORECASE).strip()
            if doc_data.crm_data.get('company_name') != 'Not found':
                search_query = f"{doc_data.crm_data['company_name']} {search_query}"
            
            search_result = search_serpapi(search_query)
            return f"🔍 **Web Search Results:**\n\n{search_result}"
        
        # Get relevant content
        relevant_content = match_section(user_input, doc_data.sections, doc_data.crm_data)
        
        # Check if it's a specific CRM query
        crm_field = is_specific_crm_query(user_input)
        if crm_field:
            return generate_crm_response(crm_field, doc_data.crm_data)
        
        # Generate AI response
        llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=openai_api_key, 
            temperature=0.1,
            timeout=30
        )
        
        # CONTINUING FROM WHERE THE CODE LEFT OFF...
# Complete the chat_with_ai function context preparation

        # Prepare context
        context = f"""
        You are analyzing a pitch deck for: {doc_data.crm_data.get('company_name', 'Unknown Company')}
        
        Company Details:
        - Ask: {doc_data.crm_data.get('ask', 'Not specified')}
        - Revenue: {doc_data.crm_data.get('revenue', 'Not specified')}
        - Valuation: {doc_data.crm_data.get('valuation', 'Not specified')}
        - Sector: {doc_data.crm_data.get('sector', 'Not specified')}
        - Stage: {doc_data.crm_data.get('stage', 'Not specified')}
        - Team: {doc_data.crm_data.get('assign', 'Not specified')}
        - Description: {doc_data.crm_data.get('description', 'Not specified')}
        
        Based on the pitch deck content, provide accurate, helpful responses. Be concise but informative.
        If the user asks about specific metrics or information not clearly stated, mention what's available vs. what's missing.
        """
        
        # Prepare messages for the AI
        messages = [
            SystemMessage(content=context),
            HumanMessage(content=f"User question: {user_input}\n\nRelevant content from pitch deck:\n{relevant_content}")
        ]
        
        response = llm.invoke(messages)
        return response.content
        
    except RateLimitError:
        return "⏰ API rate limit reached. Please wait a moment and try again."
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        return f"❌ AI service error: {str(e)}"
    except Exception as e:
        logger.error(f"Chat function error: {e}")
        return f"❌ Error processing your question: {str(e)}"

# This contains the remaining portions typically needed for a complete Streamlit pitch deck analyzer

# Web search functionality (SerpAPI integration)
def search_serpapi(query: str, num_results: int = 5) -> str:
    """Search the web using SerpAPI with error handling"""
    if not serpapi_key:
        return "❌ SerpAPI key not configured. Cannot perform web search."
    
    try:
        search_url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": serpapi_key,
            "engine": "google",
            "num": num_results,
            "gl": "in",  # India
            "hl": "en"
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if "organic_results" not in data:
            return "❌ No search results found."
        
        results = []
        for i, result in enumerate(data["organic_results"][:num_results], 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No description")
            link = result.get("link", "#")
            
            results.append(f"**{i}. {title}**\n{snippet}\n🔗 {link}\n")
        
        return "\n".join(results)
        
    except requests.exceptions.Timeout:
        return "❌ Search request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"SerpAPI request failed: {e}")
        return f"❌ Search failed: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in search: {e}")
        return f"❌ Search error: {str(e)}"

# Comprehensive analysis generation
def generate_comprehensive_analysis(doc_data: DocumentData) -> str:
    """Generate comprehensive analysis of the pitch deck"""
    if not doc_data or not doc_data.sections:
        return "No document data available for analysis."
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=openai_api_key,
            temperature=0.1,
            timeout=60
        )
        
        # Prepare sections text for analysis
        sections_text = "\n\n".join([f"**{name}:**\n{content}" for name, content in doc_data.sections.items()])
        
        if len(sections_text) > 12000:
            sections_text = sections_text[:12000] + "\n[Content truncated for analysis...]"
        
        analysis_prompt = f"""
        Analyze this pitch deck comprehensively. Provide detailed insights on:

        1. **Executive Summary** (2-3 sentences)
        2. **Strengths** (3-4 key strong points)
        3. **Areas for Improvement** (3-4 specific suggestions)
        4. **Market Analysis** (market size, opportunity, competition)
        5. **Business Model Assessment** (revenue streams, scalability)
        6. **Team Evaluation** (founder background, experience)
        7. **Financial Health** (current metrics, projections)
        8. **Investment Readiness** (overall score 1-10 with reasoning)
        9. **Recommendations** (next steps for the startup)

        Make it actionable and investor-focused.

        Pitch Deck Content:
        {sections_text}
        
        CRM Data:
        Company: {doc_data.crm_data.get('company_name', 'N/A')}
        Ask: {doc_data.crm_data.get('ask', 'N/A')}
        Revenue: {doc_data.crm_data.get('revenue', 'N/A')}
        Valuation: {doc_data.crm_data.get('valuation', 'N/A')}
        Sector: {doc_data.crm_data.get('sector', 'N/A')}
        """
        
        messages = [
            SystemMessage(content="You are a senior VC analyst providing comprehensive pitch deck analysis."),
            HumanMessage(content=analysis_prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        return f"❌ Analysis failed: {str(e)}"

# Streamlit UI Components
def render_sidebar():
    """Render sidebar with document management and controls"""
    with st.sidebar:
        st.header("📁 Document Manager")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Pitch Deck (PDF)",
            type=['pdf'],
            help="Upload a pitch deck PDF for analysis"
        )
        
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            doc_id = process_document(file_bytes, uploaded_file.name)
            
            if doc_id:
                st.success(f"✅ Successfully processed: {uploaded_file.name}")
            else:
                st.error("❌ Failed to process document")
        
        # Document selector if multiple documents
        if st.session_state.parsed_documents:
            st.subheader("📄 Processed Documents")
            
            doc_options = {}
            for doc_id, doc_data in st.session_state.parsed_documents.items():
                status_emoji = "✅" if doc_data.processing_status == "completed" else "❌"
                doc_options[f"{status_emoji} {doc_data.file_name}"] = doc_id
            
            if len(doc_options) > 1:
                selected_doc = st.selectbox(
                    "Select Document",
                    options=list(doc_options.keys()),
                    index=0
                )
                st.session_state.current_doc_id = doc_options[selected_doc]
        
        # Quick actions
        current_doc = get_current_doc()
        if current_doc and current_doc.processing_status == "completed":
            st.subheader("🎯 Quick Actions")
            
            if st.button("📊 Generate Analysis", use_container_width=True):
                st.session_state.show_comprehensive_analysis = True
            
            if st.button("🏢 CRM Summary", use_container_width=True):
                st.session_state.show_crm_summary = True
            
            if st.button("🔍 Web Search", use_container_width=True):
                if current_doc.crm_data.get('company_name') != 'Not found':
                    company_name = current_doc.crm_data['company_name']
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': f'search web for {company_name} background'
                    })
                else:
                    st.warning("Company name not found for web search")
            
            # Document stats
            st.subheader("📈 Document Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Characters", f"{current_doc.char_count:,}")
            with col2:
                st.metric("Sections", len(current_doc.sections))

def render_main_content():
    """Render main content area"""
    current_doc = get_current_doc()
    
    if not current_doc or current_doc.processing_status != "completed":
        st.title("🚀 Pitch Deck AI Analyzer")
        st.markdown("""
        ### Welcome to the AI-powered Pitch Deck Analyzer!
        
        **Features:**
        - 📄 **PDF Text Extraction** - Advanced PDF parsing with multiple fallback methods
        - 🧠 **AI Analysis** - GPT-4 powered comprehensive analysis
        - 💼 **CRM Integration** - Automatic extraction of key business metrics
        - 🔍 **Web Search** - Real-time company background research
        - 💬 **Interactive Chat** - Ask specific questions about the pitch deck
        
        **Get Started:**
        1. Upload your pitch deck PDF using the sidebar
        2. Wait for processing to complete
        3. Explore sections, chat, or generate comprehensive analysis
        
        **Supported Formats:** PDF files only
        """)
        return
    
    # Main title with company name
    company_name = current_doc.crm_data.get('company_name', 'Unknown Company')
    st.title(f"📊 {company_name}" if company_name != 'Not found' else "📊 Pitch Deck Analysis")
    
    # Status indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info(f"📄 **Document:** {current_doc.file_name}")
    with col2:
        st.success(f"✅ **Status:** Processed")
    with col3:
        st.info(f"📅 **Date:** {current_doc.created_at[:10]}")

def render_comprehensive_analysis():
    """Render comprehensive analysis section"""
    current_doc = get_current_doc()
    if not current_doc:
        return
    
    st.subheader("🔍 Comprehensive Analysis")
    
    # Generate analysis if not cached
    if not current_doc.comprehensive_analysis:
        with st.spinner("🧠 Generating comprehensive analysis..."):
            current_doc.comprehensive_analysis = generate_comprehensive_analysis(current_doc)
    
    if current_doc.comprehensive_analysis:
        st.markdown(current_doc.comprehensive_analysis)
        
        # Download button
        st.download_button(
            label="📥 Download Analysis",
            data=current_doc.comprehensive_analysis,
            file_name=f"{current_doc.file_name}_analysis.txt",
            mime="text/plain"
        )
    else:
        st.error("❌ Failed to generate comprehensive analysis")

def render_crm_summary():
    """Render CRM data summary"""
    current_doc = get_current_doc()
    if not current_doc or not current_doc.crm_data:
        return
    
    st.subheader("🏢 CRM Summary")
    
    crm = current_doc.crm_data
    
    # Key metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💰 Funding Ask", crm.get('ask', 'Not found'))
        st.metric("🏭 Sector", crm.get('sector', 'Not found'))
    
    with col2:
        st.metric("💵 Revenue", crm.get('revenue', 'Not found'))
        st.metric("📊 Stage", crm.get('stage', 'Not found'))
    
    with col3:
        st.metric("🏷️ Valuation", crm.get('valuation', 'Not found'))
        st.metric("💼 Prior Funding", crm.get('prior_funding', 'Not found'))
    
    # Additional details
    st.markdown("### 📋 Details")
    
    details = {
        "👥 Team/Founders": crm.get('assign', 'Not found'),
        "📝 Description": crm.get('description', 'Not found'),
        "📧 Source": crm.get('source', 'Not found'),
        "📅 Received": crm.get('received_date', 'Not found')
    }
    
    for label, value in details.items():
        if value != 'Not found':
            st.write(f"**{label}:** {value}")
    
    # Export CRM data
    if st.button("📤 Export CRM Data"):
        crm_json = json.dumps(crm, indent=2)
        st.download_button(
            label="📥 Download CRM Data (JSON)",
            data=crm_json,
            file_name=f"{current_doc.file_name}_crm.json",
            mime="application/json"
        )

def render_sections_view():
    """Render sections view"""
    current_doc = get_current_doc()
    if not current_doc or not current_doc.sections:
        return
    
    st.subheader("📑 Document Sections")
    
    # Section selector
    section_names = list(current_doc.sections.keys())
    selected_section = st.selectbox(
        "Select Section to View",
        options=section_names,
        index=0
    )
    
    if selected_section:
        st.markdown(f"### {selected_section}")
        content = current_doc.sections[selected_section]
        
        # Show content in expandable container
        with st.expander(f"View {selected_section} Content", expanded=True):
            st.markdown(content)
        
        # Word count
        word_count = len(content.split())
        st.caption(f"📊 **Word Count:** {word_count} | **Character Count:** {len(content)}")

def render_chat_interface():
    """Render chat interface"""
    current_doc = get_current_doc()
    if not current_doc:
        return
    
    st.subheader("💬 Chat with AI")
    
    # Display chat history
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history[-10:]):  # Show last 10 messages
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                st.chat_message("assistant").write(message['content'])
    
    # Chat input
    user_input = st.chat_input("Ask anything about the pitch deck...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Display user message
        st.chat_message("user").write(user_input)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                ai_response = chat_with_ai(user_input)
                st.write(ai_response)
        
        # Add AI response to history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': ai_response
        })
        
        # Rerun to update display
        st.rerun()
    
    # Quick question buttons
    st.subheader("🎯 Quick Questions")
    quick_questions = [
        "What is the funding ask?",
        "Who are the founders?",
        "What problem are they solving?",
        "What's their business model?",
        "Show me the market size",
        "What's their traction?",
        "Search web for company background"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        col = cols[i % 2]
        if col.button(question, key=f"quick_{i}"):
            st.session_state.chat_history.append({
                'role': 'user',
                'content': question
            })
            st.rerun()

# Main application function
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Pitch Deck AI Analyzer",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    current_doc = get_current_doc()
    
    if current_doc and current_doc.processing_status == "completed":
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "📑 Sections", "💬 Chat", "🔍 Analysis", "🏢 CRM Data"
        ])
        
        with tab1:
            render_main_content()
        
        with tab2:
            render_sections_view()
        
        with tab3:
            render_chat_interface()
        
        with tab4:
            render_comprehensive_analysis()
        
        with tab5:
            render_crm_summary()
    
    else:
        render_main_content()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        🚀 Pitch Deck AI Analyzer | Powered by OpenAI GPT-4 & SerpAPI
        </div>
        """,
        unsafe_allow_html=True
    )

# Run the application
if __name__ == "__main__":
    main()

# Additional utility functions for error handling and logging

def handle_streamlit_error(func):
    """Decorator for handling Streamlit errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Streamlit error in {func.__name__}: {e}")
            st.error(f"❌ An error occurred: {str(e)}")
            return None
    return wrapper

# Configuration for deployment
def setup_logging():
    """Setup logging for production deployment"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('pitch_deck_analyzer.log'),
            logging.StreamHandler()
        ]
    )

# Health check endpoint for deployment
def health_check():
    """Health check for deployment monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "openai": bool(openai_api_key),
            "serpapi": bool(serpapi_key)
        }
    }
