"""
Claim Extractor - PRODUCTION-READY VERSION
Handles multiple formats and has reliable fallbacks.
"""

from typing import List, Dict
import re


def extract_claims(text: str) -> List[Dict[str, str]]:
    """
    Extract structured claims from LLM output.
    Handles multiple response formats reliably.
    """

    if not text:
        return []

    claims: List[Dict[str, str]] = []

    # Strategy 1: Try to extract dash-based claims (- RISK:, - MONITORING:, etc.)
    claims = _extract_dash_based_claims(text)
    
    if claims:
        return claims
    
    # Strategy 2: Try to extract markdown sections (##)
    sections = _split_by_headers(text)
    if sections:
        for section, content in sections.items():
            claim_type = _map_section_to_type(section)
            lines = _extract_bullet_points(content)
            for line in lines:
                if line.strip():
                    claims.append({
                        "type": claim_type,
                        "statement": line.strip()
                    })
    
    if claims:
        return claims
    
    # Strategy 3: Extract by sentence (fallback)
    claims = _extract_smart_sentences(text)
    
    if claims:
        return claims
    
    # Final fallback
    return [{
        "type": "general",
        "statement": text[:200]
    }]


def _extract_dash_based_claims(text: str) -> List[Dict[str, str]]:
    """
    Extract claims in format: - RISK: statement
    This is the most reliable format.
    """
    claims = []
    
    # Pattern: - WORD: text
    pattern = r'^-\s+([A-Z]+):\s+(.+)$'
    
    for line in text.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            claim_type = match.group(1).lower()  # RISK -> risk
            statement = match.group(2).strip()
            
            # Map common types
            if claim_type not in ['risk', 'monitoring', 'warning', 'recommendation']:
                claim_type = 'general'
            
            claims.append({
                "type": claim_type,
                "statement": statement
            })
    
    return claims


def _split_by_headers(text: str) -> Dict[str, str]:
    """
    Split text into sections by markdown headers (# or ##).
    """
    sections = {}
    current_section = "unknown"
    buffer = []

    for line in text.splitlines():
        # Match headers: # or ##
        header_match = re.match(r'^#+\s+(.+)$', line.strip())
        if header_match:
            # Save previous section
            if buffer:
                section_text = "\n".join(buffer).strip()
                if section_text:
                    sections[current_section] = section_text
                buffer = []
            
            # Start new section
            current_section = header_match.group(1).lower()
        else:
            buffer.append(line)

    # Save final section
    if buffer:
        section_text = "\n".join(buffer).strip()
        if section_text:
            sections[current_section] = section_text

    return sections


def _extract_bullet_points(text: str) -> List[str]:
    """
    Extract bullet points from text (-, *, •, or numbered lists).
    """
    claims = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Bullet point (-, *, •)
        if line[0] in ['-', '*', '•']:
            claim = line.lstrip('-*•').strip()
            if claim:
                claims.append(claim)
        
        # Numbered list (1., 2., etc.)
        elif re.match(r'^\d+\.\s+', line):
            claim = re.sub(r'^\d+\.\s+', '', line).strip()
            if claim:
                claims.append(claim)

    return claims


def _extract_smart_sentences(text: str) -> List[Dict[str, str]]:
    """
    Smart sentence extraction with type classification.
    """
    claims = []
    
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        
        # Only consider substantial sentences
        if len(sentence) > 15 and len(sentence) < 500:
            claim_type = _classify_sentence(sentence)
            claims.append({
                "type": claim_type,
                "statement": sentence
            })
    
    return claims[:15]  # Limit to 15 claims


def _classify_sentence(sentence: str) -> str:
    """
    Classify a sentence into a claim type based on keywords.
    """
    s = sentence.lower()
    
    # Risk keywords
    risk_keywords = ['risk', 'danger', 'avoid', 'contraindicated', 'caution', 'can cause', 'may cause', 'leads to']
    if any(word in s for word in risk_keywords):
        return "risk"
    
    # Monitoring keywords
    monitoring_keywords = ['monitor', 'track', 'watch', 'check', 'measure', 'test', 'observe']
    if any(word in s for word in monitoring_keywords):
        return "monitoring"
    
    # Warning/Emergency keywords
    warning_keywords = ['urgent', 'immediately', 'emergency', 'seek', 'call', 'contact doctor', 'call doctor', 'hospital']
    if any(word in s for word in warning_keywords):
        return "warning"
    
    # Recommendation keywords
    recommendation_keywords = ['recommend', 'suggest', 'consider', 'should', 'may help', 'important', 'stay', 'maintain']
    if any(word in s for word in recommendation_keywords):
        return "recommendation"
    
    return "general"


def _map_section_to_type(section: str) -> str:
    """
    Map section name to claim type.
    """
    s = section.lower()
    
    if any(word in s for word in ['risk', 'concern', 'danger']):
        return "risk"
    if any(word in s for word in ['monitor', 'watch', 'track']):
        return "monitoring"
    if any(word in s for word in ['help', 'urgent', 'emergency', 'seek']):
        return "warning"
    if any(word in s for word in ['recommend', 'consider', 'suggest']):
        return "recommendation"
    
    return "general"