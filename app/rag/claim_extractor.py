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

    # ✅ Always strip markdown bold/italic before processing
    clean_text = _strip_markdown(text)

    # Strategy 1: Dash-based claims (- RISK: ..., - MONITORING: ...)
    claims = _extract_dash_based_claims(clean_text)
    if claims:
        return claims

    # Strategy 2: Inline bold section headers (**Key Considerations:** ...)
    # This is what phi3:mini / ollama models typically output
    claims = _extract_bold_section_claims(clean_text)
    if claims:
        return claims

    # Strategy 3: Markdown ## headers
    sections = _split_by_headers(clean_text)
    if sections:
        claims = []
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

    # Strategy 4: Smart sentence extraction (fallback)
    claims = _extract_smart_sentences(clean_text)
    if claims:
        return claims

    # Final fallback
    return [{
        "type": "general",
        "statement": clean_text[:200]
    }]


# ------------------------------------------------------------------
# ✅ NEW: Strip markdown formatting before processing
# ------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """
    Remove markdown bold (**text**), italic (*text*), and header markers.
    Preserves the actual content text.
    """
    # Remove bold: **text** → text
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Remove italic: *text* → text
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove inline code: `text` → text
    text = re.sub(r'`(.+?)`', r'\1', text)
    return text


# ------------------------------------------------------------------
# ✅ NEW: Handle inline bold section headers
# e.g. "Key Considerations: ..." or "What to Monitor: ..."
# This is what phi3:mini outputs instead of ## headers
# ------------------------------------------------------------------

def _extract_bold_section_claims(text: str) -> List[Dict[str, str]]:
    """
    Extract claims from inline section format:
    "Key Considerations: blah blah. What to Monitor: blah blah."

    After markdown stripping, the bold headers become plain text like:
    "Key Considerations: ..."
    """
    # Known section headers from prompt_builder.py response format
    section_patterns = [
        (r'key\s+considerations?\s*:', 'general'),
        (r'what\s+to\s+monitor\s*:',   'monitoring'),
        (r'when\s+to\s+seek\s+medical\s+help\s*:', 'warning'),
        (r'safety\s+notes?\s*:',        'recommendation'),
    ]

    # Build a splitter that splits on any known section header
    splitter = '|'.join(p for p, _ in section_patterns)
    full_pattern = re.compile(
        r'(' + splitter + r')',
        re.IGNORECASE
    )

    parts = full_pattern.split(text)

    if len(parts) <= 1:
        return []  # No section headers found

    claims = []
    i = 1  # parts[0] is preamble before first header

    while i < len(parts) - 1:
        header = parts[i].strip().lower()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        i += 2

        # Determine type from header
        claim_type = "general"
        for pattern, ctype in section_patterns:
            if re.search(pattern, header, re.IGNORECASE):
                claim_type = ctype
                break

        # Split content into individual sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        for sentence in sentences:
            sentence = sentence.strip().rstrip('.')
            # Skip very short or empty sentences
            if len(sentence) > 20:
                claims.append({
                    "type": claim_type,
                    "statement": sentence
                })

    return claims


def _extract_dash_based_claims(text: str) -> List[Dict[str, str]]:
    """
    Extract claims in format: - RISK: statement
    """
    claims = []
    pattern = r'^-\s+([A-Z]+):\s+(.+)$'

    for line in text.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            claim_type = match.group(1).lower()
            statement = match.group(2).strip()
            if claim_type not in ['risk', 'monitoring', 'warning', 'recommendation']:
                claim_type = 'general'
            claims.append({"type": claim_type, "statement": statement})

    return claims


def _split_by_headers(text: str) -> Dict[str, str]:
    """
    Split text into sections by markdown headers (# or ##).
    """
    sections = {}
    current_section = "unknown"
    buffer = []

    for line in text.splitlines():
        header_match = re.match(r'^#+\s+(.+)$', line.strip())
        if header_match:
            if buffer:
                section_text = "\n".join(buffer).strip()
                if section_text:
                    sections[current_section] = section_text
                buffer = []
            current_section = header_match.group(1).lower()
        else:
            buffer.append(line)

    if buffer:
        section_text = "\n".join(buffer).strip()
        if section_text:
            sections[current_section] = section_text

    return sections


def _extract_bullet_points(text: str) -> List[str]:
    """
    Extract bullet points from text.
    """
    claims = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0] in ['-', '*', '•']:
            claim = line.lstrip('-*•').strip()
            if claim:
                claims.append(claim)
        elif re.match(r'^\d+\.\s+', line):
            claim = re.sub(r'^\d+\.\s+', '', line).strip()
            if claim:
                claims.append(claim)
    return claims


def _extract_smart_sentences(text: str) -> List[Dict[str, str]]:
    """
    Smart sentence extraction with type classification (fallback).
    """
    claims = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if 15 < len(sentence) < 500:
            claims.append({
                "type": _classify_sentence(sentence),
                "statement": sentence
            })

    return claims[:15]


def _classify_sentence(sentence: str) -> str:
    """
    Classify a sentence into a claim type based on keywords.
    """
    s = sentence.lower()

    if any(w in s for w in ['risk', 'danger', 'avoid', 'contraindicated', 'caution', 'can cause', 'may cause']):
        return "risk"
    if any(w in s for w in ['monitor', 'track', 'watch', 'check', 'measure', 'test', 'observe']):
        return "monitoring"
    if any(w in s for w in ['urgent', 'immediately', 'emergency', 'seek', 'call', 'hospital']):
        return "warning"
    if any(w in s for w in ['recommend', 'suggest', 'consider', 'should', 'important', 'maintain']):
        return "recommendation"

    return "general"


def _map_section_to_type(section: str) -> str:
    """
    Map section name to claim type.
    """
    s = section.lower()

    if any(w in s for w in ['risk', 'concern', 'danger']):
        return "risk"
    if any(w in s for w in ['monitor', 'watch', 'track']):
        return "monitoring"
    if any(w in s for w in ['help', 'urgent', 'emergency', 'seek']):
        return "warning"
    if any(w in s for w in ['recommend', 'consider', 'suggest', 'safety', 'note']):
        return "recommendation"

    return "general"