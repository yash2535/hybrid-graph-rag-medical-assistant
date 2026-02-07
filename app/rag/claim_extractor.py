"""
Claim Extractor

Purpose:
- Convert LLM free-text responses into structured medical claims
- Enable auditing, fact-checking, and future verification
- Keep extraction deterministic and explainable

This module does NOT interpret correctness.
It only extracts claims as stated.
"""

from typing import List, Dict
import re


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_claims(text: str) -> List[Dict[str, str]]:
    """
    Extract structured claims from LLM output.

    Returns a list of:
    {
      "type": "risk" | "recommendation" | "warning" | "monitoring",
      "statement": "..."
    }
    """

    if not text:
        return []

    claims: List[Dict[str, str]] = []

    sections = _split_sections(text)

    for section, content in sections.items():
        lines = _extract_bullets(content)

        for line in lines:
            claim_type = _map_section_to_claim_type(section)
            claims.append(
                {
                    "type": claim_type,
                    "statement": line,
                }
            )

    return claims


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _split_sections(text: str) -> Dict[str, str]:
    """
    Split LLM output into logical sections.
    Expected headings:
      ## Key Concerns
      ## What to Monitor
      ## When to Seek Medical Help
      ## Safety Notes
    """

    sections = {}
    current_section = "unknown"
    buffer = []

    for line in text.splitlines():
        header_match = re.match(r"^##\s+(.*)", line.strip())
        if header_match:
            if buffer:
                sections[current_section] = "\n".join(buffer)
                buffer = []
            current_section = header_match.group(1).lower()
        else:
            buffer.append(line)

    if buffer:
        sections[current_section] = "\n".join(buffer)

    return sections


def _extract_bullets(text: str) -> List[str]:
    """
    Extract bullet points or sentence-level claims.
    """

    claims = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Bullet-style
        if line.startswith("-"):
            claims.append(line.lstrip("- ").strip())
        # Fallback: full sentence
        elif len(line.split()) > 5:
            claims.append(line)

    return claims


def _map_section_to_claim_type(section: str) -> str:
    """
    Map section titles to normalized claim types.
    """

    section = section.lower()

    if "concern" in section:
        return "risk"
    if "monitor" in section:
        return "monitoring"
    if "seek" in section:
        return "warning"
    if "safety" in section:
        return "recommendation"

    return "general"
