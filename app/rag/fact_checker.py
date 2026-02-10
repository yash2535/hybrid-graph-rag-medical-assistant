"""
Simple fact checker for Graph-RAG claims.

- Verifies claims using the provided `context` (patient profile + papers).
- Uses deterministic checks (string matching against patient medications/conditions
  and paper title/text previews). No network calls, no mutation.
- Returns augmented claims: { type, statement, verified: bool, sources: [ ... ] }
"""

from typing import List, Dict
import re
from app.rag.claim_extractor import extract_claims


def _normalize(text: str) -> str:
    return (text or "").lower()


def _match_any(token_list: List[str], text: str) -> bool:
    t = _normalize(text)
    for tok in token_list:
        if tok and tok.lower() in t:
            return True
    return False


def _patient_med_names(context: Dict) -> List[str]:
    meds = context.get("patient", {}).get("medications", []) or []
    return [m.get("name", "").lower() for m in meds if m.get("name")]


def _patient_conditions(context: Dict) -> List[str]:
    conds = context.get("patient", {}).get("conditions", []) or []
    return [c.get("name", "").lower() for c in conds if c.get("name")]


def _papers_evidence(context: Dict, claim_text: str) -> List[Dict]:
    hits = []
    for p in context.get("papers", []) or []:
        title = p.get("title", "") or ""
        preview = (p.get("text_preview") or "")[:1000]
        combined = f"{title}\n{preview}".lower()
        if claim_text.lower() in combined or any(word in combined for word in re.findall(r"\w+", claim_text.lower())[:5]):
            hits.append(
                {
                    "type": "paper",
                    "pmid": p.get("pmid"),
                    "title": p.get("title"),
                    "snippet": (preview[:300] + "...") if preview else "",
                }
            )
    return hits


def verify_claim(claim: Dict[str, str], context: Dict) -> Dict:
    """
    Verify a single claim dictionary: {"type":..., "statement":...}
    Returns claim augmented with "verified": bool and "sources": list.
    """
    statement = claim.get("statement", "")
    sources = []
    verified = False

    # Check KG: patient meds
    meds = _patient_med_names(context)
    if _match_any(meds, statement):
        sources.append({"type": "kg", "detail": "patient_medication_match", "medications": meds})
        verified = True

    # Check KG: patient conditions
    conds = _patient_conditions(context)
    if _match_any(conds, statement):
        sources.append({"type": "kg", "detail": "patient_condition_match", "conditions": conds})
        verified = True

    # Check papers (Qdrant results included in context)
    paper_hits = _papers_evidence(context, statement)
    if paper_hits:
        sources.extend(paper_hits)
        verified = True

    return {
        "type": claim.get("type", "general"),
        "statement": statement,
        "verified": verified,
        "sources": sources,
    }


def verify_claims(claims: List[Dict[str, str]], context: Dict) -> List[Dict]:
    """
    Verify a list of extracted claims.
    If input is plain text, it will run extractor first.
    """
    if not claims:
        return []

    # If claims look like raw text rather than structured, run extractor
    if isinstance(claims, str):
        claims = extract_claims(claims)

    verified = []
    for c in claims:
        try:
            v = verify_claim(c, context)
            verified.append(v)
        except Exception:
            verified.append(
                {
                    "type": c.get("type", "general"),
                    "statement": c.get("statement", ""),
                    "verified": False,
                    "sources": [],
                }
            )
    return verified