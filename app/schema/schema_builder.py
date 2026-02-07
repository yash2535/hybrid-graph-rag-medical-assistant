from datetime import datetime
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"


def _empty_entity_block() -> Dict[str, List[str]]:
    """Standard empty entity structure."""
    return {
        "drugs": [],
        "conditions": [],
        "biomarkers": [],
        "symptoms": [],
    }


def build_payload(
    *,
    text: str,
    pmid: str,
    title: str,
    journal: str,
    year: int,
    authors: List[str],
    section: str,
    chunk_index: int,
    api_query: str,
    entities: Optional[Dict[str, List[str]]] = None,
    kg_node_ids: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Build a standardized payload for Qdrant storage.

    This schema is versioned and safe for long-term storage.
    """

    payload = {
        # ---- schema metadata ----
        "schema_version": SCHEMA_VERSION,
        "source": "pubmed_api",
        "retrieved_at": datetime.utcnow().isoformat() + "Z",

        # ---- document metadata ----
        "pmid": str(pmid),
        "title": title,
        "journal": journal,
        "year": int(year),
        "authors": authors,
        "section": section,
        "chunk_index": int(chunk_index),
        "api_query": api_query,

        # ---- content ----
        "text": text,

        # ---- NLP / KG ----
        "entities": entities if entities is not None else _empty_entity_block(),
        "relations": [],
        "kg_node_ids": kg_node_ids if kg_node_ids is not None else _empty_entity_block(),

        # ---- future ML fields ----
        "study_type": None,
        "confidence_level": None,
    }

    return payload
