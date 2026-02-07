"""
Paper Search (Qdrant)

Purpose:
- Semantic search over medical research papers
- Return LLM-ready paper context
"""

from typing import List, Dict, Any

from app.vector_store.qdrant_store import get_client, COLLECTION
from app.processing.embedding import embed_texts
from app.utils.logger import get_logger
from qdrant_client.models import SearchRequest

logger = get_logger(__name__)


def search_papers(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search medical research papers stored in Qdrant.
    """

    client = get_client()
    logger.info("Searching papers", extra={"query": query})

    query_vector = embed_texts([query])[0]

    response = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    papers = []

    for hit in response.points:
        payload = hit.payload or {}

        papers.append(
            {
                "score": hit.score,
                "pmid": payload.get("pmid"),
                "title": payload.get("title"),
                "journal": payload.get("journal"),
                "year": payload.get("year"),
                "section": payload.get("section"),
                "text_preview": (payload.get("text") or "")[:500],
                "entities": payload.get("entities"),
            }
        )

    logger.info("Paper search completed", extra={"results": len(papers)})
    return papers
