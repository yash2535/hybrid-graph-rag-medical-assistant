import requests
from typing import Dict, List, Any

from app.processing.embedding import embed_texts
from app.vector_store.qdrant_store import COLLECTION


def qdrant_hybrid_search(
    question: str,
    user_context: Dict[str, Any],
    expanded_entities: Dict[str, List[str]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    terms = []
    terms.extend(user_context.get("conditions", []))
    terms.extend(user_context.get("drugs", []))

    for k in expanded_entities:
        terms.extend(expanded_entities[k])

    query_text = question + "\nContext: " + ", ".join(set(terms))
    query_vector = embed_texts([query_text])[0]

    resp = requests.post(
        f"http://localhost:6333/collections/{COLLECTION}/points/search",
        json={"vector": query_vector, "limit": top_k, "with_payload": True},
        timeout=10,
    )
    resp.raise_for_status()

    results = []
    for hit in resp.json().get("result", []):
        payload = hit["payload"]
        results.append(
            {
                "score": hit["score"],
                "pmid": payload.get("pmid"),
                "title": payload.get("title"),
                "text_preview": payload.get("text", "")[:500],
                "entities": payload.get("entities"),
            }
        )

    return results
