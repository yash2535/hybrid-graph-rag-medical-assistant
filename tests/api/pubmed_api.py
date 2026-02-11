from unittest.mock import patch
import numpy as np

from app.rag.qdrant_search import qdrant_hybrid_search


@patch("app.rag.qdrant_search.requests.post")
@patch("app.rag.qdrant_search.embed_texts")
def test_pubmed_qdrant_search(mock_embed, mock_post):
    mock_embed.return_value = [np.array([0.1, 0.2, 0.3])]

    mock_post.return_value.json.return_value = {
        "result": [
            {
                "score": 0.95,
                "payload": {
                    "pmid": "98765",
                    "title": "Metformin and Fatigue",
                    "text": "Fatigue is a known side effect...",
                    "entities": {"drugs": ["metformin"]},
                },
            }
        ]
    }
    mock_post.return_value.raise_for_status.return_value = None

    results = qdrant_hybrid_search(
        question="metformin fatigue",
        user_context={"conditions": ["diabetes"], "drugs": ["metformin"]},
        expanded_entities={"drugs": ["metformin"]},
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["pmid"] == "98765"
    assert "fatigue" in results[0]["title"].lower()