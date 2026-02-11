from unittest.mock import patch, MagicMock
import numpy as np
from app.processing.embedding import embed_texts


@patch("app.processing.embedding._get_model")
def test_embed_texts_calls_model(mock_get_model):
    mock_model = MagicMock()

    # SentenceTransformer returns a NumPy array
    mock_embeddings = np.array([[0.1, 0.2, 0.3]])
    mock_model.encode.return_value = mock_embeddings

    mock_get_model.return_value = mock_model

    embeddings = embed_texts(["test text"])

    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)   # because .tolist() is called
    mock_model.encode.assert_called_once()