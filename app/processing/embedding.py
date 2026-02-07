from typing import List

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazy-load and cache the embedding model.
    """
    global _model

    if _model is None:
        logger.info(
            "Loading embedding model",
            extra={
                "model_name": settings.EMBEDDING_MODEL_NAME,
                "device": settings.EMBEDDING_DEVICE,
            },
        )
        _model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
        )

    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate normalized embeddings for a list of texts.
    Safe for batch processing.
    """
    if not texts or not isinstance(texts, list):
        logger.warning("Embedding skipped: invalid input")
        return []

    clean_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not clean_texts:
        logger.warning("Embedding skipped: empty text list")
        return []

    model = _get_model()

    embeddings = model.encode(
        clean_texts,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    logger.debug(
        "Embeddings generated",
        extra={
            "input_texts": len(clean_texts),
            "embedding_dim": embeddings.shape[1],
        },
    )

    return embeddings.tolist()
