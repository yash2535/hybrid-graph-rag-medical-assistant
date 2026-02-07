from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

COLLECTION = settings.QDRANT_COLLECTION


def get_client() -> QdrantClient:
    """
    Create and return a Qdrant client.
    Supports local, Docker, and cloud-based Qdrant.
    """
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT,
        )
        logger.info("Qdrant client initialized")
        return client
    except Exception as e:
        logger.exception("Failed to initialize Qdrant client")
        raise RuntimeError("Qdrant connection failed") from e


def create_collection_if_not_exists(client: QdrantClient) -> None:
    """
    Create the vector collection if it does not exist.
    Safe to call multiple times.
    """
    try:
        existing = {c.name for c in client.get_collections().collections}

        if COLLECTION in existing:
            logger.info("Qdrant collection already exists", extra={"collection": COLLECTION})
            return

        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection", extra={"collection": COLLECTION})

    except Exception as e:
        logger.exception("Failed to create Qdrant collection")
        raise


def _create_payload_index_safe(
    client: QdrantClient,
    field_name: str,
    field_schema: PayloadSchemaType,
) -> None:
    """
    Create payload index safely (idempotent).
    """
    try:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field_name,
            field_schema=field_schema,
        )
        logger.info("Created payload index", extra={"field": field_name})
    except UnexpectedResponse as e:
        # Index probably already exists
        logger.debug("Payload index already exists", extra={"field": field_name})
    except Exception:
        logger.exception("Failed to create payload index", extra={"field": field_name})
        raise


def create_indexes(client: QdrantClient) -> None:
    """
    Create payload indexes for efficient filtering.
    Safe to call multiple times.
    """
    _create_payload_index_safe(client, "pmid", PayloadSchemaType.KEYWORD)
    _create_payload_index_safe(client, "year", PayloadSchemaType.INTEGER)
    _create_payload_index_safe(client, "journal", PayloadSchemaType.KEYWORD)
    _create_payload_index_safe(client, "study_type", PayloadSchemaType.KEYWORD)
    _create_payload_index_safe(client, "entities.drugs", PayloadSchemaType.KEYWORD)

    logger.info("Qdrant payload indexes ensured")
