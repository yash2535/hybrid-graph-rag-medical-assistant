import uuid
from typing import Iterable

from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from app.fetchers.pubmed_fetcher import fetch_all_pmc_articles
from app.processing.chunker import simple_chunk
from app.processing.embedding import embed_texts
from app.processing.entity_extractor import extract_medical_entities
from app.schema.schema_builder import build_payload
from app.vector_store.qdrant_store import (
    get_client,
    create_collection_if_not_exists,
    create_indexes,
    COLLECTION,
)

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _batch(iterable: Iterable, size: int):
    """Yield items in fixed-size batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------
# Main ingestion pipeline (QDRANT ONLY)
# ---------------------------------------------------------------------

def ingest_from_pubmed(query: str, max_results: int = 5) -> None:
    """
    Fetch PubMed Central articles, chunk them, embed them,
    and store ONLY in Qdrant.

    Neo4j is NOT used here by design.
    """
    logger.info("Starting PubMed ingestion", extra={"query": query})

    # ---- Qdrant setup ----
    try:
        client = get_client()
        create_collection_if_not_exists(client)
        create_indexes(client)
    except Exception:
        logger.exception("Failed to initialize Qdrant")
        return

    # ---- Fetch papers ----
    try:
        papers = fetch_all_pmc_articles(query, max_results=max_results)
    except Exception:
        logger.exception("Failed to fetch PMC articles")
        return

    if not papers:
        logger.warning("No PMC articles found", extra={"query": query})
        return

    points_buffer = []

    # ---- Process papers ----
    for paper in papers:
        pmid = paper.get("pmid", "unknown")
        title = paper.get("title", "No Title")
        text = paper.get("abstract")

        if not text:
            logger.warning("Skipping paper: no text", extra={"pmid": pmid})
            continue

        logger.info(
            "Processing paper",
            extra={"pmid": pmid, "title": title[:50]},
        )

        chunks = simple_chunk(text)
        if not chunks:
            logger.warning("No chunks produced", extra={"pmid": pmid})
            continue

        vectors = embed_texts(chunks)
        if len(vectors) != len(chunks):
            logger.error("Chunk/vector mismatch", extra={"pmid": pmid})
            continue

        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            try:
                # ---- Entity extraction (metadata only) ----
                entities = extract_medical_entities(chunk)

                payload = build_payload(
                    text=chunk,
                    pmid=pmid,
                    title=title,
                    journal=paper.get("journal", "Unknown"),
                    year=paper.get("year", 0),
                    authors=paper.get("authors", ["PMC Full Text"]),
                    section="Full Text",
                    chunk_index=i,
                    api_query=query,
                    entities=entities,
                )

                points_buffer.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=payload,
                    )
                )

            except Exception:
                logger.exception(
                    "Failed to process chunk",
                    extra={"pmid": pmid, "chunk_index": i},
                )

        # ---- Batch upload ----
        for batch in _batch(points_buffer, settings.QDRANT_BATCH_SIZE):
            try:
                client.upsert(
                    collection_name=COLLECTION,
                    points=batch,
                )
                logger.info(
                    "Batch uploaded",
                    extra={"batch_size": len(batch)},
                )
            except UnexpectedResponse:
                logger.exception("Qdrant upsert failed")
            except Exception:
                logger.exception("Unexpected error during Qdrant upsert")

        points_buffer.clear()

    logger.info("PubMed ingestion completed", extra={"query": query})


# ---------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    queries = [
    "type 2 diabetes",
    "hypertension",
    "heart disease",
    "asthma",
    "chronic kidney disease",
]

    for q in queries:
        ingest_from_pubmed(query=q, max_results=5)

   