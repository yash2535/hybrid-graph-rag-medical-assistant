from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def simple_chunk(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[str]:
    """
    Split text into overlapping chunks suitable for embeddings and NER.

    Safe for repeated calls and large documents.
    """
    if not text or not isinstance(text, str):
        logger.warning("Chunking skipped: invalid or empty text")
        return []

    size = chunk_size or settings.CHUNK_SIZE
    ovlp = overlap or settings.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=ovlp,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_text(text)

    logger.debug(
        "Text chunked",
        extra={
            "chunks": len(chunks),
            "chunk_size": size,
            "overlap": ovlp,
        },
    )

    return chunks