import pytest
from app.processing.chunker import simple_chunk


def test_chunker_empty_text():
    chunks = simple_chunk("")
    assert chunks == []


def test_chunker_none_input():
    chunks = simple_chunk(None)
    assert chunks == []


def test_chunker_non_string_input():
    chunks = simple_chunk(12345)
    assert chunks == []


def test_chunker_valid_text_returns_chunks():
    text = "This is a sentence. This is another sentence."
    chunks = simple_chunk(text, chunk_size=20, overlap=5)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


def test_chunker_is_deterministic():
    text = "Medical text for testing chunking behavior."
    c1 = simple_chunk(text)
    c2 = simple_chunk(text)

    assert c1 == c2