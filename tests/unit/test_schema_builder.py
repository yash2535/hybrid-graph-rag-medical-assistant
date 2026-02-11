from app.schema.schema_builder import build_payload, SCHEMA_VERSION


def test_build_payload_basic_fields():
    payload = build_payload(
        text="sample text",
        pmid="12345",
        title="Test Paper",
        journal="Nature",
        year=2024,
        authors=["A", "B"],
        section="abstract",
        chunk_index=0,
        api_query="diabetes fatigue",
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["pmid"] == "12345"
    assert payload["text"] == "sample text"
    assert payload["chunk_index"] == 0
    assert payload["relations"] == []


def test_build_payload_default_entities():
    payload = build_payload(
        text="text",
        pmid="1",
        title="t",
        journal="j",
        year=2024,
        authors=[],
        section="s",
        chunk_index=1,
        api_query="q",
    )

    assert "entities" in payload
    assert payload["entities"]["drugs"] == []
    assert payload["kg_node_ids"]["conditions"] == []


def test_build_payload_custom_entities():
    entities = {"drugs": ["metformin"], "conditions": ["diabetes"], "biomarkers": [], "symptoms": []}

    payload = build_payload(
        text="text",
        pmid="1",
        title="t",
        journal="j",
        year=2024,
        authors=[],
        section="s",
        chunk_index=1,
        api_query="q",
        entities=entities,
    )

    assert payload["entities"]["drugs"] == ["metformin"]