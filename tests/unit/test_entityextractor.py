from unittest.mock import patch, MagicMock
from app.processing.entity_extractor import extract_medical_entities


def test_entity_extractor_empty_text():
    result = extract_medical_entities("")
    assert result["drugs"] == []
    assert result["conditions"] == []


@patch("app.processing.entity_extractor._get_model")
def test_entity_extractor_maps_entities(mock_get_model):
    mock_model = MagicMock()
    mock_model.predict_entities.return_value = [
        {"text": "Metformin", "label": "drug"},
        {"text": "Diabetes", "label": "medical condition"},
        {"text": "Fatigue", "label": "symptom"},
    ]
    mock_get_model.return_value = mock_model

    result = extract_medical_entities("dummy text")

    assert "metformin" in result["drugs"]
    assert "diabetes" in result["conditions"]
    assert "fatigue" in result["symptoms"]