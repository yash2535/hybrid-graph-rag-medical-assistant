
from unittest.mock import patch


@patch("flask_entrypoint.call_ollama")
@patch("flask_entrypoint.search_papers")
@patch("flask_entrypoint.check_drug_interactions")
@patch("flask_entrypoint.get_wearable_summary")
@patch("flask_entrypoint.get_patient_profile")
@patch("flask_entrypoint.upsert_user_from_question")
def test_ask_api_happy_path(
    mock_upsert,
    mock_profile,
    mock_wearables,
    mock_drug_check,
    mock_search,
    mock_llm,
    client,
):
    mock_profile.return_value = {
        "patient_id": "user_1",
        "conditions": [{"name": "Diabetes"}],
        "medications": [{"name": "Metformin"}],
    }

    mock_wearables.return_value = {"available": False}
    mock_drug_check.return_value = {"safe": True, "warnings": []}
    mock_search.return_value = [{"pmid": "123", "title": "Test Paper"}]
    mock_llm.return_value = "LLM RESPONSE"

    resp = client.post(
        "/api/ask",
        json={
            "user_id": "user_1",
            "question": "Why am I tired?",
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()

    assert data["success"] is True
    assert "answer" in data
    assert "claims" in data

    mock_upsert.assert_called_once()
    mock_llm.assert_called_once()