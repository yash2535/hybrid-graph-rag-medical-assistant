from unittest.mock import patch


@patch("flask_entrypoint.extract_claims")
@patch("flask_entrypoint.call_ollama")
@patch("flask_entrypoint.build_medical_prompt")
@patch("flask_entrypoint.check_drug_interactions")
@patch("flask_entrypoint.search_papers")
@patch("flask_entrypoint.get_wearable_summary")
@patch("flask_entrypoint.get_patient_profile")
@patch("flask_entrypoint.upsert_user_from_question")
def test_rag_pipeline_success(
    mock_upsert,
    mock_profile,
    mock_wearables,
    mock_search,
    mock_drug_interactions,
    mock_prompt,
    mock_llm,
    mock_claims,
    client,
):
    # Arrange mocks
    mock_profile.return_value = {
        "patient_id": "user_1",
        "medications": [{"name": "Metformin"}],
    }

    mock_wearables.return_value = {"available": False}
    mock_search.return_value = [{"pmid": "123", "title": "Test Paper"}]
    mock_drug_interactions.return_value = {"safe": True, "warnings": []}
    mock_prompt.return_value = "FINAL PROMPT"
    mock_llm.return_value = "LLM ANSWER"
    mock_claims.return_value = ["Claim 1", "Claim 2"]

    # Act
    resp = client.post(
        "/api/ask",
        json={
            "user_id": "user_1",
            "question": "Why am I tired?",
        },
    )

    # Assert
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["success"] is True
    assert data["answer"] == "LLM ANSWER"
    assert len(data["claims"]) == 2
    assert data["context"]["papers_found"] == 1

    # Ensure pipeline steps were executed
    mock_upsert.assert_called_once()
    mock_profile.assert_called_once()
    mock_search.assert_called_once()
    mock_llm.assert_called_once()