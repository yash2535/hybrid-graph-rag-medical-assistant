from unittest.mock import patch


# -------------------------------
# GET /api/users
# -------------------------------

@patch("flask_entrypoint.get_all_patients")
def test_list_users_success(mock_get_all, client):
    mock_get_all.return_value = [
        {"patient_id": "user_1", "name": "Alice"},
        {"patient_id": "user_2", "name": "Bob"},
    ]

    resp = client.get("/api/users")

    assert resp.status_code == 200
    data = resp.get_json()

    assert data["success"] is True
    assert len(data["data"]) == 2
    assert data["data"][0]["patient_id"] == "user_1"


@patch("flask_entrypoint.get_all_patients")
def test_list_users_failure(mock_get_all, client):
    mock_get_all.side_effect = Exception("Neo4j down")

    resp = client.get("/api/users")

    assert resp.status_code == 500
    assert resp.get_json()["success"] is False


# -------------------------------
# POST /api/users
# -------------------------------

@patch("flask_entrypoint.create_patient")
def test_create_user_success(mock_create, client):
    mock_create.return_value = True

    resp = client.post(
        "/api/users",
        json={
            "user_id": "user_3",
            "name": "Charlie",
            "age": 29,
            "gender": "male",
            "blood_type": "O+",
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()

    assert data["success"] is True
    assert data["data"]["id"] == "user_3"


def test_create_user_missing_user_id(client):
    resp = client.post(
        "/api/users",
        json={"name": "NoID", "age": 40},
    )

    assert resp.status_code == 400
    data = resp.get_json()

    assert data["success"] is False
    assert "user_id is required" in data["error"]


# -------------------------------
# GET /api/patient/<id>
# -------------------------------

@patch("flask_entrypoint.get_patient_profile")
def test_get_patient_success(mock_get_profile, client):
    mock_get_profile.return_value = {
        "patient_id": "user_1",
        "conditions": [],
        "medications": [],
    }

    resp = client.get("/api/patient/user_1")

    assert resp.status_code == 200
    data = resp.get_json()

    assert data["success"] is True
    assert data["data"]["patient_id"] == "user_1"


@patch("flask_entrypoint.get_patient_profile")
def test_get_patient_failure(mock_get_profile, client):
    mock_get_profile.side_effect = Exception("Not found")

    resp = client.get("/api/patient/unknown")

    assert resp.status_code == 500
    assert resp.get_json()["success"] is False
