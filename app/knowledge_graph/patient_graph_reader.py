"""
Patient Graph Reader

Purpose:
- Write patient-related facts into Neo4j
- Read patient medical history as clean, structured JSON
- Used by Hybrid Graph-RAG (NO research papers here)

Neo4j Domain:
Patient → Disease → Medication
Patient → Medication
Disease → LabTest
"""

from typing import Dict, Any, List
from neo4j import GraphDatabase
import os


# ------------------------------------------------------------------
# Neo4j connection
# ------------------------------------------------------------------

def _get_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(user, password))


# ------------------------------------------------------------------
# WRITE: Update patient history from question
# ------------------------------------------------------------------

def upsert_user_from_question(user_id: str, question: str) -> None:
    """
    Minimal update:
    - Ensures Patient node exists
    - (Later we may extract diseases from question automatically)
    """

    driver = _get_driver()

    cypher = """
    MERGE (p:Patient {id: $user_id})
      ON CREATE SET p.created_at = datetime()
      ON MATCH  SET p.updated_at = datetime()
    """

    with driver.session() as session:
        session.run(cypher, user_id=user_id)

    driver.close()


# ------------------------------------------------------------------
# READ: Fetch full patient profile
# ------------------------------------------------------------------

def get_patient_profile(user_id: str) -> Dict[str, Any]:
    """
    Fetch complete patient medical profile as JSON.
    This output is LLM-safe and deterministic.
    """

    driver = _get_driver()

    cypher = """
    MATCH (p:Patient {id: $user_id})

    OPTIONAL MATCH (p)-[:HAS_DISEASE]->(d:Disease)
    OPTIONAL MATCH (d)-[:TREATED_BY]->(m:Medication)
    OPTIONAL MATCH (p)-[:PRESCRIBED]->(pm:Medication)
    OPTIONAL MATCH (d)-[:HAS_LAB_RESULT]->(l:LabTest)

    RETURN
      p,
      collect(DISTINCT d)  AS diseases,
      collect(DISTINCT m)  AS disease_medications,
      collect(DISTINCT pm) AS patient_medications,
      collect(DISTINCT l)  AS lab_tests
    """

    with driver.session() as session:
        record = session.run(cypher, user_id=user_id).single()

    driver.close()

    if not record:
        return {"patient_id": user_id}

    # -------------------------------------------------
    # Build clean JSON
    # -------------------------------------------------

    patient_node = record["p"]

    diseases = _format_diseases(
        record["diseases"],
        record["disease_medications"],
        record["lab_tests"],
    )

    medications = _format_medications(record["patient_medications"])

    profile = {
        "patient_id": patient_node.get("id"),
        "demographics": {
            "age": patient_node.get("age"),
            "gender": patient_node.get("gender"),
            "blood_type": patient_node.get("bloodType"),
        },
        "conditions": diseases,
        "medications": medications,
    }

    return profile


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _format_diseases(
    disease_nodes: List[Any],
    medication_nodes: List[Any],
    lab_nodes: List[Any],
) -> List[Dict[str, Any]]:
    """
    Normalize diseases with linked medications + labs
    """

    diseases = []

    for d in disease_nodes:
        if not d:
            continue

        disease = {
            "name": d.get("name"),
            "category": d.get("category"),
            "severity": d.get("severity"),
            "status": d.get("status"),
            "diagnosis_date": _safe_date(d.get("diagnosisDate")),
            "medications": [],
            "lab_results": [],
        }

        # Attach labs
        for l in lab_nodes:
            if not l:
                continue
            disease["lab_results"].append(
                {
                    "name": l.get("name"),
                    "result": l.get("result"),
                    "unit": l.get("unit"),
                    "normal_range": l.get("normalRange"),
                    "status": l.get("status"),
                    "date": _safe_date(l.get("testDate")),
                }
            )

        # Attach medications
        for m in medication_nodes:
            if not m:
                continue
            disease["medications"].append(
                {
                    "name": m.get("name"),
                    "dosage": m.get("dosage"),
                    "frequency": m.get("frequency"),
                    "purpose": m.get("purpose"),
                }
            )

        diseases.append(disease)

    return diseases


def _format_medications(nodes: List[Any]) -> List[Dict[str, Any]]:
    meds = []
    for m in nodes:
        if not m:
            continue
        meds.append(
            {
                "name": m.get("name"),
                "dosage": m.get("dosage"),
                "frequency": m.get("frequency"),
                "purpose": m.get("purpose"),
            }
        )
    return meds


def _safe_date(value):
    return str(value) if value else None
