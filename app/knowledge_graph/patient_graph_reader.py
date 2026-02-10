"""
Patient Graph Reader

Purpose:
- Write patient-related FACTS into Neo4j
- Read patient medical history as clean, structured JSON
- Used by Hybrid Graph-RAG (NO reasoning, NO research papers)

Design Principles:
- Deterministic
- Auditable
- LLM-safe (facts only, no inference)

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
# WRITE: Ensure patient exists (facts only)
# ------------------------------------------------------------------

def upsert_user_from_question(user_id: str, question: str) -> None:
    """
    Minimal deterministic update:
    - Ensure Patient node exists
    - Does NOT extract diseases or symptoms
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
# READ: Fetch patient FACTS profile
# ------------------------------------------------------------------

def get_patient_profile(user_id: str) -> Dict[str, Any]:
    """
    Fetch complete patient medical profile as FACTS ONLY.
    No inference, no reasoning.
    Output is safe to pass into LLM context.
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
        return {
            "patient_id": user_id,
            "name": None,  # ADDED: Support frontend
            "facts_version": "1.0"
        }

    patient_node = record["p"]

    profile = {
        "patient_id": patient_node.get("id"),
        "name": patient_node.get("name"),  # ADDED: Support frontend
        "age": patient_node.get("age"),    # ADDED: Support frontend
        "gender": patient_node.get("gender"),  # ADDED: Support frontend
        "bloodType": patient_node.get("bloodType"),  # ADDED: Support frontend
        "facts_version": "1.0",
        "demographics": {
            "age": patient_node.get("age"),
            "gender": patient_node.get("gender"),
            "blood_type": patient_node.get("bloodType"),
        },
        "conditions": _format_diseases(
            record["diseases"],
            record["disease_medications"],
            record["lab_tests"],
        ),
        "medications": _format_medications(record["patient_medications"]),
    }

    return profile


# ------------------------------------------------------------------
# NEW: Support frontend user management (ADDED FOR UI)
# ------------------------------------------------------------------

def get_all_patients() -> List[Dict[str, Any]]:
    """
    Get list of all patients (for frontend dropdown).
    Returns basic facts only.
    """
    driver = _get_driver()

    cypher = """
    MATCH (p:Patient)
    RETURN p.id AS id, 
           p.name AS name, 
           p.age AS age, 
           p.gender AS gender, 
           p.bloodType AS bloodType
    ORDER BY p.id
    """

    with driver.session() as session:
        result = session.run(cypher)
        patients = []
        for record in result:
            patients.append({
                "id": record["id"],
                "name": record["name"],
                "age": record["age"],
                "gender": record["gender"],
                "bloodType": record["bloodType"]
            })

    driver.close()
    return patients


def create_patient(user_id: str, name: str = None, age: int = None, 
                   gender: str = None, blood_type: str = None) -> bool:
    """
    Create a new patient node (facts only).
    Returns True if created, False if already exists.
    """
    driver = _get_driver()

    # Check if patient exists
    check_cypher = "MATCH (p:Patient {id: $user_id}) RETURN p"
    
    with driver.session() as session:
        existing = session.run(check_cypher, user_id=user_id).single()
        
        if existing:
            driver.close()
            return False
        
        # Create new patient
        create_cypher = """
        CREATE (p:Patient {
            id: $user_id,
            name: $name,
            age: $age,
            gender: $gender,
            bloodType: $blood_type,
            created_at: datetime()
        })
        """
        
        session.run(
            create_cypher,
            user_id=user_id,
            name=name,
            age=age,
            gender=gender,
            blood_type=blood_type
        )

    driver.close()
    return True


# ------------------------------------------------------------------
# Helpers (FACT NORMALIZATION ONLY)
# ------------------------------------------------------------------

def _format_diseases(
    disease_nodes: List[Any],
    medication_nodes: List[Any],
    lab_nodes: List[Any],
) -> List[Dict[str, Any]]:
    """
    Normalize disease-related FACTS.
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

        # Attach lab observations (facts only)
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

        # Attach disease-related medications
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
    """
    Normalize patient-level medication FACTS.
    """

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