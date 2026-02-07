import os
from typing import Dict, Any, List
from neo4j import GraphDatabase
from app.utils.logger import get_logger

logger = get_logger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class PatientGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )

    def close(self):
        self.driver.close()

    # --------------------------------------------------
    # CORE FUNCTION: Fetch full patient medical profile
    # --------------------------------------------------
    def get_patient_profile(self, patient_id: str) -> Dict[str, Any]:
        """
        Returns a structured patient medical profile from Neo4j
        suitable for LLM + RAG pipelines.
        """

        query = """
        MATCH (p:Patient {id: $patient_id})

        OPTIONAL MATCH (p)-[:HAS_DISEASE]->(d:Disease)
        OPTIONAL MATCH (d)-[:TREATED_BY]->(m:Medication)
        OPTIONAL MATCH (d)-[:HAS_LAB_RESULT]->(l:LabTest)

        RETURN
            p,
            collect(DISTINCT d) AS diseases,
            collect(DISTINCT m) AS medications,
            collect(DISTINCT l) AS lab_tests
        """

        with self.driver.session() as session:
            record = session.run(query, patient_id=patient_id).single()

        if not record:
            logger.warning("Patient not found", extra={"patient_id": patient_id})
            return {}

        patient = record["p"]

        diseases = [
            {
                "id": d.get("id"),
                "name": d.get("name"),
                "category": d.get("category"),
                "severity": d.get("severity"),
                "status": d.get("status"),
                "diagnosis_date": str(d.get("diagnosisDate")),
            }
            for d in record["diseases"]
            if d
        ]

        medications = [
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "dosage": m.get("dosage"),
                "frequency": m.get("frequency"),
                "purpose": m.get("purpose"),
            }
            for m in record["medications"]
            if m
        ]

        lab_tests = [
            {
                "id": l.get("id"),
                "name": l.get("name"),
                "result": l.get("result"),
                "unit": l.get("unit"),
                "normal_range": l.get("normalRange"),
                "status": l.get("status"),
                "test_date": str(l.get("testDate")),
            }
            for l in record["lab_tests"]
            if l
        ]

        return {
            "patient": {
                "id": patient.get("id"),
                "name": patient.get("name"),
                "age": patient.get("age"),
                "gender": patient.get("gender"),
                "blood_type": patient.get("bloodType"),
            },
            "conditions": diseases,
            "medications": medications,
            "lab_results": lab_tests,
        }


# --------------------------------------------------
# CLI TEST (VERY IMPORTANT)
# --------------------------------------------------
if __name__ == "__main__":
    graph = PatientGraph()
    data = graph.get_patient_profile("P001")
    graph.close()

    import json
    print(json.dumps(data, indent=2))
