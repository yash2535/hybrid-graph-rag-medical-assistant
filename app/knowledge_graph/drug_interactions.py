"""
Drug Interaction Safety Engine

Purpose:
- Extract VERIFIED drug interaction FACTS
- Extract VERIFIED drug effect / mechanism FACTS
- NO symptom inference
- NO patient-specific reasoning
- Used BEFORE calling the LLM

Design:
- Deterministic
- Auditable
- Knowledge-Graph + Rule based
"""

from typing import List, Dict, Any
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
# Public API (FACT EXTRACTOR)
# ------------------------------------------------------------------

def check_drug_interactions(medications: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Entry point used by Hybrid Graph-RAG pipeline.

    Returns FACTS only:
    - drug-drug interactions
    - drug-condition interactions
    - drug-effect mechanisms
    """

    drug_names = sorted(
        {m.get("name").lower() for m in medications if m.get("name")}
    )

    if not drug_names:
        return _safe_response("No medications provided")

    return {
        "checked_drugs": drug_names,
        "drug_drug_interactions": _check_drug_drug_facts(drug_names),
        "drug_condition_interactions": _check_drug_condition_facts(drug_names),
        "drug_effect_facts": _check_drug_effect_facts(drug_names),
    }


# ------------------------------------------------------------------
# FACT ENGINES
# ------------------------------------------------------------------

def _check_drug_drug_facts(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Drug–drug interaction FACTS (demo-scoped, diabetes-focused).
    """

    RULES = [
        {
            "drugs": {"metformin", "contrast dye"},
            "severity": "high",
            "interaction": "Increased risk of lactic acidosis",
            "mechanism": (
                "Contrast agents may impair renal function, "
                "leading to accumulation of metformin."
            ),
            "evidence": "clinical literature"
        },
        {
            "drugs": {"metformin", "insulin"},
            "severity": "moderate",
            "interaction": "Increased risk of hypoglycemia",
            "mechanism": (
                "Both drugs lower blood glucose levels."
            ),
            "evidence": "clinical guidelines"
        },
        {
            "drugs": {"metformin", "alcohol"},
            "severity": "high",
            "interaction": "Increased risk of lactic acidosis",
            "mechanism": (
                "Alcohol affects hepatic lactate metabolism."
            ),
            "evidence": "drug safety literature"
        }
    ]

    facts = []
    drug_set = set(drugs)

    for rule in RULES:
        if rule["drugs"].issubset(drug_set):
            facts.append({
                "type": "drug-drug-interaction",
                "drugs_involved": sorted(rule["drugs"]),
                "severity": rule["severity"],
                "interaction": rule["interaction"],
                "mechanism": rule["mechanism"],
                "evidence": rule["evidence"],
            })

    return facts


def _check_drug_condition_facts(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Drug–condition contraindication FACTS via Neo4j.
    """

    driver = _get_driver()
    facts = []

    cypher = """
    MATCH (d:Medication)
    WHERE toLower(d.name) IN $drug_names
    MATCH (d)-[:CONTRAINDICATED_IN]->(c:Disease)
    RETURN d.name AS drug, c.name AS condition, c.severity AS severity
    """

    with driver.session() as session:
        results = session.run(cypher, drug_names=drugs)

        for r in results:
            facts.append({
                "type": "drug-condition-interaction",
                "drug": r["drug"],
                "condition": r["condition"],
                "severity": r["severity"] or "moderate",
                "evidence": "knowledge graph"
            })

    driver.close()
    return facts


def _check_drug_effect_facts(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Drug → physiological effect FACTS.
    NO symptom inference here.
    """

    facts = []

    if "metformin" in drugs:
        facts.append({
            "type": "drug-effect",
            "drug": "metformin",
            "effect": "reduced vitamin B12 absorption",
            "mechanism": (
                "Metformin interferes with calcium-dependent "
                "vitamin B12 absorption in the terminal ileum."
            ),
            "clinical_relevance": (
                "Long-term use has been associated with vitamin B12 deficiency."
            ),
            "evidence": "well-established"
        })

    return facts


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_response(reason: str) -> Dict[str, Any]:
    return {
        "checked_drugs": [],
        "drug_drug_interactions": [],
        "drug_condition_interactions": [],
        "drug_effect_facts": [],
        "note": reason,
    }
