"""
Drug Interaction Safety Engine

Purpose:
- Detect risky drug–drug and drug–condition interactions
- Provide explainable, structured safety warnings
- Used BEFORE calling the LLM

This module MUST be deterministic and auditable.
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
# Public API
# ------------------------------------------------------------------

def check_drug_interactions(medications: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Entry point used by Hybrid Graph-RAG pipeline.

    Input:
      medications = [
        {"name": "Metformin", ...},
        {"name": "Lisinopril", ...}
      ]

    Output:
      {
        "safe": true/false,
        "warnings": [...],
        "recommendations": [...]
      }
    """

    drug_names = sorted(
        {m.get("name").lower() for m in medications if m.get("name")}
    )

    if not drug_names:
        return _safe_response("No medications available")

    warnings = []
    recommendations = []

    # Rule-based checks
    warnings.extend(_check_drug_drug_rules(drug_names))
    warnings.extend(_check_drug_condition_rules(drug_names))

    if warnings:
        recommendations.append(
            "Consult a healthcare provider before making any medication changes."
        )

    return {
        "safe": len(warnings) == 0,
        "checked_drugs": drug_names,
        "warnings": warnings,
        "recommendations": recommendations,
    }


# ------------------------------------------------------------------
# Rule Engines
# ------------------------------------------------------------------

def _check_drug_drug_rules(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Drug–drug interaction rules.
    (This can later be moved to Neo4j or external DB.)
    """

    RULES = [
        {
            "drugs": {"metformin", "contrast dye"},
            "severity": "high",
            "message": "Risk of lactic acidosis when metformin is used with contrast agents."
        },
        {
            "drugs": {"lisinopril", "potassium supplements"},
            "severity": "moderate",
            "message": "Increased risk of hyperkalemia with ACE inhibitors."
        },
    ]

    warnings = []

    for rule in RULES:
        if rule["drugs"].issubset(set(drugs)):
            warnings.append(
                {
                    "type": "drug-drug",
                    "severity": rule["severity"],
                    "message": rule["message"],
                    "drugs_involved": sorted(rule["drugs"]),
                }
            )

    return warnings


def _check_drug_condition_rules(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Drug–condition interaction rules via Neo4j.
    """

    driver = _get_driver()
    warnings = []

    cypher = """
    MATCH (d:Medication)
    WHERE toLower(d.name) IN $drug_names
    MATCH (d)-[:CONTRAINDICATED_IN]->(c:Disease)
    RETURN d.name AS drug, c.name AS condition, c.severity AS severity
    """

    with driver.session() as session:
        results = session.run(cypher, drug_names=drugs)

        for r in results:
            warnings.append(
                {
                    "type": "drug-condition",
                    "severity": r["severity"] or "moderate",
                    "message": f"{r['drug']} may be contraindicated in patients with {r['condition']}.",
                    "drug": r["drug"],
                    "condition": r["condition"],
                }
            )

    driver.close()
    return warnings


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_response(reason: str) -> Dict[str, Any]:
    return {
        "safe": True,
        "warnings": [],
        "recommendations": [reason],
    }
