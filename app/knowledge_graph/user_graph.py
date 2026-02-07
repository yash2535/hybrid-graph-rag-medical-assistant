import os
from typing import Dict, List, Any
from neo4j import GraphDatabase

from app.processing.entity_extractor import extract_medical_entities


def _get_driver():
    return GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(
            os.environ.get("NEO4J_USER", "neo4j"),
            os.environ.get("NEO4J_PASSWORD", "password"),
        ),
    )


def upsert_user_from_question(user_id: str, question: str) -> Dict[str, List[str]]:
    entities = extract_medical_entities(question)
    conditions = entities.get("conditions", [])

    if not conditions:
        return entities

    driver = _get_driver()
    cypher = """
    MERGE (u:User { id: $user_id })
    WITH u
    UNWIND $conditions AS cond
      MERGE (c:Condition { name: toLower(cond) })
      MERGE (u)-[:HAS_CONDITION]->(c)
    """

    with driver.session() as session:
        session.run(cypher, user_id=user_id, conditions=conditions)

    driver.close()
    return entities


def get_user_context(user_id: str) -> Dict[str, Any]:
    driver = _get_driver()
    cypher = """
    MATCH (u:User { id: $user_id })
    OPTIONAL MATCH (u)-[:HAS_CONDITION]->(c:Condition)
    OPTIONAL MATCH (u)-[:TAKES_DRUG]->(d:Drug)
    RETURN
      collect(DISTINCT c.name) AS conditions,
      collect(DISTINCT d.name) AS drugs
    """

    with driver.session() as session:
        record = session.run(cypher, user_id=user_id).single()

    driver.close()

    return {
        "conditions": sorted(filter(None, record["conditions"])),
        "drugs": sorted(filter(None, record["drugs"])),
    }
