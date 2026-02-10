"""
Wearables Graph Reader

Purpose:
- Read wearable / vitals FACTS from Neo4j
- Compute deterministic statistical summaries
- NO clinical interpretation
- NO causal reasoning

Design Principles:
- Deterministic
- Auditable
- LLM-safe (facts + computed observations only)

Neo4j Domain:
Patient → WearableMetric → WearableReading
"""

from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
import os
from statistics import mean


# ------------------------------------------------------------------
# Neo4j connection
# ------------------------------------------------------------------

def _get_driver():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(user, password))


# ------------------------------------------------------------------
# READ: Wearable FACT summaries
# ------------------------------------------------------------------

def get_wearable_summary(user_id: str) -> Dict[str, Any]:
    """
    Fetch wearable metrics and return computed FACT summaries.
    No medical or behavioral interpretation is performed here.
    """

    driver = _get_driver()

    cypher = """
    MATCH (p:Patient {id: $user_id})
    OPTIONAL MATCH (p)-[:HAS_METRIC]->(m:WearableMetric)
    OPTIONAL MATCH (m)-[:RECORDED_AS]->(r:WearableReading)
    RETURN
      m.type     AS metric_type,
      collect(r) AS readings
    """

    metrics = []

    with driver.session() as session:
        results = session.run(cypher, user_id=user_id)

        for record in results:
            summary = _summarize_metric(
                record["metric_type"],
                record["readings"],
            )
            if summary:
                metrics.append(summary)

    driver.close()

    return {
        "facts_version": "1.0",
        "available": bool(metrics),
        "metrics": metrics,
    }


# ------------------------------------------------------------------
# Helpers (COMPUTED OBSERVATIONS ONLY)
# ------------------------------------------------------------------

def _summarize_metric(
    metric_type: str,
    readings: List[Any]
) -> Optional[Dict[str, Any]]:
    """
    Compute deterministic statistical summaries for a wearable metric.
    """

    if not metric_type or not readings:
        return None

    values = []
    timestamps = []

    for r in readings:
        if not r:
            continue
        val = r.get("value")
        ts = r.get("timestamp")
        if val is not None:
            values.append(val)
            timestamps.append(ts)

    if not values:
        return None

    return {
        "metric": metric_type,
        "latest_value": values[-1],
        "average_value": round(mean(values), 2),
        "min_value": min(values),
        "max_value": max(values),
        "trend": _detect_trend(values),  # mathematical trend only
        "readings_count": len(values),
        "time_range": {
            "start": str(timestamps[0]) if timestamps else None,
            "end": str(timestamps[-1]) if timestamps else None,
        }
    }


def _detect_trend(values: List[float]) -> str:
    """
    Simple mathematical trend detection.
    This is NOT a clinical interpretation.
    """

    if len(values) < 3:
        return "insufficient-data"

    first = values[0]
    last = values[-1]

    if last > first * 1.05:
        return "increasing"
    if last < first * 0.95:
        return "decreasing"
    return "stable"
