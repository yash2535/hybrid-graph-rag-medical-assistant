"""
Wearables Graph Reader

Purpose:
- Read wearable / vitals data from Neo4j
- Aggregate into clinically useful summaries
- Provide trend-level insights (NOT raw streams)

Neo4j Domain:
Patient → WearableMetric → WearableReading
"""

from typing import Dict, Any, List
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
# READ: Wearables summary
# ------------------------------------------------------------------

def get_wearable_summary(user_id: str) -> Dict[str, Any]:
    """
    Fetch wearable metrics and return summarized vitals + trends.
    LLM-ready, low-noise, clinically relevant.
    """

    driver = _get_driver()

    cypher = """
    MATCH (p:Patient {id: $user_id})
    OPTIONAL MATCH (p)-[:HAS_METRIC]->(m:WearableMetric)
    OPTIONAL MATCH (m)-[:RECORDED_AS]->(r:WearableReading)
    RETURN
      m.type        AS metric_type,
      collect(r)    AS readings
    """

    with driver.session() as session:
        results = session.run(cypher, user_id=user_id)

        metrics = []
        for record in results:
            metrics.append(
                _summarize_metric(
                    record["metric_type"],
                    record["readings"],
                )
            )

    driver.close()

    # Remove empty metrics
    metrics = [m for m in metrics if m is not None]

    return {
        "available": bool(metrics),
        "metrics": metrics,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _summarize_metric(metric_type: str, readings: List[Any]) -> Dict[str, Any] | None:
    """
    Summarize a single metric into stats + trend.
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

    summary = {
        "metric": metric_type,
        "latest": values[-1],
        "average": round(mean(values), 2),
        "min": min(values),
        "max": max(values),
        "trend": _detect_trend(values),
        "readings_count": len(values),
    }

    return summary


def _detect_trend(values: List[float]) -> str:
    """
    Simple trend detection (safe + explainable).
    """

    if len(values) < 3:
        return "insufficient data"

    first = values[0]
    last = values[-1]

    if last > first * 1.05:
        return "increasing"
    if last < first * 0.95:
        return "decreasing"
    return "stable"
