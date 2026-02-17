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
Patient → WearableMetric → Reading  (label is :Reading, not :WearableReading)
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

    # ✅ FIX 1: :Reading label (not :WearableReading) matches setup_neo4j.py
    # ✅ FIX 2: Return raw r.value and r.timestamp directly (not whole Node object)
    #           This ensures Python receives correct types (int/float vs str)
    cypher = """
    MATCH (p:Patient {id: $user_id})
    OPTIONAL MATCH (p)-[:HAS_METRIC]->(m:WearableMetric)
    OPTIONAL MATCH (m)-[:RECORDED_AS]->(r:Reading)
    RETURN
      m.type AS metric_type,
      collect({value: r.value, timestamp: toString(r.timestamp)}) AS readings
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
    Handles both numeric values (e.g. 72) and string values (e.g. "138/88", "NSR").
    """

    if not metric_type or not readings:
        return None

    numeric_values = []
    raw_values = []
    timestamps = []

    for r in readings:
        if not r:
            continue
        val = r.get("value")
        ts = r.get("timestamp")
        if val is not None:
            raw_values.append(val)
            timestamps.append(ts)
            # Collect only true numeric values for statistical computation
            if isinstance(val, (int, float)):
                numeric_values.append(float(val))

    if not raw_values:
        return None

    # Numeric metric (heart_rate, steps, blood_glucose, weight, spo2, etc.)
    if numeric_values:
        return {
            "metric": metric_type,
            "latest_value": raw_values[-1],
            "average_value": round(mean(numeric_values), 2),
            "min_value": min(numeric_values),
            "max_value": max(numeric_values),
            "trend": _detect_numeric_trend(numeric_values),
            "readings_count": len(raw_values),
            "time_range": {
                "start": timestamps[0] if timestamps else None,
                "end": timestamps[-1] if timestamps else None,
            }
        }

    # String metric (blood_pressure "138/88", ecg "NSR", etc.)
    return {
        "metric": metric_type,
        "latest_value": raw_values[-1],
        "average_value": "N/A (non-numeric)",
        "min_value": "N/A",
        "max_value": "N/A",
        "trend": _detect_string_trend(raw_values),
        "readings_count": len(raw_values),
        "time_range": {
            "start": timestamps[0] if timestamps else None,
            "end": timestamps[-1] if timestamps else None,
        }
    }


def _detect_numeric_trend(values: List[float]) -> str:
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


def _detect_string_trend(values: List[str]) -> str:
    """
    For non-numeric readings, check if value changed.
    This is NOT a clinical interpretation.
    """
    if len(values) < 2:
        return "insufficient-data"
    return "stable" if values[0] == values[-1] else "changed"
