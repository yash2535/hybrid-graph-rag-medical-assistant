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
Patient → WearableMetric → Reading
"""

from typing import Dict, Any, List, Optional
from neo4j import GraphDatabase
import os
from statistics import mean


# ------------------------------------------------------------------
# Neo4j connection
# ------------------------------------------------------------------

def _get_driver():
    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(user, password))


# ------------------------------------------------------------------
# READ: Wearable FACT summaries
# ------------------------------------------------------------------

def get_wearable_summary(user_id: str) -> Dict[str, Any]:
    """
    Fetch wearable metrics and return computed FACT summaries.
    No medical or behavioral interpretation is performed here.

    Returns block compatible with prompt_builder._format_wearables()
    """

    driver = _get_driver()

    cypher = """
    MATCH (p:Patient {id: $user_id})
    OPTIONAL MATCH (p)-[:HAS_METRIC]->(wm:WearableMetric)
    OPTIONAL MATCH (wm)-[:RECORDED_AS]->(r:Reading)
    RETURN
      wm.type        AS metric_type,
      wm.name        AS metric_name,
      wm.unit        AS unit,
      wm.normalRange AS normal_range,
      collect({
          value:     toString(r.value),
          timestamp: toString(r.timestamp)
      }) AS readings
    ORDER BY wm.name
    """

    metrics = []

    with driver.session() as session:
        results = session.run(cypher, user_id=user_id)

        for record in results:
            summary = _summarize_metric(
                metric_type  = record["metric_type"],
                metric_name  = record["metric_name"],
                unit         = record["unit"],
                normal_range = record["normal_range"],
                readings     = record["readings"],
            )
            if summary:
                metrics.append(summary)

    driver.close()

    return {
        "available": bool(metrics),
        "metrics":   metrics,
    }


# ------------------------------------------------------------------
# Helpers — COMPUTED OBSERVATIONS ONLY
# ------------------------------------------------------------------

def _summarize_metric(
    metric_type:  str,
    metric_name:  str,
    unit:         str,
    normal_range: str,
    readings:     List[Any],
) -> Optional[Dict[str, Any]]:
    """
    Compute deterministic statistical summaries for a wearable metric.

    Handles:
    - Numeric values       e.g. 72, 156, 8234
    - Blood pressure       e.g. "138/88"  → uses systolic for stats
    - String values        e.g. "NSR"     → trend by equality check
    """

    if not metric_type or not readings:
        return None

    # Filter out null/empty readings
    valid_readings = [
        r for r in readings
        if r and r.get("value") not in (None, "None", "", "null")
    ]

    if not valid_readings:
        return None

    # Sort by timestamp ascending so latest is last
    valid_readings.sort(key=lambda r: r.get("timestamp", ""))

    raw_values  = [r["value"]     for r in valid_readings]
    timestamps  = [r["timestamp"] for r in valid_readings]

    # Build clean dated readings list for prompt_builder
    dated_readings = [
        {
            "date":  _clean_timestamp(r.get("timestamp", "")),
            "value": f"{r.get('value')} {unit or ''}".strip(),
        }
        for r in valid_readings
    ]

    # Compute stats
    numeric_vals = _extract_numeric_values(raw_values)

    # ── Numeric metric ───────────────────────────────────────────
    if numeric_vals:
        avg = round(mean(numeric_vals), 1)

        latest_value   = dated_readings[-1]["value"]  if dated_readings       else "not recorded"
        previous_value = dated_readings[-2]["value"]  if len(dated_readings) >= 2 else "not recorded"
        average_value  = f"{avg} {unit or ''}".strip()
        trend          = _compute_numeric_trend(numeric_vals)

        return {
            "metric":         metric_name or metric_type,
            "type":           metric_type,
            "unit":           unit,
            "normal_range":   normal_range or "N/A",
            "latest_value":   latest_value,
            "previous_value": previous_value,
            "average_value":  average_value,
            "min_value":      f"{min(numeric_vals)} {unit or ''}".strip(),
            "max_value":      f"{max(numeric_vals)} {unit or ''}".strip(),
            "trend":          trend,
            "readings_count": len(valid_readings),
            "readings":       dated_readings,
            "time_range": {
                "start": _clean_timestamp(timestamps[0])  if timestamps else None,
                "end":   _clean_timestamp(timestamps[-1]) if timestamps else None,
            },
        }

    # ── Blood pressure metric — parse systolic for stats ────────
    bp_systolic = _extract_bp_systolic(raw_values)
    if bp_systolic:
        avg            = round(mean(bp_systolic), 1)
        latest_value   = dated_readings[-1]["value"]  if dated_readings       else "not recorded"
        previous_value = dated_readings[-2]["value"]  if len(dated_readings) >= 2 else "not recorded"
        average_value  = f"{avg}/{round(mean(_extract_bp_diastolic(raw_values)), 1)} {unit or ''}".strip()
        trend          = _compute_numeric_trend(bp_systolic)

        return {
            "metric":         metric_name or metric_type,
            "type":           metric_type,
            "unit":           unit,
            "normal_range":   normal_range or "N/A",
            "latest_value":   latest_value,
            "previous_value": previous_value,
            "average_value":  average_value,
            "min_value":      f"{min(bp_systolic)} systolic",
            "max_value":      f"{max(bp_systolic)} systolic",
            "trend":          trend,
            "readings_count": len(valid_readings),
            "readings":       dated_readings,
            "time_range": {
                "start": _clean_timestamp(timestamps[0])  if timestamps else None,
                "end":   _clean_timestamp(timestamps[-1]) if timestamps else None,
            },
        }

    # ── String metric (ECG "NSR", etc.) ─────────────────────────
    latest_value   = dated_readings[-1]["value"]  if dated_readings       else "not recorded"
    previous_value = dated_readings[-2]["value"]  if len(dated_readings) >= 2 else "not recorded"
    trend          = _compute_string_trend(raw_values)

    return {
        "metric":         metric_name or metric_type,
        "type":           metric_type,
        "unit":           unit,
        "normal_range":   normal_range or "N/A",
        "latest_value":   latest_value,
        "previous_value": previous_value,
        "average_value":  "consistent readings" if trend == "stable" else "variable readings",
        "min_value":      "N/A",
        "max_value":      "N/A",
        "trend":          trend,
        "readings_count": len(valid_readings),
        "readings":       dated_readings,
        "time_range": {
            "start": _clean_timestamp(timestamps[0])  if timestamps else None,
            "end":   _clean_timestamp(timestamps[-1]) if timestamps else None,
        },
    }


# ------------------------------------------------------------------
# Numeric helpers
# ------------------------------------------------------------------

def _extract_numeric_values(raw_values: List[str]) -> List[float]:
    """
    Extract plain numeric values (non-BP, non-string).
    Skips "138/88" and "NSR" cleanly.
    """
    values = []
    for v in raw_values:
        v = str(v).strip()
        if "/" in v:
            continue  # BP format — handled separately
        try:
            values.append(float(v))
        except ValueError:
            pass  # String like "NSR" — skip cleanly
    return values


def _extract_bp_systolic(raw_values: List[str]) -> List[float]:
    """Extract systolic (first number) from BP readings like '138/88'."""
    values = []
    for v in raw_values:
        v = str(v).strip()
        if "/" in v:
            try:
                values.append(float(v.split("/")[0]))
            except ValueError:
                pass
    return values


def _extract_bp_diastolic(raw_values: List[str]) -> List[float]:
    """Extract diastolic (second number) from BP readings like '138/88'."""
    values = []
    for v in raw_values:
        v = str(v).strip()
        if "/" in v:
            try:
                values.append(float(v.split("/")[1]))
            except ValueError:
                pass
    return values


def _compute_numeric_trend(values: List[float]) -> str:
    """
    Clean human-readable trend — never exposes raw internal labels.
    Uses 5% threshold to avoid noise on small fluctuations.
    """
    if len(values) < 2:
        return "monitoring ongoing — more readings needed"

    first = values[0]
    last  = values[-1]

    if first == 0:
        return "monitoring ongoing — more readings needed"

    diff = last - first
    pct  = abs(diff / first) * 100

    if pct < 2:
        return "stable"
    elif diff > 0:
        return f"increasing ({pct:.1f}% rise over recorded period)"
    else:
        return f"decreasing ({pct:.1f}% drop over recorded period)"


def _compute_string_trend(values: List[str]) -> str:
    """
    For non-numeric readings — check if value changed.
    Never exposes raw internal labels.
    """
    if len(values) < 2:
        return "monitoring ongoing — more readings needed"
    return "stable" if values[0] == values[-1] else "changed between readings"


# ------------------------------------------------------------------
# Timestamp helper
# ------------------------------------------------------------------

def _clean_timestamp(ts: str) -> str:
    """
    Convert Neo4j timestamp to readable date.
    "2026-02-08T08:00:00Z" → "2026-02-08"
    """
    if not ts or ts in ("None", "null", ""):
        return "unknown date"
    return str(ts)[:10]