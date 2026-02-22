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
Patient → WearableMetric → Reading
"""

from typing import Dict, Any, List
from neo4j import GraphDatabase
import os


# ------------------------------------------------------------------
# Neo4j connection
# ------------------------------------------------------------------

def _get_driver():
    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
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

    Fetches:
    - Demographics
    - Conditions + severity + diagnosis date
    - Medications + purpose + what they treat
    - Lab results linked to each disease
    - Wearable metrics + all dated readings
    """

    driver = _get_driver()

    # ── Core patient + conditions + medications + labs ──────────────
    core_cypher = """
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

    # ── Wearables + all readings ─────────────────────────────────────
    wearables_cypher = """
    MATCH (p:Patient {id: $user_id})-[:HAS_METRIC]->(wm:WearableMetric)
    OPTIONAL MATCH (wm)-[:RECORDED_AS]->(r:Reading)

    RETURN
      wm.id          AS metric_id,
      wm.type        AS type,
      wm.name        AS name,
      wm.unit        AS unit,
      wm.normalRange AS normal_range,
      collect({
          value:     toString(r.value),
          timestamp: toString(r.timestamp)
      }) AS readings
    ORDER BY wm.name
    """

    with driver.session() as session:
        core_record     = session.run(core_cypher,     user_id=user_id).single()
        wearable_result = session.run(wearables_cypher, user_id=user_id)
        wearable_rows   = wearable_result.data()

    driver.close()

    if not core_record:
        return {
            "patient_id":    user_id,
            "name":          None,
            "facts_version": "1.0"
        }

    patient_node = core_record["p"]

    # ── Build wearables block ────────────────────────────────────────
    wearables_block = _format_wearables(wearable_rows)

    # ── Build lab results flat list (for prompt_builder) ────────────
    lab_results_flat = _format_labs_flat(core_record["lab_tests"])

    profile = {
        "patient_id": patient_node.get("id"),
        "name":       patient_node.get("name"),
        "age":        patient_node.get("age"),
        "gender":     patient_node.get("gender"),
        "bloodType":  patient_node.get("bloodType"),
        "facts_version": "1.0",

        # ── Demographics (used by prompt_builder) ──────────────────
        "demographics": {
            "age":        patient_node.get("age"),
            "gender":     patient_node.get("gender"),
            "blood_type": patient_node.get("bloodType"),
        },

        # ── Conditions with labs + meds attached ───────────────────
        "conditions": _format_diseases(
            core_record["diseases"],
            core_record["disease_medications"],
            core_record["lab_tests"],
        ),

        # ── Patient-level medications (used by prompt_builder) ──────
        "medications": _format_medications(core_record["patient_medications"]),

        # ── Lab results flat list (used by prompt_builder) ──────────
        "lab_results": lab_results_flat,

        # ── Wearables block (used by prompt_builder) ────────────────
        "wearables": wearables_block,
    }

    return profile


# ------------------------------------------------------------------
# Support: Frontend user management
# ------------------------------------------------------------------

def get_all_patients() -> List[Dict[str, Any]]:
    """
    Get list of all patients for frontend dropdown.
    Returns basic facts only.
    """
    driver = _get_driver()

    cypher = """
    MATCH (p:Patient)
    RETURN p.id       AS id,
           p.name     AS name,
           p.age      AS age,
           p.gender   AS gender,
           p.bloodType AS bloodType
    ORDER BY p.id
    """

    with driver.session() as session:
        result   = session.run(cypher)
        patients = [
            {
                "id":        record["id"],
                "name":      record["name"],
                "age":       record["age"],
                "gender":    record["gender"],
                "bloodType": record["bloodType"],
            }
            for record in result
        ]

    driver.close()
    return patients


def create_patient(
    user_id: str,
    name: str = None,
    age: int = None,
    gender: str = None,
    blood_type: str = None,
) -> bool:
    """
    Create a new patient node (facts only).
    Returns True if created, False if already exists.
    """
    driver = _get_driver()

    with driver.session() as session:
        existing = session.run(
            "MATCH (p:Patient {id: $user_id}) RETURN p",
            user_id=user_id,
        ).single()

        if existing:
            driver.close()
            return False

        session.run(
            """
            CREATE (p:Patient {
                id:         $user_id,
                name:       $name,
                age:        $age,
                gender:     $gender,
                bloodType:  $blood_type,
                created_at: datetime()
            })
            """,
            user_id=user_id,
            name=name,
            age=age,
            gender=gender,
            blood_type=blood_type,
        )

    driver.close()
    return True


# ------------------------------------------------------------------
# Helpers — FACT NORMALIZATION ONLY
# ------------------------------------------------------------------

def _format_diseases(
    disease_nodes: List[Any],
    medication_nodes: List[Any],
    lab_nodes: List[Any],
) -> List[Dict[str, Any]]:
    """
    Normalize disease-related FACTS.
    Each disease gets its own labs and medications attached.
    """
    diseases = []

    for d in disease_nodes:
        if not d:
            continue

        disease = {
            "name":           d.get("name"),
            "category":       d.get("category"),
            "severity":       d.get("severity"),
            "status":         d.get("status"),
            "icd10":          d.get("icd10"),
            "diagnosed":      _safe_date(d.get("diagnosisDate")),
            "medications":    [],
            "lab_results":    [],
        }

        # Attach lab results
        for l in lab_nodes:
            if not l:
                continue
            disease["lab_results"].append({
                "name":         l.get("name"),
                "result":       l.get("result"),
                "unit":         l.get("unit"),
                "normal_range": l.get("normalRange"),
                "status":       l.get("status"),
                "date":         _safe_date(l.get("testDate")),
            })

        # Attach disease-specific medications
        for m in medication_nodes:
            if not m:
                continue
            disease["medications"].append({
                "name":      m.get("name"),
                "dosage":    m.get("dosage"),
                "frequency": m.get("frequency"),
                "purpose":   m.get("purpose"),
            })

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
        meds.append({
            "name":      m.get("name"),
            "dosage":    m.get("dosage"),
            "frequency": m.get("frequency"),
            "purpose":   m.get("purpose"),
            "atc_code":  m.get("atcCode"),
        })

    return meds


def _format_labs_flat(lab_nodes: List[Any]) -> List[Dict[str, Any]]:
    """
    Flat list of all lab results for prompt_builder.
    Separate from disease-attached labs so prompt_builder
    can show them in a dedicated section.
    """
    labs = []

    for l in lab_nodes:
        if not l:
            continue
        labs.append({
            "name":         l.get("name"),
            "result":       l.get("result"),
            "unit":         l.get("unit"),
            "normal_range": l.get("normalRange"),
            "status":       l.get("status"),
            "date":         _safe_date(l.get("testDate")),
        })

    return labs


def _format_wearables(wearable_rows: List[Dict]) -> Dict[str, Any]:
    """
    Build wearables block from raw Neo4j rows.

    For each metric:
    - Extracts all dated readings
    - Calculates latest, previous, average
    - Computes clean trend label (never exposes raw internals)
    - Returns block ready for prompt_builder._format_wearables()
    """

    if not wearable_rows:
        return {"available": False, "metrics": []}

    metrics = []

    for row in wearable_rows:
        raw_readings = row.get("readings", [])

        # Filter out empty readings (no value)
        valid_readings = [
            r for r in raw_readings
            if r.get("value") and r.get("value") not in ("None", "", "null")
        ]

        # Sort readings by timestamp ascending
        valid_readings.sort(key=lambda r: r.get("timestamp", ""))

        # Build clean dated readings list
        dated_readings = [
            {
                "date":  _clean_timestamp(r.get("timestamp", "")),
                "value": f"{r.get('value')} {row.get('unit', '')}".strip(),
            }
            for r in valid_readings
        ]

        # Compute latest, previous, average for numeric metrics
        latest_value   = "not recorded yet"
        previous_value = "not recorded yet"
        average_value  = "not recorded yet"
        trend          = "monitoring ongoing — more readings needed"

        numeric_vals = _extract_numeric_values(valid_readings)

        if dated_readings:
            latest_value = dated_readings[-1]["value"]

        if len(dated_readings) >= 2:
            previous_value = dated_readings[-2]["value"]

        if len(numeric_vals) >= 2:
            avg = sum(numeric_vals) / len(numeric_vals)
            average_value = f"{avg:.1f} {row.get('unit', '')}".strip()
            trend = _compute_trend(numeric_vals)

        elif len(numeric_vals) == 1:
            average_value = f"{numeric_vals[0]} {row.get('unit', '')}".strip()
            trend = "monitoring ongoing — more readings needed"

        metrics.append({
            "metric":         row.get("name", "Unknown Metric"),
            "type":           row.get("type"),
            "unit":           row.get("unit"),
            "normal_range":   row.get("normal_range", "N/A"),
            "latest_value":   latest_value,
            "previous_value": previous_value,
            "average_value":  average_value,
            "trend":          trend,
            "readings":       dated_readings,
        })

    return {
        "available": len(metrics) > 0,
        "metrics":   metrics,
    }


def _extract_numeric_values(readings: List[Dict]) -> List[float]:
    """
    Extract numeric values from readings.
    Skips non-numeric values like "NSR", "138/88" etc.
    For blood pressure takes systolic only.
    """
    values = []
    for r in readings:
        raw = str(r.get("value", "")).strip()
        # Handle blood pressure format "138/88"
        if "/" in raw:
            try:
                systolic = float(raw.split("/")[0])
                values.append(systolic)
            except ValueError:
                pass
        else:
            try:
                values.append(float(raw))
            except ValueError:
                pass  # Non-numeric like "NSR" — skip cleanly
    return values


def _compute_trend(values: List[float]) -> str:
    """
    Compute a clean human-readable trend label.
    Never exposes raw internal values.
    """
    if len(values) < 2:
        return "monitoring ongoing — more readings needed"

    first = values[0]
    last  = values[-1]
    diff  = last - first

    if abs(diff) < 2:
        return "stable"
    elif diff > 0:
        pct = (diff / first) * 100 if first != 0 else 0
        return f"increasing ({pct:.1f}% rise over recorded period)"
    else:
        pct = (abs(diff) / first) * 100 if first != 0 else 0
        return f"decreasing ({pct:.1f}% drop over recorded period)"


def _clean_timestamp(ts: str) -> str:
    """
    Convert Neo4j timestamp string to readable date.
    e.g. "2026-02-08T08:00:00Z" → "2026-02-08"
    """
    if not ts:
        return "unknown date"
    return ts[:10]  # Take YYYY-MM-DD portion only


def _safe_date(value) -> str:
    return str(value) if value else None