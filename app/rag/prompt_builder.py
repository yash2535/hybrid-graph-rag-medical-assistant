"""
Clinical Prompt Builder

Purpose:
- Convert structured context into a safe, clinical-grade LLM prompt
- Enforce medical guardrails
- Reduce hallucination and unsafe recommendations

This file contains ZERO business logic.
Only formatting + safety instructions.
"""

from typing import Dict, Any, List


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def build_medical_prompt(
    question: str,
    context: Dict[str, Any],
) -> str:
    """
    Build a clinical-grade prompt for medical Q&A.

    Inputs:
      - question (str)
      - context:
          {
            "patient": {...},
            "wearables": {...},
            "papers": [...],
            "drug_interactions": {...}
          }
    """

    patient = context.get("patient", {})
    wearables = context.get("wearables", {})
    papers = context.get("papers", [])
    drug_safety = context.get("drug_interactions", {})

    prompt = f"""
You are a clinical decision-support assistant.
You are NOT a doctor.
You MUST follow medical safety rules.

========================
PATIENT PROFILE
========================
{_format_patient(patient)}

========================
WEARABLE VITALS SUMMARY
========================
{_format_wearables(wearables)}

========================
MEDICATION SAFETY
========================
{_format_drug_safety(drug_safety)}

========================
RELEVANT MEDICAL LITERATURE
========================
{_format_papers(papers)}

========================
USER QUESTION
========================
{question}

========================
INSTRUCTIONS (STRICT)
========================
1. Be concise, factual, and practical.
2. DO NOT make a diagnosis.
3. DO NOT prescribe new medications.
4. Highlight risks relevant to the patient’s conditions.
5. Reference wearable trends if relevant.
6. Mention medication safety warnings if present.
7. Clearly state when medical attention is needed.
8. End with: "Consult your healthcare provider."

========================
RESPONSE FORMAT
========================
## Key Concerns
- ...

## What to Monitor
- ...

## When to Seek Medical Help
- ...

## Safety Notes
- ...

Respond ONLY using the provided information.
"""

    return prompt.strip()


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _format_patient(patient: Dict[str, Any]) -> str:
    if not patient:
        return "No patient data available."

    lines = [
        f"Patient ID: {patient.get('patient_id')}",
    ]

    demo = patient.get("demographics", {})
    if demo:
        lines.append(
            f"Demographics: Age {demo.get('age')}, "
            f"Gender {demo.get('gender')}, "
            f"Blood Type {demo.get('blood_type')}"
        )

    conditions = patient.get("conditions", [])
    if conditions:
        lines.append("\nConditions:")
        for c in conditions:
            lines.append(
                f"- {c.get('name')} "
                f"(Severity: {c.get('severity')}, Status: {c.get('status')})"
            )

    meds = patient.get("medications", [])
    if meds:
        lines.append("\nMedications:")
        for m in meds:
            lines.append(
                f"- {m.get('name')} ({m.get('dosage')}, {m.get('frequency')})"
            )

    return "\n".join(lines)


def _format_wearables(wearables: Dict[str, Any]) -> str:
    if not wearables or not wearables.get("available"):
        return "No wearable data available."

    lines = []
    for m in wearables.get("metrics", []):
        lines.append(
            f"- {m['metric']}: "
            f"Latest {m['latest']}, "
            f"Avg {m['average']}, "
            f"Trend: {m['trend']}"
        )

    return "\n".join(lines)


def _format_drug_safety(drug_safety: Dict[str, Any]) -> str:
    if not drug_safety:
        return "No safety data available."

    if drug_safety.get("safe"):
        return "No known drug interaction risks detected."

    lines = ["⚠️ Drug Interaction Warnings:"]
    for w in drug_safety.get("warnings", []):
        lines.append(
            f"- ({w.get('severity')}) {w.get('message')}"
        )

    return "\n".join(lines)


def _format_papers(papers: List[Dict[str, Any]]) -> str:
    if not papers:
        return "No relevant research papers found."

    lines = []
    for i, p in enumerate(papers[:3], start=1):
        lines.append(
            f"[{i}] {p.get('title')} "
            f"({p.get('journal')}, {p.get('year')})"
        )
        lines.append(
            f"    Summary: {p.get('text_preview', '')[:300]}"
        )

    return "\n".join(lines)
