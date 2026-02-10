"""
Clinical Prompt Builder

Purpose:
- Convert structured FACT context into a safe, clinical-grade LLM prompt
- Enforce medical guardrails
- Prevent hallucination, diagnosis, or prescription

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
    Build a clinical-grade prompt for medical explanation.
    Context contains FACTS ONLY.
    """

    patient = context.get("patient", {})
    wearables = context.get("wearables", {})
    papers = context.get("papers", [])
    drug_facts = context.get("drug_facts", {}) or context.get("drug_interactions", {})

    prompt = f"""
You are a clinical explanation assistant.
You are NOT a doctor.
You do NOT diagnose diseases.
You do NOT prescribe or recommend new medications.
You provide educational, safety-focused explanations only.

CRITICAL SAFETY RULES (MUST FOLLOW):
- Use ONLY the information explicitly provided below.
- Do NOT introduce new medical facts, mechanisms, or interactions.
- Do NOT infer drug interactions beyond those listed.
- Do NOT assume missing patient data.
- If information is insufficient, state this clearly.
- Your role is explanation, not decision-making.

========================
PATIENT FACTS
========================
{_format_patient(patient)}

========================
WEARABLE OBSERVATIONS (FACTS)
========================
{_format_wearables(wearables)}

========================
MEDICATION SAFETY FACTS
========================
{_format_drug_facts(drug_facts)}

========================
RELEVANT MEDICAL LITERATURE
========================
{_format_papers(papers)}

========================
USER QUESTION
========================
{question}

========================
RESPONSE GUIDELINES
========================
1. Be concise, factual, and easy to understand.
2. Do NOT make a diagnosis or confirm a medical condition.
3. Do NOT prescribe, suggest, or adjust medications.
4. Explain potential risks or considerations relevant to the patient's conditions.
5. Reference wearable trends ONLY if they are directly relevant.
6. Reference medication safety facts ONLY if they are present.
7. Clearly state when medical attention may be required.
8. If uncertainty exists, say so explicitly.
9. End the response with the sentence:
   "Consult your healthcare provider."

========================
RESPONSE FORMAT (MANDATORY)
========================
## Key Considerations
- Summarize the most relevant safety or health considerations.

## What to Monitor
- List measurable or observable factors the patient should be aware of.

## When to Seek Medical Help
- Describe situations where professional medical advice is necessary.

## Safety Notes
- Provide general, non-prescriptive safety guidance.

Respond ONLY using the provided facts.
Do NOT include any additional assumptions or external knowledge.
"""

    return prompt.strip()



# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _format_patient(patient: Dict[str, Any]) -> str:
    if not patient:
        return "No patient data available."

    lines = [f"Patient ID: {patient.get('patient_id')}"]

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
            f"Latest {m['latest_value']}, "
            f"Avg {m['average_value']}, "
            f"Trend: {m['trend']}"
        )

    return "\n".join(lines)


def _format_drug_facts(drug_facts: Dict[str, Any]) -> str:
    if not drug_facts:
        return "No medication safety data available."

    lines = []

    for f in drug_facts.get("drug_drug_interactions", []):
        lines.append(
            f"- Drug–Drug ({f['severity']}): "
            f"{', '.join(f['drugs_involved'])} → {f['interaction']}"
        )

    for f in drug_facts.get("drug_condition_interactions", []):
        lines.append(
            f"- Drug–Condition ({f['severity']}): "
            f"{f['drug']} contraindicated in {f['condition']}"
        )

    for f in drug_facts.get("drug_effect_facts", []):
        lines.append(
            f"- Drug Effect: {f['drug']} → {f['effect']} "
            f"(Mechanism: {f['mechanism']})"
        )

    return "\n".join(lines) if lines else "No known medication risks identified."


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
