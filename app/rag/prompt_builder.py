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

    patient    = context.get("patient", {})
    wearables  = context.get("wearables", {})
    papers     = context.get("papers", [])
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
- ALWAYS reference the patient's ACTUAL numbers — never use generic ranges alone.
- ALWAYS directly answer the question asked FIRST (yes/no + brief reason).
- If information is insufficient, state this clearly.
- NEVER expose internal system labels like "insufficient-data", "non-numeric",
  or "N/A" to the user — replace with plain language like "not enough data yet".
- Your role is explanation, not decision-making.
- If no research papers are provided, write ONLY:
  "No research papers available for this query."
  Do NOT add any "general knowledge" or assumptions after this.
- The Direct Answer must be YES or NO — pick one and stay consistent
  throughout the entire response.

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
Rules: Cite ONLY papers listed below. Include journal + year.
If the section below says "No research papers available", skip ## What the Research Says entirely.
========================
{_format_papers(papers)}

========================
USER QUESTION
========================
{question}

========================
RESPONSE FORMAT (MANDATORY — FOLLOW EXACTLY)
========================

## Direct Answer
- Answer YES or NO to the question first.
- In 2-3 sentences explain why, using the patient's actual data.
- Do NOT use generic explanations. Reference their real numbers.

## Your Data This Week
| Metric | Reading | Normal Range | Date |
|--------|---------|--------------|------|
[one row per reading, real values only, no placeholder text]



## Key Considerations
- Summarize the most relevant safety or health considerations.
- Maximum 3 bullet points.
- Stay strictly on topic — do NOT mention unrelated conditions.

## What to Monitor
- List specific measurable things the patient should track.
- Be concrete (e.g., "check BP every morning after waking").

## When to Seek Medical Help
- Describe clear, specific situations requiring professional attention.
- Use the patient's actual condition and medication names.

## Safety Notes
- Brief, non-prescriptive safety guidance only.
- Only mention medications or conditions relevant to the question.

Always end with exactly this line:
"Consult your healthcare provider before making any changes."

========================
STRICT OUTPUT RULES
========================
- Direct Answer: ONE word first — YES or NO. Then 2 sentences max. Do NOT skip.
- Data Table: real values and dates ONLY. Zero narrative text in table cells.
- Research: Cite ONLY the papers provided in RELEVANT MEDICAL LITERATURE above.
  Include journal name and year. If that section contains "No research papers available",
  skip ## What the Research Says entirely. Do NOT fabricate findings.
- Do NOT introduce conditions or medications not listed in Patient Facts.
- Do NOT mention any metric unrelated to the question.
- Do NOT add explanations inside table cells.
- Do NOT leak any internal system values or labels into the response.
- Do NOT use generic advice that ignores the patient's actual numbers.
- Maximum 2 sentences per bullet point.
- EVERY section is mandatory except ## What the Research Says (skip if no papers).
- Never truncate mid-sentence — shorten bullet points if needed but complete every section.
- Keep total response concise — quality over length.
========================
"""
   
    return prompt.strip()


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _format_patient(patient: Dict[str, Any]) -> str:
    if not patient:
        return "No patient data available."

    lines = [f"Patient ID: {patient.get('patient_id', 'Unknown')}"]

    # Demographics
    demo = patient.get("demographics", {})
    if demo:
        lines.append(
            f"Demographics: Age {demo.get('age', 'N/A')}, "
            f"Gender {demo.get('gender', 'N/A')}, "
            f"Blood Type {demo.get('blood_type', 'N/A')}"
        )

    # Conditions
    conditions = patient.get("conditions", [])
    if conditions:
        lines.append("\nConditions:")
        for c in conditions:
            diagnosed = f", Diagnosed: {c.get('diagnosed')}" if c.get("diagnosed") else ""
            lines.append(
                f"  - {c.get('name')} "
                f"(Severity: {c.get('severity')}, "
                f"Status: {c.get('status')}"
                f"{diagnosed})"
            )

    # Medications
    meds = patient.get("medications", [])
    if meds:
        lines.append("\nMedications:")
        for m in meds:
            purpose = f" — Purpose: {m.get('purpose')}" if m.get("purpose") else ""
            treats  = f" | Treats: {m.get('treats')}"  if m.get("treats")   else ""
            lines.append(
                f"  - {m.get('name')} "
                f"({m.get('dosage')}, {m.get('frequency')})"
                f"{purpose}{treats}"
            )

    # Lab Results
    labs = patient.get("lab_results", [])
    if labs:
        lines.append("\nRecent Lab Results:")
        for l in labs:
            lines.append(
                f"  - {l.get('name')}: {l.get('result')} {l.get('unit', '')} "
                f"(Normal: {l.get('normal_range', 'N/A')}, "
                f"Status: {l.get('status', 'N/A')}, "
                f"Date: {l.get('date', 'N/A')})"
            )

    return "\n".join(lines)


def _format_wearables(wearables: Dict[str, Any]) -> str:
    if not wearables or not wearables.get("available"):
        return "No wearable data available."

    lines = []
    for m in wearables.get("metrics", []):

        # Sanitize trend — never expose raw internal system labels
        trend = m.get("trend", "")
        if (
            not trend
            or "insufficient" in trend.lower()
            or "non-numeric"   in trend.lower()
            or trend.strip()   == "N/A"
        ):
            trend = "monitoring ongoing — more readings needed"

        lines.append(
            f"  - {m.get('metric', 'Unknown Metric')}: "
            f"Latest {m.get('latest_value', 'not recorded')}, "
            f"Previous {m.get('previous_value', 'not recorded')}, "
            f"Avg {m.get('average_value', 'not recorded')}, "
            f"Normal Range: {m.get('normal_range', 'N/A')}, "
            f"Trend: {trend}"
        )

        # Show individual dated readings if available
        readings = m.get("readings", [])
        if readings:
            for r in readings:
                date  = r.get("date",  "unknown date")
                value = r.get("value", "unknown value")
                lines.append(f"      [{date}] → {value}")

    return "\n".join(lines) if lines else "No wearable metrics recorded."


def _format_drug_facts(drug_facts: Dict[str, Any]) -> str:
    if not drug_facts:
        return "No medication safety data available."

    lines = []

    # Drug-Drug interactions
    for f in drug_facts.get("drug_drug_interactions", []):
        drugs = ", ".join(f.get("drugs_involved", []))
        lines.append(
            f"  - Drug–Drug ({f.get('severity', 'unknown')}): "
            f"{drugs} → {f.get('interaction', 'N/A')}"
        )

    # Drug-Condition interactions
    for f in drug_facts.get("drug_condition_interactions", []):
        lines.append(
            f"  - Drug–Condition ({f.get('severity', 'unknown')}): "
            f"{f.get('drug')} contraindicated in {f.get('condition')}"
        )

    # Drug effect facts
    for f in drug_facts.get("drug_effect_facts", []):
        lines.append(
            f"  - Drug Effect: {f.get('drug')} → {f.get('effect')} "
            f"(Mechanism: {f.get('mechanism', 'N/A')})"
        )

    return "\n".join(lines) if lines else "No known medication risks identified."


def _format_papers(papers: List[Dict[str, Any]]) -> str:
    if not papers:
        return "No relevant research papers found."

    lines = []
    for i, p in enumerate(papers[:3], start=1):
        lines.append(
            f"[{i}] {p.get('title', 'Untitled')} "
            f"({p.get('journal', 'Unknown Journal')}, {p.get('year', 'N/A')})"
        )
        preview = p.get("text_preview", "")[:300]
        if preview:
            lines.append(f"     Summary: {preview}")

    return "\n".join(lines)