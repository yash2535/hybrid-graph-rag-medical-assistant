"""
Hybrid Graph-RAG Pipeline

Flow:
1. Update patient graph from question (Neo4j)
2. Fetch patient medical profile (Neo4j) — includes labs + wearables
3. Check drug interactions (Neo4j / rules)
4. Search medical research papers (Qdrant)
5. Merge all context
6. Build clinical prompt
7. Generate LLM response
8. Extract structured medical claims
"""

from app.knowledge_graph.patient_graph_reader import (
    upsert_user_from_question,
    get_patient_profile,
)
from app.knowledge_graph.drug_interactions import (
    check_drug_interactions,
)
from app.vector_store.paper_search import (
    search_papers,
)
from app.rag.prompt_builder import (
    build_medical_prompt,
)
from app.rag.claim_extractor import (
    extract_claims,
)
from app.llm.ollama_client import (
    call_ollama,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_hybrid_rag_pipeline(
    user_id,
    question,
) -> dict:
    """
    End-to-end Hybrid Graph-RAG execution.

    Args:
        user_id:  Patient ID from Neo4j (e.g. "user_1")
        question: The patient's health question

    Returns:
        dict with keys: response, claims, context
    """

    logger.info("Starting Hybrid Graph-RAG", extra={"user_id": user_id})

    # ----------------------------------------------------------------
    # 1. Ensure patient node exists in Neo4j
    # ----------------------------------------------------------------
    logger.info("Step 1 — Upserting patient node")
    upsert_user_from_question(user_id, question)

    # ----------------------------------------------------------------
    # 2. Fetch FULL patient profile from Neo4j
    #    Now includes: demographics, conditions, medications,
    #                  lab_results (flat), wearables + readings
    # ----------------------------------------------------------------
    logger.info("Step 2 — Fetching patient profile (full graph)")
    patient_profile = get_patient_profile(user_id)

    # Wearables now come directly from patient_graph_reader
    # No separate wearables_graph call needed
    wearables = patient_profile.pop("wearables", {"available": False, "metrics": []})

    logger.info(
        "Patient profile loaded",
        extra={
            "conditions":  len(patient_profile.get("conditions",  [])),
            "medications":  len(patient_profile.get("medications",  [])),
            "lab_results":  len(patient_profile.get("lab_results",  [])),
            "wearable_metrics": len(wearables.get("metrics", [])),
        }
    )

    # ----------------------------------------------------------------
    # 3. Drug interaction safety check
    # ----------------------------------------------------------------
    logger.info("Step 3 — Checking drug interactions")
    drug_interactions = check_drug_interactions(
        medications=patient_profile.get("medications", [])
    )

    # ----------------------------------------------------------------
    # 4. Search relevant research papers (Qdrant vector search)
    # ----------------------------------------------------------------
    logger.info("Step 4 — Searching research papers")
    papers = search_papers(
        query=question,
        top_k=3,  # Keep to 3 — prompt_builder uses max 3
    )

    logger.info(f"Found {len(papers)} relevant papers")

    # ----------------------------------------------------------------
    # 5. Merge full context
    # ----------------------------------------------------------------
    context = {
        "patient":          patient_profile,
        "wearables":        wearables,
        "papers":           papers,
        "drug_interactions": drug_interactions,
    }

    logger.info("Step 5 — Context merged", extra={
        "has_wearables":     wearables.get("available", False),
        "has_papers":        len(papers) > 0,
        "has_drug_facts":    bool(drug_interactions),
    })

    
    # ----------------------------------------------------------------
    # 6. Build clinical-grade prompt
    # ----------------------------------------------------------------
    logger.info("Step 6 — Building clinical prompt")
    prompt = build_medical_prompt(
        question=question,
        context=context,
    )

    # ----------------------------------------------------------------
    # 7. LLM generation
    # ----------------------------------------------------------------
    logger.info("Step 7 — Calling LLM")
    response = call_ollama(prompt)

    # ----------------------------------------------------------------
    # 8. Extract structured medical claims
    # ----------------------------------------------------------------
    logger.info("Step 8 — Extracting medical claims")
    claims = extract_claims(response)

    # ----------------------------------------------------------------
    # Output to console
    # ----------------------------------------------------------------
    _print_results(
        user_id=user_id,
        question=question,
        patient_profile=patient_profile,
        wearables=wearables,
        response=response,
        claims=claims,
    )

    logger.info("Hybrid Graph-RAG completed successfully")

    return {
        "response": response,
        "claims":   claims,
        "context":  context,
    }


# ------------------------------------------------------------------
# Console output helper
# ------------------------------------------------------------------

def _print_results(
    user_id: str,
    question: str,
    patient_profile: dict,
    wearables: dict,
    response: str,
    claims: list,
) -> None:
    """
    Clean structured console output for debugging.
    """

    print("\n" + "=" * 60)
    print("HYBRID GRAPH-RAG PIPELINE — RESULTS")
    print("=" * 60)

    # Patient summary
    print(f"\n👤 Patient  : {patient_profile.get('name', 'Unknown')} ({user_id})")
    print(f"   Age      : {patient_profile.get('age', 'N/A')}")
    print(f"   Question : {question}")

    # Conditions
    conditions = patient_profile.get("conditions", [])
    if conditions:
        print(f"\n🏥 Conditions ({len(conditions)}):")
        for c in conditions:
            print(f"   - {c.get('name')} [{c.get('severity')}]")

    # Medications
    meds = patient_profile.get("medications", [])
    if meds:
        print(f"\n💊 Medications ({len(meds)}):")
        for m in meds:
            print(f"   - {m.get('name')} {m.get('dosage')} — {m.get('frequency')}")

    # Lab results
    labs = patient_profile.get("lab_results", [])
    if labs:
        print(f"\n🧪 Lab Results ({len(labs)}):")
        for l in labs:
            print(f"   - {l.get('name')}: {l.get('result')} {l.get('unit')} [{l.get('status')}]")

    # Wearables
    metrics = wearables.get("metrics", [])
    if metrics:
        print(f"\n⌚ Wearable Metrics ({len(metrics)}):")
        for m in metrics:
            print(f"   - {m.get('metric')}: Latest {m.get('latest_value')} | Trend: {m.get('trend')}")
            for r in m.get("readings", []):
                print(f"       [{r.get('date')}] → {r.get('value')}")

    # LLM Response
    print("\n" + "=" * 60)
    print("💬 Medical Assistant Response")
    print("=" * 60)
    print(response)

    # Claims
    print("\n" + "=" * 60)
    print("✅ Extracted Medical Claims")
    print("=" * 60)
    for c in claims:
        print(f"  {c.get('type', 'general')} — {c.get('claim', c)}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Test with different patients by changing user_id
    run_hybrid_rag_pipeline(
        
    )