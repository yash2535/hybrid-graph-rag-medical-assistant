"""
Hybrid Graph-RAG Pipeline

Flow:
1. Update patient graph from question (Neo4j)
2. Fetch patient medical profile (Neo4j)
3. Fetch wearable summaries (Neo4j)
4. Search medical research papers (Qdrant)
5. Check drug interactions (Neo4j / rules)
6. Merge all context
7. Build clinical prompt
8. Generate LLM response
9. Extract structured medical claims
"""

from app.knowledge_graph.patient_graph_reader import (
    upsert_user_from_question,
    get_patient_profile,
)
from app.knowledge_graph.wearables_graph import (
    get_wearable_summary,
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


def run_hybrid_rag_pipeline():
    """
    End-to-end Hybrid Graph-RAG execution
    """

    user_id = "P001"
    question = (
        "I am feeling tired from last week. Could this be related to my diabetes or medications"
    )

    logger.info("Starting Hybrid Graph-RAG", extra={"user_id": user_id})

    # -------------------------------------------------
    # 1. Update patient medical history from question
    # -------------------------------------------------
    logger.info("Updating patient graph from question")
    upsert_user_from_question(user_id, question)

    # -------------------------------------------------
    # 2. Fetch patient medical profile (Neo4j)
    # -------------------------------------------------
    logger.info("Fetching patient profile")
    patient_profile = get_patient_profile(user_id)

    # -------------------------------------------------
    # 3. Fetch wearable data summary (Neo4j)
    # -------------------------------------------------
    logger.info("Fetching wearable summary")
    wearables_summary = get_wearable_summary(user_id)

    # -------------------------------------------------
    # 4. Search research papers (Qdrant)
    # -------------------------------------------------
    logger.info("Searching research papers")
    papers = search_papers(
        query=question,
        top_k=5,
    )

    # -------------------------------------------------
    # 5. Drug interaction safety check
    # -------------------------------------------------
    logger.info("Checking drug interactions")
    drug_interactions = check_drug_interactions(
        medications=patient_profile.get("medications", [])
    )

    # -------------------------------------------------
    # 6. Merge full context
    # -------------------------------------------------
    context = {
        "patient": patient_profile,
        "wearables": wearables_summary,
        "papers": papers,
        "drug_interactions": drug_interactions,
    }

    # -------------------------------------------------
    # 7. Build clinical-grade prompt
    # -------------------------------------------------
    prompt = build_medical_prompt(
        question=question,
        context=context,
    )

    # -------------------------------------------------
    # 8. LLM generation
    # -------------------------------------------------
    logger.info("Calling LLM")
    response = call_ollama(prompt)

    # -------------------------------------------------
    # 9. Extract structured medical claims
    # -------------------------------------------------
    claims = extract_claims(response)

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
    print("\n===== FINAL ANSWER =====\n")
    print(response)

    print("\n===== STRUCTURED CLAIMS =====\n")
    for c in claims:
        print("-", c)

    logger.info("Hybrid Graph-RAG completed")


if __name__ == "__main__":
    run_hybrid_rag_pipeline()
