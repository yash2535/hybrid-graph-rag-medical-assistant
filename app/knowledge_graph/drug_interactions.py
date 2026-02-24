"""
Drug Interaction Safety Engine - CORRECTED VERSION

Purpose:
- Extract VERIFIED drug interaction FACTS
- Extract VERIFIED drug effect / mechanism FACTS
- NO symptom inference
- NO patient-specific reasoning
- Used BEFORE calling the LLM

Design:
- Deterministic
- Auditable
- Knowledge-Graph + Rule based
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, Neo4jError
import os


# ------------------------------------------------------------------
# Neo4j connection
# ------------------------------------------------------------------

def _get_driver():
    """
    Creates and returns a Neo4j driver.
    Should be used with context manager or closed explicitly.
    """
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()  # Verify connection works
        return driver
    except (ServiceUnavailable, Neo4jError) as e:
        raise ConnectionError(f"Failed to connect to Neo4j at {uri}: {e}")


# ------------------------------------------------------------------
# Public API (FACT EXTRACTOR)
# ------------------------------------------------------------------

def check_drug_interactions(medications: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Entry point used by Hybrid Graph-RAG pipeline.

    Returns FACTS only:
    - drug-drug interactions
    - drug-condition interactions
    - drug-effect mechanisms
    
    Args:
        medications: List of dicts with 'name' key, e.g. [{'name': 'metformin'}, ...]
    
    Returns:
        Dict with interaction facts and metadata
    """
    
    # Validate input
    if not isinstance(medications, list):
        return _safe_response("Invalid input: medications must be a list")
    
    drug_names = sorted(
        {str(m.get("name", "")).lower().strip() 
         for m in medications if m.get("name")}
    )

    if not drug_names:
        return _safe_response("No medications provided")

    return {
        "checked_drugs": drug_names,
        "drug_drug_interactions": _check_drug_drug_facts(drug_names),
        "drug_condition_interactions": _check_drug_condition_facts(drug_names),
        "drug_effect_facts": _check_drug_effect_facts(drug_names),
    }


# ------------------------------------------------------------------
# FACT ENGINES
# ------------------------------------------------------------------

def _check_drug_effect_facts(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Extract drug effect facts from hardcoded knowledge base.
    """
    facts = []

    # ── Patient 1: John Doe — Diabetes + Hypertension ──────────────
    if "metformin" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "metformin",
            "effect": "reduced vitamin B12 absorption",
            "mechanism": "Metformin interferes with calcium-dependent B12 absorption in the terminal ileum.",
            "clinical_relevance": "Long-term use associated with B12 deficiency and peripheral neuropathy.",
            "evidence": "well-established"
        })

    if "lisinopril" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "lisinopril",
            "effect": "orthostatic hypotension and dizziness",
            "mechanism": "ACE inhibition reduces angiotensin II-mediated vasoconstriction, lowering systemic vascular resistance.",
            "clinical_relevance": "Most pronounced at initiation, dose increases, or in volume-depleted patients.",
            "evidence": "well-established"
        })

    if "amlodipine" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "amlodipine",
            "effect": "peripheral vasodilation causing dizziness and flushing",
            "mechanism": "Calcium channel blockade causes smooth muscle relaxation and peripheral vasodilation.",
            "clinical_relevance": "Dizziness risk is additive when combined with other antihypertensives.",
            "evidence": "well-established"
        })

    if "lisinopril" in drugs and "amlodipine" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "lisinopril + amlodipine",
            "effect": "additive blood pressure lowering",
            "mechanism": "Combined ACE inhibition and calcium channel blockade produces greater BP reduction than either alone.",
            "clinical_relevance": "Monitor for symptomatic hypotension, especially on standing.",
            "evidence": "clinical guidelines"
        })

    # ── Patient 2: Sarah Smith — Heart Disease ──────────────────────
    if "atorvastatin" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "atorvastatin",
            "effect": "myopathy and elevated liver enzymes",
            "mechanism": "HMG-CoA reductase inhibition can impair muscle cell energy metabolism at high doses.",
            "clinical_relevance": "Risk increases with higher doses (40-80mg). Monitor for muscle pain and LFTs.",
            "evidence": "well-established"
        })

    if "aspirin" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "aspirin",
            "effect": "GI irritation and bleeding risk",
            "mechanism": "Irreversible COX-1 inhibition reduces prostaglandin-mediated gastric mucosal protection.",
            "clinical_relevance": "Even low-dose (81mg) aspirin increases GI bleed risk, especially with age.",
            "evidence": "well-established"
        })

    if "metoprolol" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "metoprolol",
            "effect": "bradycardia, fatigue, and masking of hypoglycemia symptoms",
            "mechanism": "Beta-1 selective blockade reduces heart rate and blunts sympathetic response to low blood sugar.",
            "clinical_relevance": "Particularly relevant in diabetic patients — hypoglycemia sweating is preserved but tachycardia is masked.",
            "evidence": "well-established"
        })

    # ── Patient 3: Michael Johnson — Asthma ────────────────────────
    if "albuterol inhaler" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "albuterol inhaler",
            "effect": "tachycardia and tremor with overuse",
            "mechanism": "Beta-2 agonism causes bronchodilation but also stimulates cardiac beta-1 receptors at high doses.",
            "clinical_relevance": "Frequent rescue inhaler use (>2x/week) indicates uncontrolled asthma requiring review.",
            "evidence": "clinical guidelines"
        })

    if "fluticasone inhaler" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "fluticasone inhaler",
            "effect": "oral candidiasis and HPA axis suppression with high doses",
            "mechanism": "Inhaled corticosteroid deposits in oropharynx; systemic absorption increases at high doses.",
            "clinical_relevance": "Patients should rinse mouth after each use to prevent thrush.",
            "evidence": "well-established"
        })

    if "montelukast" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "montelukast",
            "effect": "neuropsychiatric effects including mood changes and sleep disturbance",
            "mechanism": "Leukotriene receptor antagonism in the CNS may affect neurological function.",
            "clinical_relevance": "FDA black box warning for neuropsychiatric events. Monitor for anxiety, depression, and sleep issues.",
            "evidence": "FDA black box warning"
        })

    # ── Patient 4: Emily Davis — Chronic Kidney Disease ────────────
    if "losartan" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "losartan",
            "effect": "hyperkalemia and acute kidney injury risk",
            "mechanism": "ARB blockade of angiotensin II reduces aldosterone, impairing renal potassium excretion.",
            "clinical_relevance": "CKD patients already at risk for hyperkalemia — monitor potassium closely.",
            "evidence": "well-established"
        })

    if "furosemide" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "furosemide",
            "effect": "electrolyte depletion (hypokalemia, hyponatremia) and dehydration",
            "mechanism": "Loop diuretic inhibits Na-K-2Cl cotransporter in the thick ascending limb of Henle.",
            "clinical_relevance": "Monitor electrolytes regularly. Dehydration worsens renal function in CKD.",
            "evidence": "well-established"
        })

    if "furosemide" in drugs and "losartan" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "furosemide + losartan",
            "effect": "opposing potassium effects requiring close monitoring",
            "mechanism": "Furosemide lowers potassium; losartan raises it. Net effect varies by dose and renal function.",
            "clinical_relevance": "In CKD, losartan's hyperkalemic effect often dominates — monitor K+ levels frequently.",
            "evidence": "clinical guidelines"
        })

    if "erythropoietin" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "erythropoietin",
            "effect": "hypertension and thrombotic events",
            "mechanism": "Increased red cell mass raises blood viscosity and can activate platelet aggregation.",
            "clinical_relevance": "Target hemoglobin should not exceed 11-12 g/dL in CKD to minimize cardiovascular risk.",
            "evidence": "well-established"
        })

    # ── Patient 5: Robert Brown — Diabetes + Hypertension + Heart Disease ──
    if "insulin glargine" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "insulin glargine",
            "effect": "hypoglycemia and weight gain",
            "mechanism": "Basal insulin lowers fasting glucose but risks overcorrection, especially with missed meals.",
            "clinical_relevance": "Elderly patients have heightened hypoglycemia risk and may not feel classic warning symptoms.",
            "evidence": "well-established"
        })

    if "insulin glargine" in drugs and "metformin" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "insulin glargine + metformin",
            "effect": "additive glucose lowering with increased hypoglycemia risk",
            "mechanism": "Insulin directly lowers glucose; metformin reduces hepatic glucose output — combined effect is synergistic.",
            "clinical_relevance": "Monitor fasting glucose closely. Hypoglycemia risk is higher in elderly patients.",
            "evidence": "clinical guidelines"
        })

    if "carvedilol" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "carvedilol",
            "effect": "bradycardia, hypotension, and masking of hypoglycemia",
            "mechanism": "Non-selective beta blockade reduces HR and BP; blunts tachycardia response to hypoglycemia.",
            "clinical_relevance": "High caution in diabetic patients on insulin — sweating is preserved but palpitations masked.",
            "evidence": "well-established"
        })

    if "carvedilol" in drugs and "amlodipine" in drugs:
        facts.append({
            "type": "drug-effect", "drug": "carvedilol + amlodipine",
            "effect": "additive hypotension and bradycardia",
            "mechanism": "Beta blockade combined with calcium channel blockade produces compounded negative chronotropic and vasodilatory effects.",
            "clinical_relevance": "Monitor BP and HR closely. Risk of symptomatic hypotension especially on standing.",
            "evidence": "clinical guidelines"
        })

    return facts


def _check_drug_condition_facts(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Drug–condition contraindication FACTS via Neo4j.
    
    Raises:
        ConnectionError if Neo4j is unavailable
    """
    facts = []
    driver = None
    
    try:
        driver = _get_driver()
        
        cypher = """
        MATCH (d:Medication)
        WHERE toLower(d.name) IN $drug_names
        MATCH (d)-[:CONTRAINDICATED_IN]->(c:Disease)
        RETURN d.name AS drug, c.name AS condition, c.severity AS severity
        """

        with driver.session() as session:
            results = session.run(cypher, drug_names=drugs)
            for r in results:
                facts.append({
                    "type": "drug-condition-interaction",
                    "drug": r["drug"],
                    "condition": r["condition"],
                    "severity": r["severity"] or "moderate",
                    "evidence": "knowledge graph"
                })
    
    except (ConnectionError, Exception) as e:
        # Log but don't crash — gracefully degrade
        print(f"Warning: Could not query Neo4j for drug-condition facts: {e}")
        facts.append({
            "type": "error",
            "message": "Drug-condition interaction check unavailable",
            "reason": str(e)
        })
    
    finally:
        if driver:
            driver.close()
    
    return facts


def _check_drug_drug_facts(drugs: List[str]) -> List[Dict[str, Any]]:
    """
    Extract drug-drug interaction facts from hardcoded rules.
    This function is deterministic and requires no external dependencies.
    """
    RULES = [
        # Existing rules
        {
            "drugs": {"metformin", "contrast dye"},
            "severity": "high",
            "interaction": "Increased risk of lactic acidosis",
            "mechanism": "Contrast agents may impair renal function, leading to metformin accumulation.",
            "evidence": "clinical literature"
        },
        {
            "drugs": {"metformin", "insulin"},
            "severity": "moderate",
            "interaction": "Increased risk of hypoglycemia",
            "mechanism": "Both drugs lower blood glucose levels.",
            "evidence": "clinical guidelines"
        },
        {
            "drugs": {"metformin", "alcohol"},
            "severity": "high",
            "interaction": "Increased risk of lactic acidosis",
            "mechanism": "Alcohol affects hepatic lactate metabolism.",
            "evidence": "drug safety literature"
        },
        # Patient 2: Heart Disease
        {
            "drugs": {"aspirin", "atorvastatin"},
            "severity": "low",
            "interaction": "Minor increase in bleeding risk",
            "mechanism": "Aspirin inhibits platelet aggregation; statins have mild antiplatelet properties.",
            "evidence": "clinical literature"
        },
        {
            "drugs": {"metoprolol", "aspirin"},
            "severity": "low",
            "interaction": "NSAIDs may reduce antihypertensive efficacy",
            "mechanism": "Prostaglandin inhibition by aspirin can counteract beta-blocker BP effects.",
            "evidence": "clinical guidelines"
        },
        # Patient 3: Asthma
        {
            "drugs": {"albuterol inhaler", "montelukast"},
            "severity": "low",
            "interaction": "Complementary mechanisms — no adverse interaction",
            "mechanism": "Beta-2 agonist and leukotriene antagonist act on different pathways.",
            "evidence": "clinical guidelines"
        },
        # Patient 4: CKD
        {
            "drugs": {"losartan", "furosemide"},
            "severity": "moderate",
            "interaction": "Risk of acute kidney injury and electrolyte imbalance",
            "mechanism": "Volume depletion from furosemide activates RAAS; ARB blockade then impairs compensatory response.",
            "evidence": "clinical guidelines"
        },
        {
            "drugs": {"furosemide", "calcium carbonate"},
            "severity": "low",
            "interaction": "Reduced furosemide absorption",
            "mechanism": "Calcium may bind to furosemide in the GI tract, reducing bioavailability.",
            "evidence": "pharmacokinetic data"
        },
        # Patient 5: Multi-condition
        {
            "drugs": {"insulin glargine", "carvedilol"},
            "severity": "moderate",
            "interaction": "Masking of hypoglycemia symptoms",
            "mechanism": "Non-selective beta blockade blunts tachycardia response to low blood sugar.",
            "evidence": "well-established"
        },
        {
            "drugs": {"metformin", "carvedilol"},
            "severity": "low",
            "interaction": "Carvedilol may impair glycemic control",
            "mechanism": "Beta blockade can inhibit glycogenolysis and mask hypoglycemia signs.",
            "evidence": "clinical literature"
        },
        {
            "drugs": {"aspirin", "carvedilol"},
            "severity": "low",
            "interaction": "NSAIDs may attenuate beta-blocker antihypertensive effect",
            "mechanism": "Prostaglandin inhibition reduces vasodilatory compensation.",
            "evidence": "clinical guidelines"
        },
        {
            "drugs": {"atorvastatin", "aspirin"},
            "severity": "low",
            "interaction": "Minor additive bleeding risk",
            "mechanism": "Both have mild antiplatelet properties.",
            "evidence": "clinical literature"
        },
    ]

    facts = []
    drug_set = set(drugs)

    for rule in RULES:
        if rule["drugs"].issubset(drug_set):
            facts.append({
                "type": "drug-drug-interaction",
                "drugs_involved": sorted(rule["drugs"]),
                "severity": rule["severity"],
                "interaction": rule["interaction"],
                "mechanism": rule["mechanism"],
                "evidence": rule["evidence"],
            })

    return facts


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _safe_response(reason: str) -> Dict[str, Any]:
    """
    Return a standardized empty response with a note.
    """
    return {
        "checked_drugs": [],
        "drug_drug_interactions": [],
        "drug_condition_interactions": [],
        "drug_effect_facts": [],
        "note": reason,
    }