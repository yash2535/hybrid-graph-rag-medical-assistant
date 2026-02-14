import json
from app.llm.ollama_client import call_ollama
from app.knowledge_graph.patient_graph_reader import _get_driver as get_driver
from app.utils.logger import get_logger

logger = get_logger(__name__)

def analyze_health_intent(user_text: str):
    """
    Analyzes user text to find NEW medical facts (Diagnosis, Medication, Allergy).
    Returns a LIST of facts found.
    """
    prompt = f"""
    Analyze the user text. Identify ALL NEW medical facts about the user.
    
    Rules:
    1. Ignore questions, hypotheticals, or third-party statements.
    2. NORMALIZE terms (e.g., "sugar disease" -> "Diabetes Mellitus").
    3. If multiple facts exist (e.g., "I have fever and take aspirin"), extract BOTH.
    
    User Text: "{user_text}"
    
    Return ONLY a valid JSON ARRAY of objects. Example:
    [
        {{ "category": "Condition", "original_term": "high fever", "normalized_term": "Fever" }},
        {{ "category": "Medication", "original_term": "aspirin", "normalized_term": "Aspirin" }}
    ]
    
    If no facts found, return: []
    """
    
    try:
        response = call_ollama(prompt)
        # Clean potential markdown
        clean_json = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        
        # Ensure it's a list
        if isinstance(data, dict):
            data = [data]
            
        # Filter for valid entries
        valid_facts = [item for item in data if item.get("category") and item.get("normalized_term")]
        return valid_facts
        
    except Exception as e:
        logger.error(f"Autopilot analysis failed: {e}")
        return []

def apply_graph_update(user_id: str, category: str, entity_name: str):
    """
    Writes to Neo4j ONLY when confirmed by user.
    """
    driver = get_driver()
    query = ""
    
    if category == "Condition":
        query = """
        MATCH (u:Patient {id: $uid})
        MERGE (c:Condition {name: $name})
        MERGE (u)-[:HAS_CONDITION]->(c)
        RETURN u.id
        """
    elif category == "Medication":
        query = """
        MATCH (u:Patient {id: $uid})
        MERGE (d:Drug {name: $name})
        MERGE (u)-[:TAKES_DRUG]->(d)
        RETURN u.id
        """
    elif category == "Allergy":
        query = """
        MATCH (u:Patient {id: $uid})
        MERGE (a:Allergy {name: $name})
        MERGE (u)-[:HAS_ALLERGY]->(a)
        RETURN u.id
        """
        
    if not query:
        return False, "Invalid category"

    try:
        with driver.session() as session:
            result = session.run(query, uid=user_id, name=entity_name)
            if result.single():
                return True, f"Successfully added {category}: {entity_name}"
            else:
                return False, f"Patient {user_id} not found in Neo4j."
    except Exception as e:
        logger.error(f"Graph update failed: {e}")
        return False, str(e)