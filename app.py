from flask import Flask, render_template, request, jsonify
import logging
from app.knowledge_graph.patient_graph_reader import (
    upsert_user_from_question,
    get_patient_profile,
    get_all_patients,  # ADDED
    create_patient,    # ADDED
)
from app.knowledge_graph.wearables_graph import get_wearable_summary
from app.knowledge_graph.drug_interactions import check_drug_interactions
from app.vector_store.paper_search import search_papers
from app.rag.prompt_builder import build_medical_prompt
from app.rag.claim_extractor import extract_claims
from app.llm.ollama_client import call_ollama
from app.utils.logger import get_logger

app = Flask(__name__)
logger = get_logger(__name__)

# Configure logging for Flask
logging.basicConfig(level=logging.INFO)


@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


# ------------------------------------------------------------------
# NEW: User Management Endpoints (ADDED FOR FRONTEND)
# ------------------------------------------------------------------

@app.route("/api/users", methods=["GET"])
def get_users():
    """Get all patients for dropdown."""
    try:
        patients = get_all_patients()
        return jsonify({
            "success": True,
            "data": patients
        })
    except Exception as e:
        logger.exception("Error retrieving users")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/users", methods=["POST"])
def create_user():
    """Create a new patient."""
    try:
        data = request.get_json()
        
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({
                "success": False,
                "error": "user_id is required"
            }), 400
        
        name = data.get('name')
        age = data.get('age')
        gender = data.get('gender')
        blood_type = data.get('blood_type')
        
        # Create patient
        success = create_patient(
            user_id=user_id,
            name=name,
            age=age,
            gender=gender,
            blood_type=blood_type
        )
        
        if not success:
            return jsonify({
                "success": False,
                "error": f"Patient '{user_id}' already exists"
            }), 400
        
        return jsonify({
            "success": True,
            "data": {
                "id": user_id,
                "name": name,
                "age": age,
                "gender": gender,
                "bloodType": blood_type
            },
            "message": f"Patient '{user_id}' created successfully"
        })
        
    except Exception as e:
        logger.exception("Error creating user")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ------------------------------------------------------------------
# EXISTING: Medical Question Answering
# ------------------------------------------------------------------

@app.route("/api/ask", methods=["POST"])
def ask_question():
    """
    Main API endpoint for medical questions.
    Runs the full Hybrid Graph-RAG pipeline.
    """
    try:
        data = request.json
        user_id = data.get("user_id")
        question = data.get("question", "")

        if not user_id:
            return jsonify({
                "success": False,
                "error": "user_id is required"
            }), 400

        if not question:
            return jsonify({
                "success": False,
                "error": "question is required"
            }), 400

        logger.info(f"Processing question for user {user_id}", extra={"question": question[:50]})

        # 1. Update patient graph
        upsert_user_from_question(user_id, question)

        # 2. Fetch patient profile
        patient_profile = get_patient_profile(user_id)
        
        if not patient_profile:
            return jsonify({
                "success": False,
                "error": f"Patient '{user_id}' not found"
            }), 404

        # 3. Fetch wearables
        wearables_summary = get_wearable_summary(user_id)

        # 4. Search papers
        papers = search_papers(query=question, top_k=3)

        # 5. Check drug interactions
        medications = patient_profile.get("medications", [])
        drug_interactions = check_drug_interactions(medications=medications)

        # 6. Build context
        context = {
            "patient": patient_profile,
            "wearables_data": wearables_summary,
            "papers": papers,
            "drug_interactions": drug_interactions,
        }

        # 7. Build prompt
        prompt = build_medical_prompt(question=question, context=context)

        # 8. Call LLM
        response = call_ollama(prompt)

        # 9. Extract claims
        claims = extract_claims(response)

        logger.info("Question processed successfully")

        # UPDATED: Format response for frontend
        return jsonify({
            "success": True,
            "answer": response,
            "claims": claims,
            "context": {
                "patient": {
                    "patient_id": patient_profile.get("patient_id"),
                    "name": patient_profile.get("name"),
                    "age": patient_profile.get("age"),
                    "gender": patient_profile.get("gender"),
                    "bloodType": patient_profile.get("bloodType"),
                    "conditions": [c["name"] for c in patient_profile.get("conditions", [])],
                    "medications": [m["name"] for m in patient_profile.get("medications", [])]
                },
                "drug_interactions": drug_interactions,
                "papers_found": len(papers),
                "wearables_available": wearables_summary.get("available", False),
                "wearables_data": wearables_summary.get("data", {})
            }
        })

    except Exception as e:
        logger.exception("Error processing question")
        return jsonify({
            "error": str(e), 
            "success": False
        }), 500


@app.route("/api/patient/<user_id>", methods=["GET"])
def get_patient(user_id):
    """Get patient profile."""
    try:
        profile = get_patient_profile(user_id)
        
        if not profile:
            return jsonify({
                "success": False,
                "error": f"Patient '{user_id}' not found"
            }), 404
            
        return jsonify({
            "success": True, 
            "data": profile
        })
        
    except Exception as e:
        logger.exception(f"Error fetching patient {user_id}")
        return jsonify({
            "error": str(e), 
            "success": False
        }), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok", 
        "message": "Medical Assistant API is running"
    })


if __name__ == "__main__":
    print("=" * 60)
    print("🏥 Medical Assistant - Hybrid Graph-RAG API")
    print("=" * 60)
    print("Server: http://0.0.0.0:5000")
    print("Health: http://0.0.0.0:5000/api/health")
    print("=" * 60)
    
    app.run(debug=True, host="0.0.0.0", port=5000)