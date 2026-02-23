from flask import Blueprint, render_template, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from app.models import db, User
from app.utils.logger import get_logger

# Project Imports
from app.knowledge_graph.patient_graph_reader import get_patient_profile, create_patient
from app.knowledge_graph.autopilot import analyze_health_intent, apply_graph_update
from app.knowledge_graph.wearables_graph import get_wearable_summary
from app.knowledge_graph.drug_interactions import check_drug_interactions
from app.vector_store.paper_search import search_papers
from app.rag.prompt_builder import build_medical_prompt
from app.rag.claim_extractor import extract_claims
from app.llm.ollama_client import call_ollama

logger = get_logger(__name__)

# Create a Blueprint object to hold the routes
api_bp = Blueprint("api", __name__)

@api_bp.route("/")
def index():
    return render_template("index.html")

@api_bp.route("/api/register", methods=["POST"])
def register():
    try:
        data     = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"success": False, "error": "Username/Password required"}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({"success": False, "error": "User already exists"}), 400

        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        create_patient(user_id=username)  # Sync to Neo4j
        return jsonify({"success": True, "message": "User registered successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/api/login", methods=["POST"])
def login():
    try:
        data     = request.get_json()
        username = data.get("username")
        password = data.get("password")

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            return jsonify({"success": False, "error": "Invalid credentials"}), 401

        access_token = create_access_token(identity=username)
        return jsonify({
            "success":      True,
            "access_token": access_token,
            "username":     username,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/api/ask", methods=["POST"])
@jwt_required()
def ask_question():
    try:
        user_id  = get_jwt_identity()
        data     = request.json
        question = data.get("question", "")

        if not question:
            return jsonify({"success": False, "error": "question is required"}), 400

        logger.info(f"Processing question for user: {user_id}")

        facts               = analyze_health_intent(question)
        suggestions_payload = []
        if facts:
            for fact in facts:
                suggestions_payload.append({
                    "message": (
                        f"I noticed you mentioned '{fact['original_term']}'. "
                        f"Add '{fact['normalized_term']}' to your profile?"
                    ),
                    "data": {
                        "category": fact["category"],
                        "entity":   fact["normalized_term"],
                    },
                })

        patient_profile = get_patient_profile(user_id)
        if not patient_profile:
            create_patient(user_id=user_id)
            patient_profile = get_patient_profile(user_id)

        wearables_summary = get_wearable_summary(user_id)
        wearable_metrics  = wearables_summary.get("metrics", [])
        wearables_count   = len(wearable_metrics)

        papers = search_papers(query=question, top_k=3)

        medications       = patient_profile.get("medications", []) if patient_profile else []
        drug_interactions = check_drug_interactions(medications=medications)

        context = {
            "patient":    patient_profile,
            "wearables":  wearables_summary,
            "papers":     papers,
            "drug_facts": drug_interactions,
        }

        prompt   = build_medical_prompt(question=question, context=context)
        response = call_ollama(prompt)
        claims   = extract_claims(response)

        conditions_count = len(patient_profile.get("conditions",  [])) if patient_profile else 0
        meds_count       = len(patient_profile.get("medications", [])) if patient_profile else 0
        labs_count       = len(patient_profile.get("lab_results", [])) if patient_profile else 0

        drug_warnings = []
        if drug_interactions:
            drug_warnings = drug_interactions.get("drug_drug_interactions", [])

        return jsonify({
            "success":     True,
            "answer":      response,
            "claims":      claims,
            "suggestions": suggestions_payload,
            "context": {
                "patient_id":          user_id,
                "conditions_count":    conditions_count,
                "meds_count":          meds_count,
                "labs_count":          labs_count,
                "wearables_available": wearables_count > 0,
                "wearables_count":     wearables_count,
                "papers_found":        len(papers),
                "drug_warnings_count": len(drug_warnings),
                "has_drug_warnings":   len(drug_warnings) > 0,
            },
        })

    except Exception as e:
        logger.exception("Error processing question")
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/api/confirm_update", methods=["POST"])
@jwt_required()
def confirm_update():
    try:
        user_id = get_jwt_identity()
        data    = request.json
        success, message = apply_graph_update(
            user_id,
            data.get("category"),
            data.get("entity"),
        )

        if success:
            return jsonify({"success": True,  "message": message})
        else:
            return jsonify({"success": False, "error": message}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@api_bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Medical Assistant API running"})