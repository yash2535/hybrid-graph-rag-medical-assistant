from flask import Flask, render_template, request, jsonify
import logging
import os

# Flask Extensions
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Project Imports
from app.knowledge_graph.patient_graph_reader import get_patient_profile, create_patient
from app.knowledge_graph.autopilot import analyze_health_intent, apply_graph_update
from app.knowledge_graph.wearables_graph import get_wearable_summary
from app.knowledge_graph.drug_interactions import check_drug_interactions
from app.vector_store.paper_search import search_papers
from app.rag.prompt_builder import build_medical_prompt
from app.rag.claim_extractor import extract_claims
from app.llm.ollama_client import call_ollama
from app.utils.logger import get_logger

# ============================================================
# App Initialization
# ============================================================
app = Flask(__name__)
logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================
# Database Configuration
# ============================================================
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "yash2535")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "medical_ai_user")

app.config["SQLALCHEMY_DATABASE_URI"] = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret-key-change-this")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = 3600

db = SQLAlchemy(app)
jwt = JWTManager(app)

# ============================================================
# Database Model
# ============================================================
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(50), default="patient")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
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


@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            return jsonify({"success": False, "error": "Invalid credentials"}), 401

        access_token = create_access_token(identity=username)
        return jsonify({"success": True, "access_token": access_token, "username": username})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
@jwt_required()
def ask_question():
    try:
        user_id = get_jwt_identity()
        data = request.json
        question = data.get("question", "")

        if not question:
            return jsonify({"success": False, "error": "question is required"}), 400

        logger.info(f"Processing question for user {user_id}")

        # --- 1. AUTOPILOT (Multi-Fact) ---
        facts = analyze_health_intent(question)
        suggestions_payload = []
        if facts:
            for fact in facts:
                suggestions_payload.append({
                    "message": f"I noticed you mentioned '{fact['original_term']}'. Add '{fact['normalized_term']}' to your profile?",
                    "data": {
                        "category": fact["category"],
                        "entity": fact["normalized_term"],
                    },
                })

        # --- 2. RAG PIPELINE ---
        patient_profile = get_patient_profile(user_id)
        if not patient_profile:
            create_patient(user_id=user_id)
            patient_profile = get_patient_profile(user_id)

        # FIX: fetch all data sources
        wearables_summary = get_wearable_summary(user_id)
        papers = search_papers(query=question, top_k=3)
        medications = patient_profile.get("medications", []) if patient_profile else []
        drug_interactions = check_drug_interactions(medications=medications)

        # FIX: use correct key names matching prompt_builder.py expectations
        # "wearables_data" → "wearables"
        # "drug_interactions" → "drug_facts"
        context = {
            "patient": patient_profile,
            "wearables": wearables_summary,       # ✅ Fixed: was "wearables_data"
            "papers": papers,
            "drug_facts": drug_interactions,      # ✅ Fixed: was "drug_interactions"
        }

        prompt = build_medical_prompt(question=question, context=context)
        response = call_ollama(prompt)
        claims = extract_claims(response)

        return jsonify({
            "success": True,
            "answer": response,
            "claims": claims,
            "suggestions": suggestions_payload,
            "context": {
                "patient_name": user_id,
                "papers_found": len(papers),
                "wearables_available": wearables_summary.get("available", False),  # ✅ Fixed: was bool(wearables_summary) which is always True
            },
        })

    except Exception as e:
        logger.exception("Error processing question")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/confirm_update", methods=["POST"])
@jwt_required()
def confirm_update():
    try:
        user_id = get_jwt_identity()
        data = request.json
        success, message = apply_graph_update(
            user_id,
            data.get("category"),
            data.get("entity"),
        )

        if success:
            return jsonify({"success": True, "message": message})
        else:
            return jsonify({"success": False, "error": message}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Medical Assistant API running"})


if __name__ == "__main__":
    with app.app_context():
        try:
            db.create_all()
        except Exception as e:
            print(f"❌ DB Error: {e}")
    app.run(debug=True, host="0.0.0.0", port=5000)