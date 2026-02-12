from flask import Flask, render_template, request, jsonify
import logging
import os

# Flask Extensions
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Project Imports
from app.knowledge_graph.patient_graph_reader import (
    upsert_user_from_question,
    get_patient_profile,
    create_patient,
)
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
# PostgreSQL Configuration (Environment Based)
# ============================================================
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "yash2535")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "medical_ai_user")

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ============================================================
# JWT Configuration
# ============================================================
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "super-secret-key-change-this")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = 3600  # 1 hour

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
# Frontend Route
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================
@app.route("/api/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"success": False, "error": "Username and password required"}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({"success": False, "error": "User already exists"}), 400

        new_user = User(username=username)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        # Create Neo4j patient node
        create_patient(user_id=username)

        return jsonify({"success": True, "message": "User registered successfully"})

    except Exception:
        logger.exception("Registration error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        user = User.query.filter_by(username=username).first()

        if not user:
            return jsonify({"success": False, "error": "User not found"}), 404

        if not user.check_password(password):
            return jsonify({"success": False, "error": "Invalid credentials"}), 401

        access_token = create_access_token(identity=username)

        return jsonify({
            "success": True,
            "access_token": access_token
        })

    except Exception:
        logger.exception("Login error")
        return jsonify({"success": False, "error": "Internal server error"}), 500


# ============================================================
# PROTECTED MEDICAL ROUTE
# ============================================================
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

        upsert_user_from_question(user_id, question)
        patient_profile = get_patient_profile(user_id)

        if not patient_profile:
            return jsonify({"success": False, "error": "Patient not found"}), 404

        wearables_summary = get_wearable_summary(user_id)
        papers = search_papers(query=question, top_k=3)

        medications = patient_profile.get("medications", [])
        drug_interactions = check_drug_interactions(medications=medications)

        context = {
            "patient": patient_profile,
            "wearables_data": wearables_summary,
            "papers": papers,
            "drug_interactions": drug_interactions,
        }

        prompt = build_medical_prompt(question=question, context=context)
        response = call_ollama(prompt)
        claims = extract_claims(response)

        return jsonify({
            "success": True,
            "answer": response,
            "claims": claims
        })

    except Exception:
        logger.exception("Error processing question")
        return jsonify({"success": False, "error": "Internal server error"}), 500


# ============================================================
# HEALTH CHECK
# ============================================================
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Medical Assistant API running (PostgreSQL + JWT Enabled)"
    })


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
