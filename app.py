from flask import Flask
import logging

# Flask Extensions
from flask_jwt_extended import JWTManager

# Project Imports
from app.config import settings
from app.models import db
from app.routes.api import api_bp
from app.utils.logger import get_logger

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================
# App Initialization
# ============================================================
app = Flask(__name__)

# Load Database & JWT configuration from config.py
app.config["SQLALCHEMY_DATABASE_URI"]        = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"]                 = settings.JWT_SECRET_KEY
app.config["JWT_ACCESS_TOKEN_EXPIRES"]       = settings.JWT_ACCESS_TOKEN_EXPIRES

# Initialize Extensions
db.init_app(app)
jwt = JWTManager(app)

# Register the routes blueprint
app.register_blueprint(api_bp)

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    with app.app_context():
        try:
            db.create_all()
            logger.info("✅ Database tables checked/created")
        except Exception as e:
            logger.error(f"❌ DB Error: {e}")
            
    app.run(debug=True, host="0.0.0.0", port=5000)
