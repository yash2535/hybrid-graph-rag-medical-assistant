from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_TIMEOUT: int = 30

    QDRANT_COLLECTION: str = "medical_documents"
    EMBEDDING_DIM: int = 1024
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100

    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32

    # NER / GLiNER
    NER_MODEL_NAME: str = "urchade/gliner_small-v2.1"
    NER_CONFIDENCE_THRESHOLD: float = 0.3
    NER_DEVICE: str | None = None

    NER_LABELS: list[str] = [
        "drug",
        "medical condition",
        "biomarker",
        "symptom",
    ]

    # NCBI / PubMed
    NCBI_EMAIL: str = "yashjagdale77@gmail.com"
    NCBI_REQUEST_DELAY: float = 0.5
    MIN_TEXT_LENGTH: int = 500

    QDRANT_BATCH_SIZE: int = 128

    # ✅ Pydantic v2 way
    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()
