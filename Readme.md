# 🧠 Hybrid Graph-RAG Medical Assistant

> 🔎 **Live Test Coverage Dashboard:**  
> https://yash2535.github.io/hybrid-graph-rag-medical-assistant/coverage/

---

## 📌 Overview

The **Hybrid Graph-RAG Medical Assistant** is a privacy-preserving medical question-answering system that combines:

- 🧩 **Structured reasoning** using a Neo4j Knowledge Graph
- 📚 **Semantic retrieval** using a Qdrant Vector Database
- 🧠 **Local LLM inference** using Ollama
- ⚠️ **Safety validation** for medical reliability

It integrates structured patient context with research paper embeddings to generate **safe, explainable, and auditable responses** — fully offline.

---

## ✨ Key Features

- 🧠 Hybrid Graph + Vector Retrieval
- 🏥 Patient-specific reasoning via Neo4j
- 📄 Research-backed answers via semantic search
- ⚠️ Drug interaction & safety checks
- 📊 Structured claims output for transparency
- 🔒 Fully local execution — no cloud dependency
- 🧪 300+ automated tests with live coverage dashboard

---

## 🏗️ Architecture

```
User Question
      │
      ▼
Neo4j Patient Graph ──────────────┐
 ├── Context Builder ────────► Local LLM (Ollama)
Qdrant Research Papers ───────────┘
                                  │
                                  ▼
              Safe + Structured Medical Response
```

---

## 🧩 Core Components

| Layer          | Technology       | Purpose                                      |
|----------------|------------------|----------------------------------------------|
| Graph Database | Neo4j            | Patient data, conditions, medications        |
| Vector Database| Qdrant           | Semantic research paper retrieval            |
| Embeddings     | BAAI/bge-m3      | Dense vector representation                  |
| LLM            | phi3:mini (Ollama) | Local answer generation                    |
| Safety Layer   | Custom logic     | Drug interaction & red-flag checks           |
| Testing        | Pytest           | 300+ automated tests                         |

---

## 📁 Project Structure

```
hybrid-graph-rag-medical-assistant/
│
├── app/
│   ├── fetchers/
│   │   └── pubmed_fetcher.py
│   ├── ingestion/
│   │   └── pubmed_ingest.py
│   ├── knowledge_graph/
│   │   ├── autopilot.py
│   │   ├── patient_graph_reader.py
│   │   ├── setup_neo4j.py
│   │   └── wearables_graph.py
│   ├── llm/
│   │   └── ollama_client.py
│   ├── processing/
│   │   ├── chunker.py
│   │   ├── embedding.py
│   │   └── entity_extractor.py
│   ├── rag/
│   │   ├── claim_extractor.py
│   │   ├── fact_checker.py
│   │   ├── graph_rag_pipeline.py
│   │   ├── prompt_builder.py
│   │   └── qdrant_search.py
│   ├── routes/
│   │   └── api.py
│   ├── schema/
│   │   └── schema_builder.py
│   ├── utils/
│   │   └── logger.py
│   ├── vector_store/
│   │   ├── paper_search.py
│   │   └── qdrant_store.py
│   ├── config.py
│   └── models.py
│
├── docs/
│   ├── htmlcov/
│   └── templates/
├── tests/
├── app.py
├── docker-compose.yml
├── pytest.ini
└── README.md
```

---

## ⚙️ Prerequisites

### 1. Python 3.9+

### 2. Neo4j
- Neo4j Desktop or AuraDB
- Default connection: `bolt://localhost:7687`

### 3. Qdrant (via Docker)

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Verify at: `http://localhost:6333`

### 4. Ollama (Local LLM)

Install from: https://ollama.com

```bash
ollama pull phi3:mini
ollama list   # verify installation
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
```

---

## 🚀 Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Confirm All Services Are Running

| Service | Check                                          |
|---------|------------------------------------------------|
| Neo4j   | Running on `bolt://localhost:7687`             |
| Qdrant  | http://localhost:6333                          |
| Ollama  | Run `ollama serve` if not already active       |

---

## ▶️ Running the Pipeline

```bash
python -m app.rag.graph_rag_pipeline
```

The pipeline executes the following steps:

1. Update patient graph
2. Retrieve patient profile
3. Retrieve wearable summaries
4. Retrieve relevant research papers
5. Run drug interaction checks
6. Generate a safe, structured answer via LLM

---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/test_suite.py -v
```

Generate a local HTML coverage report:

```bash
pytest tests/test_suite.py --cov=app --cov-report=html:docs/coverage
```

📊 Live coverage dashboard:  
https://yash2535.github.io/hybrid-graph-rag-medical-assistant/coverage/

---

## 📌 Sample Output

```
===== FINAL ANSWER =====
Personalized medical guidance based on patient profile and research context.

===== STRUCTURED CLAIMS =====
- Risk assessment
- Monitoring advice
- Emergency warning signs
```

---

## 🚨 System Requirements

| Model      | Minimum RAM |
|------------|-------------|
| phi3:mini  | ~3 GB       |
| llama3     | > 4.6 GB    |

> For systems with 8 GB RAM, **phi3:mini** is recommended.

---

## 🔮 Future Improvements

- GPU acceleration
- Sparse + dense hybrid retrieval
- Clinical citation linking (PMID)
- Web UI (React / Streamlit)
- FHIR-compatible patient records
- CI/CD automated coverage deployment

---

## 📜 Disclaimer

This system is intended for **educational and research purposes only**.  
It does not replace professional medical advice, diagnosis, or treatment.