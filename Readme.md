
```markdown
# 🧠 Hybrid Graph-RAG Medical Assistant

[![Coverage Report](https://img.shields.io/badge/Test%20Coverage-View%20Report-blue)](https://yash2535.github.io/hybrid-graph-rag-medical-assistant/coverage/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-green)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-orange)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)

> 🔎 **Live Test Coverage Dashboard**  
> https://yash2535.github.io/hybrid-graph-rag-medical-assistant/coverage/

---

## 📌 Overview

The **Hybrid Graph-RAG Medical Assistant** is a privacy-preserving medical question-answering system that combines:

- 🧩 **Structured reasoning** using a Neo4j Knowledge Graph  
- 📚 **Semantic retrieval** using Qdrant Vector Database  
- 🧠 **Local LLM inference** using Ollama  
- ⚠️ **Safety validation layer** for medical reliability  

It integrates structured patient context with research paper embeddings to generate **safe, explainable, and auditable responses** — fully offline.

---

## ✨ Key Features

- 🧠 Hybrid Graph + Vector Retrieval
- 🏥 Patient-specific reasoning using Neo4j
- 📄 Research-backed answers using semantic search
- ⚠️ Drug interaction & safety checks
- 📊 Structured claims output for transparency
- 🔒 Fully local execution (no cloud dependency)
- 🧪 300+ automated tests with coverage dashboard

---

## 🏗️ High-Level Architecture

```

User Question
│
▼
Neo4j Patient Graph ─────────────┐
├── Context Builder ───► Local LLM (Ollama)
Qdrant Research Papers ───────────┘
│
▼
Safe + Structured Medical Response

```

---

## 🧩 Core Components

| Layer | Technology | Purpose |
|-------|------------|----------|
| Graph Database | Neo4j | Patient data, conditions, medications |
| Vector Database | Qdrant | Semantic research paper retrieval |
| Embeddings | BAAI/bge-m3 | Dense vector representation |
| LLM | phi3:mini (Ollama) | Local answer generation |
| Safety Layer | Custom logic | Drug interaction & red-flag checks |
| Testing | Pytest | 300+ automated tests |

---

## 📁 Project Structure

```

hybrid-graph-rag-medical-assistant/
│
├── app/
│   ├── rag/
│   │   └── graph_rag_pipeline.py
│   ├── vector_store/
│   │   ├── qdrant_store.py
│   │   └── paper_search.py
│   ├── knowledge_graph/
│   ├── processing/
│   │   ├── embedding.py
│   │   └── entity_extractor.py
│   ├── llm/
│   │   └── ollama_client.py
│   ├── routes/
│   └── utils/
│
├── tests/
├── docs/coverage/          # GitHub Pages coverage report
├── requirements.txt
├── pytest.ini
└── README.md

```

---

## ⚙️ Prerequisites

### 1️⃣ Python
```

Python 3.9+

````

### 2️⃣ Neo4j
- Neo4j Desktop or AuraDB
- Running on: `bolt://localhost:7687`

### 3️⃣ Qdrant (Docker)

```bash
docker run -d -p 6333:6333 qdrant/qdrant
````

Check:

```
http://localhost:6333
```

### 4️⃣ Ollama (Local LLM)

Install:
👉 [https://ollama.com](https://ollama.com)

Pull lightweight model:

```bash
ollama pull phi3:mini
```

Verify:

```bash
ollama list
```

---

## 🔐 Environment Variables

Create a `.env` file:

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

## 🚀 Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Ensure Services Are Running

| Service | Status                                         |
| ------- | ---------------------------------------------- |
| Neo4j   | Running                                        |
| Qdrant  | [http://localhost:6333](http://localhost:6333) |
| Ollama  | `ollama serve`                                 |

---

## ▶️ Run the Full Pipeline

```bash
python -m app.rag.graph_rag_pipeline
```

Pipeline Steps:

1. Update patient graph
2. Retrieve patient profile
3. Retrieve wearable summaries
4. Retrieve research papers
5. Run drug interaction checks
6. Generate safe answer via LLM

---

## 🧪 Run Tests

```bash
pytest tests/test_suite.py -v
```

Generate coverage:

```bash
pytest tests/test_suite.py --cov=app --cov-report=html:docs/coverage
```

📊 Live coverage:
[https://yash2535.github.io/hybrid-graph-rag-medical-assistant/coverage/](https://yash2535.github.io/hybrid-graph-rag-medical-assistant/coverage/)

---

## 📌 Sample Output

```
===== FINAL ANSWER =====
Personalized medical guidance

===== STRUCTURED CLAIMS =====
- Risk assessment
- Monitoring advice
- Emergency warning signs
```

---

## 🚨 System Requirements

| Model     | RAM Requirement |
| --------- | --------------- |
| phi3:mini | ~3GB            |
| llama3    | >4.6GB          |

For 8GB systems → **phi3:mini recommended**

---

## 🔮 Future Improvements

* GPU acceleration
* Sparse + dense hybrid retrieval
* Clinical citation linking (PMID)
* Web UI (React / Streamlit)
* FHIR-compatible patient records
* CI/CD auto coverage deployment

---

## 📜 Disclaimer

This system is for **educational and research purposes only**.
It does not replace professional medical advice, diagnosis, or treatment.

---

## 👨‍💻 Author

**Yash Jagdale**
AI Systems | Graph RAG | Healthcare AI | Mainframe + AI Hybrid Systems

````

---

# 🚀 After Updating

Run:

```bash
git add README.md
git commit -m "Refactor README with structured professional format"
git push origin main
````

---

Your repository will now look:

✔ Structured
✔ Professional
✔ Recruiter-ready
✔ Research-grade
✔ Portfolio-strong

---

If you'd like next-level polish, I can:

* Add architecture diagram image
* Add system flow diagram (PNG)
* Add CI badge
* Add project maturity level section
* Make it conference-paper style

Just tell me 👌
