# 🧠 Hybrid Graph-RAG Medical Assistant

**Neo4j + Qdrant + Local LLM (Ollama)**

This project implements a **Hybrid Graph-RAG (Retrieval-Augmented Generation)** pipeline for medical question answering.  
It combines **structured patient data (Neo4j Knowledge Graph)** with **unstructured medical research papers (Qdrant Vector DB)** and generates **safe, explainable answers using a local LLM via Ollama**.

---

## ✨ Key Features

- 🧩 **Knowledge Graph Reasoning** using Neo4j (patient profile, conditions, medications)
- 📚 **Semantic Paper Retrieval** using Qdrant + Transformer embeddings
- 🧠 **Local LLM Inference** using Ollama (no cloud dependency)
- ⚠️ **Medical Safety Checks** (drug interactions, red-flag symptoms)
- 🧾 **Structured Claims Output** for explainability and auditing
- 🔒 Fully **offline & privacy-preserving**

---

## 🏗️ High-Level Architecture

```

User Question
│
▼
Neo4j Patient Graph ──────┐
├── Context Builder ──► Local LLM (Ollama)
Qdrant Research Papers ───┘
│
▼
Structured & Safe Medical Answer

```

---

## 📁 Project Structure

```

Main_Health/
│
├── app/
│   ├── rag/
│   │   └── graph_rag_pipeline.py   # Main pipeline entry
│   ├── vector_store/
│   │   ├── qdrant_store.py
│   │   └── paper_search.py
│   ├── graph/
│   │   └── patient_graph.py
│   ├── processing/
│   │   └── embedding.py
│   ├── llm/
│   │   └── ollama_client.py
│   └── utils/
│       └── logger.py
│
├── requirements.txt
├── README.md
└── .env

````

---

## ⚙️ Prerequisites

Make sure the following are installed:

### 1️⃣ Python
```bash
Python 3.9+
````

### 2️⃣ Neo4j

* Neo4j Desktop **or** Neo4j AuraDB
* Database running and accessible

### 3️⃣ Qdrant

Run Qdrant locally using Docker:

```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

Verify:

```bash
http://localhost:6333
```

### 4️⃣ Ollama (Local LLM)

Install Ollama from:
👉 [https://ollama.com](https://ollama.com)

Pull a lightweight model (recommended):

```bash
ollama pull phi3:mini
```

Verify:

```bash
ollama list
```

---

## 🔐 Environment Variables (`.env`)

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

## 🧪 Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

Activate it:

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

| Service | Command / Check                    |
| ------- | ---------------------------------- |
| Neo4j   | Running on `bolt://localhost:7687` |
| Qdrant  | `http://localhost:6333`            |
| Ollama  | `ollama serve`                     |

---

## ▶️ How to Run the Entire Project

From the project root:

```bash
python -m app.rag.graph_rag_pipeline
```

This will:

1. Update patient graph from the question
2. Fetch patient profile from Neo4j
3. Fetch wearable summaries
4. Retrieve medical papers from Qdrant
5. Perform drug interaction checks
6. Generate a **safe, explainable answer using Ollama**

---

## 📌 Sample Output

```
===== FINAL ANSWER =====
<Key medical guidance>

===== STRUCTURED CLAIMS =====
- Risk assessment
- Monitoring advice
- Emergency warning signs
```

---

## 🧠 Models Used

| Component  | Model                        |
| ---------- | ---------------------------- |
| Embeddings | BAAI/bge-m3                  |
| LLM        | phi3:mini (local via Ollama) |
| Vector DB  | Qdrant                       |
| Graph DB   | Neo4j                        |

---

## 🚨 Notes on System Requirements

* `llama3:latest` requires **> 4.6 GB RAM**
* For 8 GB systems, **phi3:mini** is recommended
* Fully local execution (no GPU required)

---

## 🔮 Future Enhancements

* GPU-accelerated inference
* Hybrid dense + sparse retrieval
* Clinical citation linking (PMID)
* Web UI (Streamlit / React)
* FHIR-compliant medical records

---

## 📜 Disclaimer

This system is for **educational and research purposes only**.
It does **not** replace professional medical diagnosis or treatment.

---




