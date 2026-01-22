# RAG Assistant

**LangChain + ChromaDB | ReadyTensor Agentic AI Developer Certification**

---

## Overview

This project implements a Retrieval-Augmented Generation (RAG) assistant that answers user questions using a local document collection (data/) instead of relying on static model training knowledge.

The system solves the knowledge limitation and hallucination problem in large language models by grounding responses in retrieved source documents. Instead of generating speculative answers, the assistant retrieves semantically relevant information from a curated knowledge base and produces source-backed, verifiable responses.

This project demonstrates a complete, production-oriented RAG pipeline:

Ingest → Chunk → Embed → Store → Retrieve → Generate

It is developed as part of the ReadyTensor Agentic AI Developer Certification and follows the architectural principles taught in the learning material on embeddings, vector databases, chunking strategies, semantic retrieval, and retrieval-augmented generation.

---

## Key Features

### Core RAG Capabilities:

- Vector database retrieval using ChromaDB (cosine similarity / HNSW index)

- Semantic chunking with overlap via RecursiveCharacterTextSplitter

- SentenceTransformers embeddings (sentence-transformers/all-MiniLM-L6-v2)

- Persistent vector storage using ChromaDB persistence

- Multi-LLM backend support:

- OpenAI (ChatOpenAI)

- Groq (ChatGroq)

- Google Gemini (ChatGoogleGenerativeAI)

- Environment-based model switching using .env configuration

### Safety, Reliability, and Control

**Relevance Thresholding:**

- Vector distance filtering rejects off-topic queries

- Prevents hallucinated answers for unrelated inputs

**Prompt Hardening:**

- System prompt enforces grounded answers only

- Explicit refusal when information is missing

- Immunity to instruction override attempts (prompt injection resistance)

**Grounded Answering:**

- Answers use retrieved context only

- Refusal behavior:
“I don't know based on the provided documents.”

**Transparency:**

- Source filenames and chunk IDs exposed

- Vector distance scores displayed for retrieval quality evaluation

**Traceability:**

- Metadata preservation (source, chunk_id)

- Full audit trail from answer → chunk → document

---

## System Architecture

### Two-Phase RAG Pipeline:
#### 1. Knowledge Ingestion (Insertion Phase)

- Load .txt documents from /data

- Split into overlapping semantic chunks

- Generate embeddings using SentenceTransformer

- Store embeddings + text + metadata in persistent ChromaDB

#### 2. Retrieval & Generation (Inference Phase)

- Embed user query

- Perform vector similarity search

- Apply relevance threshold filtering

- Build grounded context

- Generate response using LLM

- Enforce citation + refusal logic

---

## Dataset Sources and Structure:

- Local .txt files stored in /data

- Users can change the contents upon which their AI Assistant's knowledge base draws from, by editing the /data folder

**Multi-domain content including:**

- Artificial Intelligence

- Biotechnology

- Quantum Computing

- Sustainable Energy

- Space Exploration

**Mixed-domain design enables:**

- Cross-topic retrieval testing

- Relevance threshold validation

- Semantic filtering evaluation

Each document is treated as a first-class source

Metadata is preserved for traceability and citation

---

## Evaluation Framework

### Metrics:

- Vector Distance Scores (cosine distance from ChromaDB)

- Relevance Threshold Filtering

- Refusal Accuracy

- Grounding Consistency

- Source Traceability

- Hallucination Prevention

### Behavioral Validation:

- In-scope queries → grounded, cited answers

- Out-of-scope queries → safe refusal

- Low-confidence queries → rejection via thresholding

- No speculative generation allowed

---

## Technologies Used

- Python

- LangChain (orchestration, prompts, pipelines)

- ChromaDB (vector storage)

- Sentence-Transformers (embeddings)

- OpenAI / Groq / Google Gemini (LLM backends)

- dotenv (secure environment configuration)

- Streamlit (optional UI interface)

---


## Project Structure

```text
project-root/
│
├── src/
│   ├── app.py              # CLI RAG assistant
│   ├── Vectordb.py         # Vector database logic
│   ├── streamlit_app.py    # Optional UI interface
│
├── data/                   # Knowledge base (.txt documents)
│
├── README.md
├── requirements.txt
└── .env
```
---

## Setup Instructions

### 1. Install Dependencies
```
pip install -r requirements.txt
```
### 2. Add Documents

Place your .txt files into:

/data/

**After adding files, always run ingestion**

```
python src/app.py --mode ingest
```

### 3. Configure API Keys

This project supports the following LLM providers:

- OpenAI → ChatOpenAI

- Groq → ChatGroq

- Google Gemini → ChatGoogleGenerativeAI

**Set only one API key for the provider you wish to use.**

Create a .env file:
```
OPENAI_API_KEY=your_key_here
# or
GROQ_API_KEY=your_key_here
# or
GOOGLE_API_KEY=your_key_here
```
**Important:**

The application selects the LLM provider based on which API key is detected in the environment.

If multiple API keys are present, the system will use the first supported key it detects (based on internal priority logic).

To avoid unintended provider selection, ensure that **only the API key for your desired provider is defined in the .env file.**

For example, if using Groq, remove or comment out any OPENAI_API_KEY or GOOGLE_API_KEY entries.

Optional configuration:
```
OPENAI_MODEL=gpt-4o-mini
GROQ_MODEL=llama-3.1-8b-instant
GOOGLE_MODEL=gemini-2.0-flash

RELEVANCE_THRESHOLD=1.3
CHUNK_SIZE=900
CHUNK_OVERLAP=150
COLLECTION_NAME=rag_collection
```
**Running the Project**

CLI Mode

Navigate to project root:
```
python src/app.py
```
Ingest Only
```
python src/app.py --mode ingest
```
Chat Mode
```
python src/app.py --mode chat
```
Optional UI
```
streamlit run src/streamlit_app.py
```
**Example Behavior**

Valid Query
User: What is quantum computing?
→ Grounded answer with citations
→ Source chunks + distances shown

Off-topic Query
User: What day is it today?
→ "I don't know based on the provided documents."

Low-Relevance Query
→ Query rejected by relevance threshold
→ Safe refusal behavior

**Deployment Considerations:**

- Logging & monitoring integration

- Periodic re-embedding for data freshness

- Model version tracking

- Dataset versioning

- Threshold tuning

- Multi-format ingestion (PDF, HTML, APIs)

- Real-time ingestion pipelines

- Vector DB scaling strategies

**Educational Alignment:**

- This project directly implements concepts from the ReadyTensor learning materials:

- Embeddings & semantic search

- Chunking strategies

- Vector databases

- RAG pipelines

- Agentic memory layers

- Grounded generation

- Hallucination control

- Retrieval-based reasoning

- Trustworthy AI design

-Agentic system architecture

**Outcome**

This system transforms a general-purpose LLM into a domain-aware, evidence-grounded AI assistant capable of:

- Reliable knowledge retrieval

- Safe refusal behavior

- Transparent reasoning

- Source-traceable answers

- Controlled generation

- Production-aligned architecture

- Agentic memory foundation