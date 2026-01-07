# Project 1 — RAG Assistant (LangChain + ChromaDB)

## Overview
This project is a Retrieval-Augmented Generation (RAG) assistant that answers questions using a local document collection (`data/`).  
It implements a full RAG pipeline: **ingest → chunk → embed → store → retrieve → generate** using LangChain and ChromaDB.

## The Key Features of this RAG Assistant:
- **Vector database retrieval** with ChromaDB (cosine similarity)
- **Chunking with overlap** using `RecursiveCharacterTextSplitter`
- **SentenceTransformers embeddings** (`all-MiniLM-L6-v2`)
- **Multi-LLM support** via environment variables (OpenAI / Groq / Google)
- **Grounded answering + citations**: answers include `[source|chunk_id]`
- **Relevance thresholding** rejects off-topic / low-similarity queries
- **Transparent scoring**: prints vector distance scores for retrieved chunks

## Requirements
- Python 3.10+ recommended
- Install dependencies:
  ```bash
  pip install -r requirements.txt

## To run:
- Ensure requirements are met
- place your desired .txt files into /data
- Complete API key configuration
- navigate to project root in terminal/ Command Prompt
- 'python src\app.py'