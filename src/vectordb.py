import os
import hashlib
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDB:
    """
    ChromaDB-backed vector store using SentenceTransformers embeddings.
    Stores text chunks + embeddings + metadata for retrieval-augmented generation.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        persist_path: str = "./chroma_db",
    ):
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "rag_collection")
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Persistent Chroma DB
        self.client = chromadb.PersistentClient(path=persist_path)

        # Make similarity space explicit
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Chunking
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        return self.splitter.split_text(text)

    @staticmethod
    def _make_stable_id(source: str, chunk_index: int, chunk_text: str) -> str:
        base = f"{source}|{chunk_index}|{chunk_text}"
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()
        return f"{source}_chunk_{chunk_index}_{h[:12]}"

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Ingest docs into Chroma: chunk → embed → store (with metadata + stable IDs).
        Each document is expected to be {"content": str, "metadata": dict}.
        """
        if not documents:
            print("No documents provided for ingestion.")
            return

        all_chunks: List[str] = []
        all_metadatas: List[Dict[str, Any]] = []
        all_ids: List[str] = []

        for doc in documents:
            content = (doc.get("content") or "").strip()
            metadata = doc.get("metadata") or {}

            source = metadata.get("source", "unknown_source")

            chunks = self.chunk_text(content)
            for i, chunk in enumerate(chunks):
                chunk_id = self._make_stable_id(source, i, chunk)
                all_chunks.append(chunk)
                all_metadatas.append(
                    {
                        **metadata,
                        "source": source,
                        "chunk_id": chunk_id,
                        "chunk_index": i,
                    }
                )
                all_ids.append(chunk_id)

        if not all_chunks:
            print("No chunks produced; nothing to ingest.")
            return

        print(f"Embedding {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=False).tolist()

        # Add to Chroma
        # NOTE: If you re-run ingestion, stable IDs prevent duplicates.
        self.collection.add(
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids,
            embeddings=embeddings,
        )

        print("Ingestion complete.")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def count(self) -> int:
        return self.collection.count()
