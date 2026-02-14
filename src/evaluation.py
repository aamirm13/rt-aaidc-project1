from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from app import RAGAssistant


@dataclass
class TestCase:
    query: str
    expected_source: str  # must match the filename in /data exactly (e.g., "quantum_computing.txt")
    expected_keyword: Optional[str] = None  # keyword sanity check within retrieved docs


def _safe_first(lst, default=None):
    return lst[0] if lst else default


def evaluate_retrieval(
    assistant: RAGAssistant,
    test_cases: List[TestCase],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Produces TWO metrics:
      1) retrieval_hit_rate: evaluates raw VectorDB retrieval (no relevance threshold gating).
      2) post_threshold_hit_rate: evaluates assistant.invoke() behavior AFTER relevance threshold gating.

    Notes:
      - retrieval_hit_rate measures whether the expected_source is in the top_k retrieved sources.
      - post_threshold_hit_rate measures whether expected_source appears in sources returned by invoke()
        (which may be empty if the relevance threshold rejects the query).
    """
    retrieval_hits = 0
    post_threshold_hits = 0

    details: List[Dict[str, Any]] = []

    for tc in test_cases:
        
        # 1) RAW RETRIEVAL (no gating)
        
        raw = assistant.vector_db.search(tc.query, n_results=top_k)

        raw_docs = (raw.get("documents") or [[]])[0]
        raw_metas = (raw.get("metadatas") or [[]])[0]
        raw_dists = (raw.get("distances") or [[]])[0]

        raw_sources = [m.get("source") for m in raw_metas if isinstance(m, dict)]
        raw_hit = tc.expected_source in raw_sources
        retrieval_hits += int(raw_hit)

        raw_best_distance = _safe_first(raw_dists, None)

        # Optional sanity check: does any retrieved chunk contain the expected keyword?
        keyword_hit = None
        if tc.expected_keyword:
            keyword = tc.expected_keyword.lower()
            keyword_hit = any((doc or "").lower().find(keyword) != -1 for doc in raw_docs)

        # 2) POST-THRESHOLD (invoke() includes relevance gate)
        
        invoked = assistant.invoke(tc.query, n_results=top_k, show_scores=True)

        invoked_sources = [s.get("source") for s in (invoked.get("sources") or []) if isinstance(s, dict)]
        post_hit = tc.expected_source in invoked_sources
        post_threshold_hits += int(post_hit)

        invoked_dists = invoked.get("distances") or []
        post_best_distance = _safe_first(invoked_dists, None)

        # The invoke() method may reject off-topic queries and return no sources.
        gated_out = bool(invoked_dists) and (not invoked_sources)

        details.append(
            {
                "query": tc.query,
                "expected_source": tc.expected_source,
                "top_k": top_k,
                # raw retrieval
                "raw_retrieved_sources": raw_sources,
                "raw_hit": raw_hit,
                "raw_best_distance": raw_best_distance,
                "raw_keyword_hit": keyword_hit,
                # post-threshold (invoke)
                "post_threshold_retrieved_sources": invoked_sources,
                "post_threshold_hit": post_hit,
                "post_threshold_best_distance": post_best_distance,
                "post_threshold_gated_out": gated_out,
                # for visibility
                "relevance_threshold": getattr(assistant, "relevance_threshold", None),
            }
        )

    n = len(test_cases)
    retrieval_hit_rate = (retrieval_hits / n) if n else 0.0
    post_threshold_hit_rate = (post_threshold_hits / n) if n else 0.0

    return {
        "top_k": top_k,
        "num_cases": n,
        "retrieval_hit_rate": retrieval_hit_rate,
        "post_threshold_hit_rate": post_threshold_hit_rate,
        "details": details,
    }


def main():
    # IMPORTANT: set expected_source filenames to match your /data filenames exactly.
    test_cases = [
        TestCase(query="What is quantum computing?", expected_source="quantum_computing.txt"),
        TestCase(query="What is biotechnology?", expected_source="biotechnology.txt"),
        TestCase(query="What is sustainable energy?", expected_source="sustainable_energy.txt"),
        # keyword sanity-check example:
        # TestCase(query="Explain quantum superposition.", expected_source="quantum_computing.txt", expected_keyword="superposition"),
    ]

    assistant = RAGAssistant()

    # Ensure DB is populated (same behavior as your app.py main)
    if assistant.vector_db.count() == 0:
        print("Vector DB is empty. Running ingestion first...")
        assistant.ingest("data")

    report = evaluate_retrieval(assistant, test_cases, top_k=5)

    print("\n=== Retrieval Evaluation Report ===")
    print(f"Top-K: {report['top_k']}")
    print(f"Cases: {report['num_cases']}")
    print(f"Raw Retrieval Top-K Hit Rate: {report['retrieval_hit_rate']:.2%}")
    print(f"Post-Threshold Top-K Hit Rate: {report['post_threshold_hit_rate']:.2%}\n")

    for i, row in enumerate(report["details"], start=1):
        print(f"[{i}] Query: {row['query']}")
        print(f"    Expected source: {row['expected_source']}")
        print(f"    Relevance threshold: {row['relevance_threshold']}")
        print("    --- Raw retrieval (no gating) ---")
        print(f"    Hit: {row['raw_hit']}")
        print(f"    Best distance: {row['raw_best_distance']}")
        if row["raw_keyword_hit"] is not None:
            print(f"    Keyword hit: {row['raw_keyword_hit']}")
        print(f"    Retrieved sources: {row['raw_retrieved_sources']}")
        print("    --- Post-threshold (invoke) ---")
        print(f"    Hit: {row['post_threshold_hit']}")
        print(f"    Best distance: {row['post_threshold_best_distance']}")
        print(f"    Gated out (distances exist but sources empty): {row['post_threshold_gated_out']}")
        print(f"    Retrieved sources: {row['post_threshold_retrieved_sources']}\n")


if __name__ == "__main__":
    main()
