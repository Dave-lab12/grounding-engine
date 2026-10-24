"""BEIR dataset loader support (nfcorpus and other BEIR datasets)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tests.evaluation.benchmark_schema import BenchmarkCase


@dataclass
class BEIRQrel:
    query_id: str
    doc_id: str
    score: int


@dataclass
class BEIRCorpusEntry:
    doc_id: str
    title: str
    text: str
    url: str = ""


@dataclass
class BEIRQuery:
    query_id: str
    text: str
    url: str = ""


@dataclass
class BEIRBenchmarkCase:
    """BEIR-formatted benchmark case with ground-truth relevance labels."""

    query_id: str
    query: str
    corpus_ids: list[str]
    corpus_titles: dict[str, str]
    corpus_texts: dict[str, str]
    qrel_scores: dict[str, int]
    query_url: str = ""
    domain: str = "mixed"


def load_corpus(corpus_path: Path) -> dict[str, BEIRCorpusEntry]:
    import json

    docs: dict[str, BEIRCorpusEntry] = {}
    with corpus_path.open() as fh:
        for line in fh:
            entry = json.loads(line)
            docs[entry["_id"]] = BEIRCorpusEntry(
                doc_id=entry["_id"],
                title=entry.get("title", ""),
                text=entry.get("text", ""),
                url=entry.get("metadata", {}).get("url", ""),
            )
    return docs


def load_queries(queries_path: Path) -> dict[str, BEIRQuery]:
    import json

    queries: dict[str, BEIRQuery] = {}
    with queries_path.open() as fh:
        for line in fh:
            entry = json.loads(line)
            queries[entry["_id"]] = BEIRQuery(
                query_id=entry["_id"],
                text=entry["text"],
                url=entry.get("metadata", {}).get("url", ""),
            )
    return queries


def load_qrels(qrel_path: Path) -> dict[str, list[BEIRQrel]]:
    qrels: dict[str, list[BEIRQrel]] = {}
    with qrel_path.open() as fh:
        fh.readline()  # the header row
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                if query_id not in qrels:
                    qrels[query_id] = []
                qrels[query_id].append(BEIRQrel(query_id=query_id, doc_id=doc_id, score=score))
    return qrels


@dataclass
class BEIRLoader:
    """Loads a BEIR-compatible dataset directory for benchmark use."""

    root: Path
    corpus: dict[str, BEIRCorpusEntry] = field(default_factory=dict)
    queries: dict[str, BEIRQuery] = field(default_factory=dict)
    qrels: dict[str, list[BEIRQrel]] = field(default_factory=dict)

    def load(self) -> "BEIRLoader":
        corpus_path = self.root / "corpus.jsonl"
        queries_path = self.root / "queries.jsonl"
        qrel_path = self.root / "qrels" / "dev.tsv"

        if corpus_path.exists():
            self.corpus = load_corpus(corpus_path)
        if queries_path.exists():
            self.queries = load_queries(queries_path)
        if qrel_path.exists():
            self.qrels = load_qrels(qrel_path)

        return self

    def to_benchmark_cases(
        self,
        limit: int | None = None,
        min_relevant: int = 1,
    ) -> list[BenchmarkCase]:
        """Convert BEIR dataset to BenchmarkCase objects.

        Args:
            limit: Return only this many cases (useful for smoke/dev runs).
            min_relevant: Skip queries with fewer than this many relevant docs.
        """
        cases: list[BenchmarkCase] = []
        all_qrel_qids = [qid for qid in self.queries if qid in self.qrels]
        if limit:
            all_qrel_qids = all_qrel_qids[:limit]

        for qid in all_qrel_qids:
            if qid not in self.qrels:
                continue
            relevant = [q for q in self.qrels[qid] if q.score > 0]
            if len(relevant) < min_relevant:
                continue

            query_obj = self.queries[qid]
            relevant_doc_ids = [r.doc_id for r in relevant]
            relevant_scores = {r.doc_id: r.score for r in relevant}

            candidate_chunks = []
            for doc_id in relevant_doc_ids:
                if doc_id in self.corpus:
                    entry = self.corpus[doc_id]
                    candidate_chunks.append(
                        {
                            "chunk_id": doc_id,
                            "title": entry.title,
                            "text": entry.text,
                            "url": entry.url,
                            "relevance_score": relevant_scores.get(doc_id, 1),
                        }
                    )

            cases.append(
                BenchmarkCase(
                    id=f"beir-{qid}",
                    query=query_obj.text,
                    query_type="open_book_qa",
                    difficulty="medium",
                    expected_answer=None,
                    ground_truth_chunk_ids=relevant_doc_ids,
                    ground_truth_statements=[],
                    domain="nutrition" if "nutritionfacts" in query_obj.url else "mixed",
                    suite="beir",
                    expected_behavior=None,
                    stale_chunk_ids=[],
                    candidate_chunks=candidate_chunks,
                    variant_outputs={},
                )
            )

        return cases

    def corpus_for_seeding(self) -> list[dict]:
        """Return corpus documents in format suitable for vector store seeding."""
        return [
            {
                "doc_id": doc_id,
                "title": entry.title,
                "text": entry.text,
                "url": entry.url,
            }
            for doc_id, entry in self.corpus.items()
        ]

    def ir_metrics(
        self,
        retrieved_ids: list[str],
        query_id: str,
        k: int = 10,
    ) -> dict[str, float]:
        """Compute IR metrics for a retrieval run against this dataset.

        Args:
            retrieved_ids: Ranked list of doc IDs returned by the retriever.
            query_id: BEIR query ID to look up ground truth for.
            k: Cutoff for P@K, R@K, NDCG@K.

        Returns:
            dict with keys: precision_at_k, recall_at_k, mrr, ndcg_at_k
        """
        if query_id not in self.qrels:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

        relevant = {q.doc_id: q.score for q in self.qrels[query_id] if q.score > 0}
        if not relevant:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}

        retrieved_k = retrieved_ids[:k]
        retrieved_set = set(retrieved_k)

        hits = len(retrieved_set & set(relevant.keys()))
        precision = hits / k if k > 0 else 0.0
        recall = hits / len(relevant) if relevant else 0.0

        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_k, start=1):
            if doc_id in relevant:
                mrr = 1.0 / rank
                break

        relevance_scores = relevant
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_k, start=1):
            if doc_id in relevance_scores:
                rel = float(relevance_scores[doc_id])
                dcg += rel / _log2(i + 1)

        idcg = 0.0
        sorted_relevant = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (doc_id, rel) in enumerate(sorted_relevant[:k], start=1):
            idcg += rel / _log2(i + 1)

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "mrr": mrr,
            "ndcg_at_k": ndcg,
        }


def _log2(x: float) -> float:
    from math import log2 as _log2
    return _log2(x) if x > 0 else 0.0