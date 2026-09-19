"""
Semantic memory: cross-customer knowledge (policies, rules, product catalog).

Backed by ChromaDB with sentence-transformers embeddings.
Seeded with a small default policy corpus on first init.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config.settings import settings
from src.utils.logger import get_logger
from src.utils.tracing import span

log = get_logger(__name__)

_COLLECTION = "policies"

_SEED_DOCS: list[tuple[str, str, dict[str, Any]]] = [
    ("POL-RET-001",
     "Proactive retention outreach is appropriate when a mid or high value "
     "customer shows a sustained login-frequency decline (>25%) over 7+ days "
     "combined with negative support sentiment, and no compliance hold applies.",
     {"category": "retention", "policy_id": "POL-RET-001"}),

    ("POL-RET-002",
     "Retention gestures for mid-tier customers should be capped at the "
     "equivalent of one month of fees. High-tier customers may receive up to "
     "three months of fees, subject to HITL approval.",
     {"category": "retention", "policy_id": "POL-RET-002"}),

    ("POL-HITL-001",
     "Any proposed action with a cost above $200, or any action involving a "
     "credit line change, must be routed to HITL for approval.",
     {"category": "hitl", "policy_id": "POL-HITL-001"}),

    ("POL-HITL-002",
     "Any action based on a sensitive inferred life event (medical hardship, "
     "financial distress, marital change, or relocation) must be routed to "
     "HITL regardless of cost.",
     {"category": "hitl", "policy_id": "POL-HITL-002"}),

    ("POL-COMP-001",
     "If a customer reports a legal threat, an account takeover, or any "
     "self-harm signal, all autonomous outbound communication is halted and "
     "the case is routed to the appropriate escalation queue.",
     {"category": "compliance", "policy_id": "POL-COMP-001"}),

    ("POL-OFFER-001",
     "A personalized product offer (loan, credit line, or savings product) "
     "requires an inferred life event with confidence >= 0.7 AND a compliance "
     "clearance. It may not be sent autonomously.",
     {"category": "offer", "policy_id": "POL-OFFER-001"}),

    ("POL-SUPPORT-001",
     "If a customer explicitly requests a payment plan or forbearance, route "
     "to support_intervention with subtype 'payment_arrangement'. Do not "
     "upsell in the same interaction.",
     {"category": "support", "policy_id": "POL-SUPPORT-001"}),

    ("POL-FRAUD-001",
     "Signals consistent with account takeover or unauthorized access place "
     "a compliance/fraud hold on the account pending human review.",
     {"category": "fraud", "policy_id": "POL-FRAUD-001"}),
]


class SemanticMemory:
    def __init__(self, persist_dir: Path | None = None) -> None:
        path = str(persist_dir or settings.chroma_db_path)
        self._client = chromadb.PersistentClient(path=path)
        try:
            self._embed = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model,
            )
        except Exception as e:
            log.warning("semantic.embedding_fallback", error=str(e))
            self._embed = embedding_functions.DefaultEmbeddingFunction()

        self._col = self._client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=self._embed,
            metadata={"hnsw:space": "cosine"},
        )
        self._seed_if_empty()

    def _seed_if_empty(self) -> None:
        if self._col.count() > 0:
            log.info("semantic.loaded", count=self._col.count())
            return
        ids = [d[0] for d in _SEED_DOCS]
        docs = [d[1] for d in _SEED_DOCS]
        metas = [d[2] for d in _SEED_DOCS]
        self._col.add(ids=ids, documents=docs, metadatas=metas)
        log.info("semantic.seeded", count=len(ids))

    def add(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None
            ) -> None:
        """Incremental add — no full re-index needed."""
        self._col.add(ids=[doc_id], documents=[text],
                      metadatas=[metadata or {}])

    def query(self, text: str, k: int = 3,
              where: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with span("semantic.query", k=k) as s:
            res = self._col.query(
                query_texts=[text],
                n_results=k,
                where=where,
            )
            out: list[dict[str, Any]] = []
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            for i, _id in enumerate(ids):
                out.append({
                    "id": _id,
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i] if i < len(dists) else None,
                })
            s.attributes["returned"] = len(out)
            return out

    def count(self) -> int:
        return self._col.count()