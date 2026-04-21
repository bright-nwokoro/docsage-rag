from dataclasses import dataclass
from typing import Any, Hashable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def rrf_fuse(
    ranked_lists: list[list[tuple[Hashable, Any]]],
    k: int,
    top_k: int,
) -> list[tuple[Hashable, Any]]:
    """Reciprocal Rank Fusion.

    Each input list is ranked best-first. Score(item) = Σ 1 / (k + rank_i) across lists where
    item appears (1-indexed rank). Returns items sorted by descending fused score, limited to top_k.
    Item payloads are merged (later lists override earlier keys).
    """
    scores: dict[Hashable, float] = {}
    payloads: dict[Hashable, Any] = {}

    for ranked in ranked_lists:
        for rank, (item_id, payload) in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in payloads:
                payloads[item_id] = payload
            elif isinstance(payload, dict) and isinstance(payloads[item_id], dict):
                payloads[item_id] = {**payloads[item_id], **payload}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(item_id, payloads[item_id]) for item_id, _ in ranked[:top_k]]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    doc_id: str
    source: str
    page: int
    content: str
    score: float


async def _vector_search(
    session: AsyncSession, query_embedding: list[float], k: int
) -> list[tuple[int, dict]]:
    sql = text(
        """
        SELECT c.id, c.doc_id, d.filename, c.page_number, c.content,
               (c.embedding <=> (:q_emb)::vector) AS distance
        FROM chunks c
        JOIN docs d ON d.id = c.doc_id
        ORDER BY c.embedding <=> (:q_emb)::vector
        LIMIT :k
        """
    )
    result = await session.execute(sql, {"q_emb": str(query_embedding), "k": k})
    rows = result.all()
    return [
        (
            r.id,
            {
                "doc_id": str(r.doc_id),
                "source": r.filename,
                "page": r.page_number,
                "content": r.content,
                "distance": float(r.distance),
            },
        )
        for r in rows
    ]


async def _keyword_search(
    session: AsyncSession, query: str, k: int
) -> list[tuple[int, dict]]:
    sql = text(
        """
        SELECT c.id, c.doc_id, d.filename, c.page_number, c.content,
               ts_rank(c.content_tsv, plainto_tsquery('english', :q)) AS rank
        FROM chunks c
        JOIN docs d ON d.id = c.doc_id
        WHERE c.content_tsv @@ plainto_tsquery('english', :q)
        ORDER BY rank DESC
        LIMIT :k
        """
    )
    result = await session.execute(sql, {"q": query, "k": k})
    rows = result.all()
    return [
        (
            r.id,
            {
                "doc_id": str(r.doc_id),
                "source": r.filename,
                "page": r.page_number,
                "content": r.content,
                "rank": float(r.rank),
            },
        )
        for r in rows
    ]


async def retrieve(
    session: AsyncSession,
    query: str,
    query_embedding: list[float],
    candidate_k: int,
    top_k: int,
    rrf_k: int,
) -> list[RetrievedChunk]:
    # Run sequentially: SQLAlchemy async sessions do not support concurrent
    # operations on the same connection (asyncio.gather would raise an InvalidRequestError).
    vec_hits = await _vector_search(session, query_embedding, candidate_k)
    kw_hits = await _keyword_search(session, query, candidate_k)

    fused = rrf_fuse([vec_hits, kw_hits], k=rrf_k, top_k=top_k)
    out: list[RetrievedChunk] = []
    score_map = dict(_fused_with_scores([vec_hits, kw_hits], rrf_k))
    max_score = max(score_map.values(), default=1.0)

    for item_id, payload in fused:
        raw = score_map.get(item_id, 0.0)
        out.append(
            RetrievedChunk(
                chunk_id=int(item_id),
                doc_id=payload["doc_id"],
                source=payload["source"],
                page=payload["page"],
                content=payload["content"],
                score=raw / max_score if max_score > 0 else 0.0,
            )
        )
    return out


def _fused_with_scores(
    ranked_lists: list[list[tuple[int, dict]]], k: int
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return list(scores.items())
