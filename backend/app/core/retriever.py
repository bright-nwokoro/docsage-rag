from typing import Any, Hashable


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
