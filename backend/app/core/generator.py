from collections.abc import AsyncIterator
from typing import Any

import json_repair

from app.core.citations import (
    RAG_ANSWER_SCHEMA,
    RAG_SYSTEM_PROMPT,
    build_user_prompt,
    verify_citations,
)

MAX_HISTORY_TURNS = 8


def _truncate_history(
    history: list[dict[str, str]], max_turns: int = MAX_HISTORY_TURNS
) -> list[dict[str, str]]:
    """Keep the last N turns (each turn = one user msg + one assistant msg = 2 items)."""
    return history[-(max_turns * 2) :]


async def generate_stream(
    openai_client: Any,
    model: str,
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]],
) -> AsyncIterator[dict[str, Any]]:
    """Stream a RAG answer from OpenAI. Yields SSE-shaped dicts:
      {"event": "answer_delta", "data": {"text": str}}
      {"event": "citation",     "data": {source, page, score}}
      {"event": "done",         "data": {"verified_citations": [...]}}
      {"event": "error",        "data": {"message": str}}
    """
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    messages.extend(_truncate_history(history))
    messages.append({"role": "user", "content": build_user_prompt(question, chunks)})

    try:
        stream = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_answer",
                    "strict": True,
                    "schema": RAG_ANSWER_SCHEMA,
                },
            },
        )
    except Exception as e:
        yield {"event": "error", "data": {"message": f"upstream: {type(e).__name__}: {e}"}}
        return

    buf = ""
    emitted_answer_len = 0
    emitted_citation_keys: set[tuple[str, int]] = set()

    try:
        async for event in stream:
            delta = event.choices[0].delta.content or ""
            if not delta:
                continue
            buf += delta

            try:
                parsed = json_repair.loads(buf)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue

            # 1) Emit new answer characters.
            answer_so_far = parsed.get("answer", "")
            if isinstance(answer_so_far, str) and len(answer_so_far) > emitted_answer_len:
                new_text = answer_so_far[emitted_answer_len:]
                emitted_answer_len = len(answer_so_far)
                yield {"event": "answer_delta", "data": {"text": new_text}}

            # 2) Emit any newly-complete citation entries.
            cits = parsed.get("citations", [])
            if isinstance(cits, list):
                for c in cits:
                    if not isinstance(c, dict):
                        continue
                    src = c.get("source")
                    page = c.get("page")
                    score = c.get("score")
                    # Guard: require all three fields with correct types and non-empty values.
                    if not src or page is None:
                        continue
                    if not (
                        isinstance(src, str)
                        and isinstance(page, int)
                        and isinstance(score, (int, float))
                    ):
                        continue
                    key = (src, page)
                    if key in emitted_citation_keys:
                        continue
                    emitted_citation_keys.add(key)
                    yield {
                        "event": "citation",
                        "data": {"source": src, "page": page, "score": float(score)},
                    }
    except Exception as e:
        yield {"event": "error", "data": {"message": f"stream: {type(e).__name__}: {e}"}}
        return

    # Final verification pass on the fully accumulated buffer.
    try:
        final = json_repair.loads(buf) if buf else {}
    except Exception:
        final = {}
    final_citations = final.get("citations", []) if isinstance(final, dict) else []
    retrieved_pairs = [{"source": c["source"], "page": c["page"]} for c in chunks]
    verified = verify_citations(
        final_citations if isinstance(final_citations, list) else [], retrieved_pairs
    )

    yield {"event": "done", "data": {"verified_citations": verified}}
