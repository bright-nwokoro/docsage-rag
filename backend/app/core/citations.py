from typing import Any

RAG_SYSTEM_PROMPT = """You answer strictly from the provided context chunks.

Rules:
- Every factual claim in your answer must be supported by a chunk visible in the context.
- In the `citations` field, include one entry per chunk you actually used. Use the exact `source` filename and `page` shown in the chunk header. Provide a `score` between 0.0 and 1.0 reflecting how much you relied on that chunk.
- If the context does not contain the answer, say so explicitly and return an empty `citations` array. Do NOT guess.
- Never invent sources, page numbers, or facts not present in the context.
"""

RAG_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source": {"type": "string"},
                    "page": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["source", "page", "score"],
            },
        },
    },
    "required": ["answer", "citations"],
}


def build_user_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Render the user message: question + numbered context chunks with source/page headers."""
    lines: list[str] = [f"Question: {question}", "", "Context:"]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[chunk {i} | source=\"{c['source']}\" page={c['page']}]")
        lines.append(str(c["content"]).strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def verify_citations(
    citations: list[dict[str, Any]], retrieved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Drop any (source, page) pair in citations that is not present in retrieved chunks."""
    allowed = {(r["source"], r["page"]) for r in retrieved}
    return [c for c in citations if (c.get("source"), c.get("page")) in allowed]
