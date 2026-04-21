import hashlib
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import NullPool, text
from sqlalchemy.ext.asyncio import create_async_engine


def _deterministic_vector(s: str) -> list[float]:
    h = hashlib.sha256(s.encode()).digest()
    return [((h[i % len(h)] / 255.0) - 0.5) for i in range(1536)]


class _FakeStream:
    def __init__(self, pieces):
        self._pieces = pieces

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._pieces:
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=c))])


@pytest.mark.asyncio
async def test_query_streams_answer_and_citations(client, clean_db, monkeypatch):
    # Seed one doc + one chunk directly in the DB.
    # NullPool prevents asyncpg connections from being held across event loop
    # boundaries, which causes "Future attached to a different loop" errors
    # when multiple integration tests run in the same pytest session.
    engine = create_async_engine(os.environ["DATABASE_URL"], poolclass=NullPool)
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "INSERT INTO docs (filename, page_count) VALUES (:f, :p) RETURNING id"
            ),
            {"f": "seed.pdf", "p": 1},
        )
        doc_id = result.scalar_one()
        vec = _deterministic_vector("pgvector enables similarity search in postgres")
        await conn.execute(
            text(
                "INSERT INTO chunks (doc_id, page_number, chunk_index, content, embedding) "
                "VALUES (:d, 1, 0, :c, (:e)::vector)"
            ),
            {
                "d": doc_id,
                "c": "Pgvector enables similarity search in Postgres using cosine distance.",
                "e": str(vec),
            },
        )
    await engine.dispose()

    # Mock OpenAI: embeddings return deterministic vectors, chat returns a streamed JSON.
    async def fake_embed_create(*, model, input):
        resp = MagicMock()
        resp.data = [MagicMock(embedding=_deterministic_vector(t)) for t in input]
        return resp

    full = json.dumps(
        {
            "answer": "Pgvector enables similarity search.",
            "citations": [{"source": "seed.pdf", "page": 1, "score": 0.9}],
        }
    )
    pieces = [full[i : i + 10] for i in range(0, len(full), 10)]

    async def fake_chat_create(**_):
        return _FakeStream(pieces)

    fake_openai = MagicMock()
    fake_openai.embeddings = MagicMock()
    fake_openai.embeddings.create = AsyncMock(side_effect=fake_embed_create)
    fake_openai.chat = MagicMock()
    fake_openai.chat.completions = MagicMock()
    fake_openai.chat.completions.create = AsyncMock(side_effect=fake_chat_create)

    from app.core import openai_client as oc
    import app.routes.query as query_route

    oc.get_openai_client.cache_clear()
    monkeypatch.setattr(oc, "get_openai_client", lambda: fake_openai)
    monkeypatch.setattr(query_route, "get_openai_client", lambda: fake_openai)

    resp = await client.post(
        "/query",
        json={"question": "What does pgvector enable?", "history": []},
    )
    assert resp.status_code == 200
    text_out = resp.text

    # sse-starlette serialises events like:  event: answer_delta\ndata: {...}\n\n
    assert "event: answer_delta" in text_out
    assert "event: citation" in text_out
    assert "event: done" in text_out
    assert "seed.pdf" in text_out
