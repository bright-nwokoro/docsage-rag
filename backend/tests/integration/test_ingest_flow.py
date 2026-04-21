import hashlib
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.models import Chunk, Doc


def _deterministic_vector(s: str) -> list[float]:
    """Map a string to a stable 1536-dim unit-ish vector for tests."""
    h = hashlib.sha256(s.encode()).digest()
    vec = [((h[i % len(h)] / 255.0) - 0.5) for i in range(1536)]
    return vec


@pytest.mark.asyncio
async def test_ingest_inserts_doc_and_chunks(client, clean_db, monkeypatch):
    # Patch the OpenAI client used by ingest so we don't hit the API.
    async def fake_create(*, model, input):
        resp = MagicMock()
        resp.data = [MagicMock(embedding=_deterministic_vector(t)) for t in input]
        return resp

    fake_openai = MagicMock()
    fake_openai.embeddings = MagicMock()
    fake_openai.embeddings.create = AsyncMock(side_effect=fake_create)

    from app.core import openai_client as oc
    import app.routes.ingest as ingest_route

    oc.get_openai_client.cache_clear()
    monkeypatch.setattr(oc, "get_openai_client", lambda: fake_openai)
    monkeypatch.setattr(ingest_route, "get_openai_client", lambda: fake_openai)

    pdf_path = "tests/fixtures/sample.pdf"
    with open(pdf_path, "rb") as f:
        resp = await client.post(
            "/ingest",
            files={"file": ("sample.pdf", f, "application/pdf")},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "sample.pdf"
    assert body["chunk_count"] >= 1
    assert body["page_count"] >= 1

    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.connect() as conn:
        doc_count = (await conn.execute(select(func.count()).select_from(Doc))).scalar_one()
        chunk_count = (await conn.execute(select(func.count()).select_from(Chunk))).scalar_one()
        assert doc_count == 1
        assert chunk_count == body["chunk_count"]

        # Verify content_tsv is populated automatically by the generated column.
        row = (await conn.execute(text("SELECT content_tsv::text FROM chunks LIMIT 1"))).first()
        assert row is not None and row[0]
    await engine.dispose()
