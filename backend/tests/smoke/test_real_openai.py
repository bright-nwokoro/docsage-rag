"""Opt-in smoke test — hits real OpenAI. Skipped by default.

Run locally before shipping a release:

    pytest backend/tests/smoke -m smoke
"""
import pytest

from app.core.embeddings import EmbeddingsClient
from app.core.openai_client import get_openai_client


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_real_embeddings_are_1536_dim():
    client = EmbeddingsClient(openai_client=get_openai_client(), model="text-embedding-3-small")
    [vec] = await client.embed_batch(["hello world"])
    assert len(vec) == 1536
