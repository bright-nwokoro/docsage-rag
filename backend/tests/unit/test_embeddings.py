from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.embeddings import EmbeddingsClient


@pytest.mark.asyncio
async def test_embed_batch_calls_openai_once_for_small_input():
    fake_response = MagicMock()
    fake_response.data = [
        MagicMock(embedding=[0.1] * 1536),
        MagicMock(embedding=[0.2] * 1536),
    ]
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    client = EmbeddingsClient(openai_client=fake_client, model="test-model", batch_size=100)
    vecs = await client.embed_batch(["alpha", "beta"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 1536
    fake_client.embeddings.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_embed_batch_splits_across_multiple_batches():
    fake_client = MagicMock()

    async def fake_create(*, model, input):
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[float(i)] * 1536) for i in range(len(input))]
        return resp

    fake_client.embeddings = MagicMock()
    fake_client.embeddings.create = AsyncMock(side_effect=fake_create)

    client = EmbeddingsClient(openai_client=fake_client, model="test-model", batch_size=2)
    vecs = await client.embed_batch(["a", "b", "c", "d", "e"])
    assert len(vecs) == 5
    # ceil(5/2) = 3 calls
    assert fake_client.embeddings.create.await_count == 3


@pytest.mark.asyncio
async def test_embed_batch_empty_returns_empty():
    fake_client = MagicMock()
    fake_client.embeddings = MagicMock()
    fake_client.embeddings.create = AsyncMock()
    client = EmbeddingsClient(openai_client=fake_client, model="test-model", batch_size=100)
    assert await client.embed_batch([]) == []
    fake_client.embeddings.create.assert_not_awaited()
