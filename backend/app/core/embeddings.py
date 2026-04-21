from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class EmbeddingsClient:
    """Thin wrapper over OpenAI embeddings with batching and retry."""

    def __init__(self, openai_client: Any, model: str, batch_size: int = 100):
        self._client = openai_client
        self._model = model
        self._batch_size = batch_size

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    async def _embed_one_batch(self, inputs: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self._model, input=inputs)
        return [d.embedding for d in resp.data]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(await self._embed_one_batch(batch))
        return results
