import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.generator import generate_stream


class FakeStream:
    """Simulates OpenAI streaming by yielding chat.completions deltas."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=c))])


@pytest.mark.asyncio
async def test_generator_emits_answer_deltas_then_done():
    """A simple happy path: the model emits a valid JSON with answer + one citation."""
    full = json.dumps(
        {
            "answer": "Hello world",
            "citations": [{"source": "a.pdf", "page": 1, "score": 0.9}],
        }
    )
    # Simulate streaming by slicing into small pieces
    pieces = [full[i : i + 5] for i in range(0, len(full), 5)]

    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=FakeStream(pieces))

    retrieved = [{"source": "a.pdf", "page": 1, "content": "anything"}]

    events = []
    async for evt in generate_stream(
        openai_client=fake_client,
        model="test-model",
        question="Q?",
        chunks=retrieved,
        history=[],
    ):
        events.append(evt)

    event_types = [e["event"] for e in events]
    assert "answer_delta" in event_types
    assert event_types.count("citation") >= 1
    assert event_types[-1] == "done"

    # Accumulated answer_delta texts should concatenate to "Hello world"
    acc = "".join(e["data"]["text"] for e in events if e["event"] == "answer_delta")
    assert acc == "Hello world"

    # Final done event carries verified citations
    done = events[-1]
    assert done["data"]["verified_citations"] == [
        {"source": "a.pdf", "page": 1, "score": 0.9}
    ]


@pytest.mark.asyncio
async def test_generator_drops_invalid_citations_in_done_event():
    full = json.dumps(
        {
            "answer": "x",
            "citations": [
                {"source": "a.pdf", "page": 1, "score": 0.9},
                {"source": "ghost.pdf", "page": 99, "score": 0.5},
            ],
        }
    )
    pieces = [full]

    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=FakeStream(pieces))

    retrieved = [{"source": "a.pdf", "page": 1, "content": "."}]

    events = []
    async for evt in generate_stream(
        openai_client=fake_client,
        model="test-model",
        question="q",
        chunks=retrieved,
        history=[],
    ):
        events.append(evt)

    done = next(e for e in events if e["event"] == "done")
    sources = {c["source"] for c in done["data"]["verified_citations"]}
    assert sources == {"a.pdf"}
