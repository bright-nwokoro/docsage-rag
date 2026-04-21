import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.core.embeddings import EmbeddingsClient
from app.core.generator import generate_stream
from app.core.openai_client import get_openai_client
from app.core.retriever import retrieve
from app.db import get_session
from app.schemas import QueryRequest

router = APIRouter()


@router.post("/query")
async def query(
    body: QueryRequest,
    session: AsyncSession = Depends(get_session),
) -> EventSourceResponse:
    settings = get_settings()
    openai_client = get_openai_client()

    # 1. Embed the question.
    embeddings = EmbeddingsClient(
        openai_client=openai_client, model=settings.OPENAI_EMBED_MODEL
    )
    [q_embedding] = await embeddings.embed_batch([body.question])

    # 2. Retrieve.
    retrieved = await retrieve(
        session=session,
        query=body.question,
        query_embedding=q_embedding,
        candidate_k=settings.CANDIDATE_K,
        top_k=settings.TOP_K,
        rrf_k=settings.RRF_K,
    )

    if not retrieved:
        async def empty_stream() -> AsyncIterator[dict]:
            yield {"event": "answer_delta", "data": json.dumps({
                "text": "I couldn't find anything relevant in the uploaded documents."
            })}
            yield {"event": "done", "data": json.dumps({"verified_citations": []})}
        return EventSourceResponse(empty_stream())

    # 3. Build chunks payload for the generator.
    chunks_payload = [
        {"source": r.source, "page": r.page, "content": r.content, "score": r.score}
        for r in retrieved
    ]

    # 4. Stream generation.
    history = [{"role": h.role, "content": h.content} for h in body.history]

    async def event_stream() -> AsyncIterator[dict]:
        async for event in generate_stream(
            openai_client=openai_client,
            model=settings.OPENAI_CHAT_MODEL,
            question=body.question,
            chunks=chunks_payload,
            history=history,
        ):
            yield {"event": event["event"], "data": json.dumps(event["data"])}

    return EventSourceResponse(event_stream())
