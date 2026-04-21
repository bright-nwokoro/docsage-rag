import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.chunker import chunk_pdf
from app.core.embeddings import EmbeddingsClient
from app.core.openai_client import get_openai_client
from app.db import get_session
from app.models import Chunk, Doc
from app.schemas import IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    settings = get_settings()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only .pdf files are supported")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        chunks, page_count = chunk_pdf(
            tmp_path,
            max_tokens=settings.MAX_CHUNK_TOKENS,
            overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
        )
        if not chunks:
            raise HTTPException(status_code=422, detail="no text extracted from PDF")

        embeddings_client = EmbeddingsClient(
            openai_client=get_openai_client(),
            model=settings.OPENAI_EMBED_MODEL,
        )
        vectors = await embeddings_client.embed_batch([c.content for c in chunks])

        doc = Doc(filename=file.filename, page_count=page_count)
        session.add(doc)
        await session.flush()  # populate doc.id

        session.add_all(
            [
                Chunk(
                    doc_id=doc.id,
                    page_number=c.page_number,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    embedding=vec,
                )
                for c, vec in zip(chunks, vectors, strict=True)
            ]
        )
        await session.commit()

        return IngestResponse(
            doc_id=doc.id,
            filename=file.filename,
            page_count=page_count,
            chunk_count=len(chunks),
        )
    finally:
        tmp_path.unlink(missing_ok=True)
