import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Doc
from app.schemas import DocSummary

router = APIRouter()


@router.get("", response_model=list[DocSummary])
async def list_docs(session: AsyncSession = Depends(get_session)) -> list[DocSummary]:
    result = await session.execute(select(Doc).order_by(Doc.uploaded_at.desc()))
    rows = result.scalars().all()
    return [
        DocSummary(
            id=d.id, filename=d.filename, page_count=d.page_count, uploaded_at=d.uploaded_at
        )
        for d in rows
    ]


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_doc(
    doc_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> None:
    existing = await session.get(Doc, doc_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="doc not found")
    await session.execute(delete(Doc).where(Doc.id == doc_id))
    await session.commit()
