import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: uuid.UUID
    filename: str
    page_count: int
    chunk_count: int


class DocSummary(BaseModel):
    id: uuid.UUID
    filename: str
    page_count: int
    uploaded_at: datetime


class HistoryTurn(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryTurn] = Field(default_factory=list)
