"""initial schema: docs, chunks, pgvector

Revision ID: 0001
Revises:
Create Date: 2026-04-21 00:00:00

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "docs",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("page_count", sa.Integer, nullable=False),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "doc_id",
            UUID(as_uuid=True),
            sa.ForeignKey("docs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "content_tsv",
            TSVECTOR,
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
        sa.Column("embedding", Vector(1536), nullable=False),
    )

    op.create_index("chunks_doc_id_idx", "chunks", ["doc_id"])
    op.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat "
        "(embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.execute("CREATE INDEX chunks_tsv_idx ON chunks USING GIN (content_tsv)")


def downgrade() -> None:
    op.drop_index("chunks_tsv_idx", table_name="chunks")
    op.drop_index("chunks_embedding_idx", table_name="chunks")
    op.drop_index("chunks_doc_id_idx", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("docs")
