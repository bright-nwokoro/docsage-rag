# DocSage RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build DocSage end-to-end locally — a FastAPI + Next.js 14 RAG chatbot over PDFs with hybrid retrieval, forced citations, and partial-JSON streaming — runnable with `make up && make migrate && make dev`.

**Architecture:** Docker Compose Postgres with pgvector, FastAPI async backend with SQLAlchemy 2.0 + asyncpg, Next.js 14 App Router frontend with SSE streaming. Every answer is forced through a strict JSON schema; citations are post-hoc verified against the retrieved chunk set. Hybrid retrieval fuses pgvector cosine with Postgres full-text via Reciprocal Rank Fusion.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2.0 (async), asyncpg, Alembic, pgvector 0.7, OpenAI SDK 1.x (`text-embedding-3-small` + `gpt-4o-mini`), tiktoken, unstructured, json-repair, pytest. Node 20, Next.js 14, React 18, TypeScript, Tailwind, shadcn/ui, pnpm, `@microsoft/fetch-event-source`, Vitest.

**Spec:** [`docs/superpowers/specs/2026-04-21-docsage-rag-design.md`](../specs/2026-04-21-docsage-rag-design.md)

---

## Phase A — Bootstrap & infrastructure

### Task 1: `.gitignore`

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/

# Node
node_modules/
.next/
dist/
.turbo/
*.tsbuildinfo

# Env
.env
.env.local
.env.*.local

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Docker volumes
pgdata/

# Uploads (local dev)
backend/uploads/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore for Python, Node, env, and IDE files"
```

---

### Task 2: Python version pin

**Files:**
- Create: `.python-version`

- [ ] **Step 1: Write `.python-version`**

```
3.11
```

- [ ] **Step 2: Commit**

```bash
git add .python-version
git commit -m "chore: pin Python to 3.11"
```

---

### Task 3: Docker Compose for Postgres + pgvector

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Write `docker-compose.yml`**

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    container_name: docsage-db
    environment:
      POSTGRES_USER: docsage
      POSTGRES_PASSWORD: docsage
      POSTGRES_DB: docsage
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U docsage -d docsage"]
      interval: 2s
      timeout: 2s
      retries: 20

volumes:
  pgdata:
```

- [ ] **Step 2: Start and verify**

Run:
```bash
docker compose up -d
docker compose ps
```

Expected: service `db` shows `Up` and `(healthy)` within 10–20s.

- [ ] **Step 3: Verify pgvector extension is available**

Run:
```bash
docker compose exec db psql -U docsage -d docsage -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT extversion FROM pg_extension WHERE extname='vector';"
```

Expected: one row returned with `extversion` like `0.7.x`.

- [ ] **Step 4: Tear down (clean state for later tasks)**

Run:
```bash
docker compose down
```

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: add docker-compose with pgvector/pg16"
```

---

### Task 4: Root `Makefile`

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write `Makefile`**

```makefile
.PHONY: help up down migrate dev test test-unit test-integration eval seed clean fmt lint install

help:
	@echo "DocSage Makefile targets:"
	@echo "  make install           Install backend + frontend deps"
	@echo "  make up                Start Postgres (Docker)"
	@echo "  make down              Stop Postgres (keeps volume)"
	@echo "  make migrate           Run Alembic migrations"
	@echo "  make dev               Run backend + frontend concurrently"
	@echo "  make test              Run all tests"
	@echo "  make test-unit         Run backend unit tests only"
	@echo "  make test-integration  Run backend integration tests"
	@echo "  make eval              Run golden-set eval (scaffold)"
	@echo "  make seed              Placeholder (deferred)"
	@echo "  make fmt               Format code"
	@echo "  make lint              Lint code"
	@echo "  make clean             Stop Postgres and delete volume"

install:
	cd backend && pip install -r requirements.txt -r requirements-dev.txt
	cd frontend && pnpm install

up:
	docker compose up -d --wait

down:
	docker compose down

migrate:
	cd backend && alembic upgrade head

dev:
	@echo "Starting backend (8000) and frontend (3000)..."
	@trap 'kill 0' INT; \
	(cd backend && uvicorn app.main:app --reload --port 8000) & \
	(cd frontend && pnpm dev) & \
	wait

test: test-unit test-integration
	cd frontend && pnpm test --run

test-unit:
	cd backend && pytest tests/unit -v

test-integration:
	cd backend && pytest tests/integration -v

eval:
	cd backend && python tests/eval/run_eval.py

seed:
	@echo "Seed mode deferred — see docs/superpowers/specs/2026-04-21-docsage-rag-design.md"

fmt:
	cd backend && ruff format app tests
	cd frontend && pnpm format

lint:
	cd backend && ruff check app tests
	cd frontend && pnpm lint

clean:
	docker compose down -v
```

- [ ] **Step 2: Verify help target**

Run:
```bash
make help
```

Expected: prints target list.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: add root Makefile for common dev tasks"
```

---

## Phase B — Backend skeleton

### Task 5: Backend dependencies

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/requirements-dev.txt`
- Create: `backend/pyproject.toml`

- [ ] **Step 1: Write `backend/requirements.txt`**

```
fastapi==0.110.0
uvicorn[standard]==0.29.0
pydantic==2.6.4
pydantic-settings==2.2.1
sqlalchemy[asyncio]==2.0.29
asyncpg==0.29.0
alembic==1.13.1
pgvector==0.2.5
openai==1.17.0
tiktoken==0.6.0
unstructured[pdf]==0.13.2
pypdf==4.2.0
tenacity==8.2.3
python-multipart==0.0.9
json-repair==0.25.2
sse-starlette==2.1.0
```

- [ ] **Step 2: Write `backend/requirements-dev.txt`**

```
-r requirements.txt
pytest==8.1.1
pytest-asyncio==0.23.6
httpx==0.27.0
ruff==0.4.1
```

- [ ] **Step 3: Write `backend/pyproject.toml`**

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]
ignore = ["E501"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "smoke: opt-in tests that hit real external APIs (skipped by default)",
]
addopts = "-m 'not smoke'"
testpaths = ["tests"]
```

- [ ] **Step 4: Create virtualenv and install**

Run:
```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Expected: all packages install without errors.

- [ ] **Step 5: Verify ruff works**

Run:
```bash
cd backend && ruff --version
```

Expected: prints version.

- [ ] **Step 6: Commit**

```bash
git add backend/requirements.txt backend/requirements-dev.txt backend/pyproject.toml
git commit -m "chore(backend): add requirements and ruff/pytest config"
```

---

### Task 6: Backend env example

**Files:**
- Create: `backend/.env.example`

- [ ] **Step 1: Write `backend/.env.example`**

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

# Database
DATABASE_URL=postgresql+asyncpg://docsage:docsage@localhost:5432/docsage

# Retrieval
MAX_CHUNK_TOKENS=512
CHUNK_OVERLAP_TOKENS=64
TOP_K=5
CANDIDATE_K=20
RRF_K=60

# CORS (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000
```

- [ ] **Step 2: Copy to `.env` for local dev**

Run:
```bash
cp backend/.env.example backend/.env
```

Then edit `backend/.env` and replace `sk-...` with a real key.

- [ ] **Step 3: Commit `.env.example`**

```bash
git add backend/.env.example
git commit -m "chore(backend): add .env.example with OpenAI and DB config"
```

---

### Task 7: Config module

**Files:**
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`

- [ ] **Step 1: Create empty `backend/app/__init__.py`**

```python
```

- [ ] **Step 2: Write `backend/app/config.py`**

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

    DATABASE_URL: str

    MAX_CHUNK_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 64
    TOP_K: int = 5
    CANDIDATE_K: int = 20
    RRF_K: int = 60

    ALLOWED_ORIGINS: str = "http://localhost:3000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 3: Verify config loads**

Run:
```bash
cd backend && python -c "from app.config import get_settings; s = get_settings(); print('OK', s.OPENAI_CHAT_MODEL)"
```

Expected: `OK gpt-4o-mini`.

- [ ] **Step 4: Commit**

```bash
git add backend/app/__init__.py backend/app/config.py
git commit -m "feat(backend): add pydantic-settings config module"
```

---

### Task 8: Database session

**Files:**
- Create: `backend/app/db.py`

- [ ] **Step 1: Write `backend/app/db.py`**

```python
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

_settings = get_settings()

engine = create_async_engine(
    _settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/db.py
git commit -m "feat(backend): add async SQLAlchemy engine and session factory"
```

---

### Task 9: SQLAlchemy models

**Files:**
- Create: `backend/app/models.py`

- [ ] **Step 1: Write `backend/app/models.py`**

```python
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Computed, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Doc(Base):
    __tablename__ = "docs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="doc", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("docs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=False,
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)

    doc: Mapped[Doc] = relationship(back_populates="chunks")
```

Note: `content_tsv` is a generated column in Postgres. `Computed(..., persisted=True)` tells SQLAlchemy to omit it from INSERT statements (Postgres rejects INSERTs that specify values for `GENERATED ALWAYS AS ... STORED` columns).

- [ ] **Step 2: Commit**

```bash
git add backend/app/models.py
git commit -m "feat(backend): add Doc and Chunk SQLAlchemy models"
```

---

### Task 10: Alembic init + initial migration

**Files:**
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/script.py.mako`
- Create: `backend/alembic/versions/0001_initial.py`

- [ ] **Step 1: Write `backend/alembic.ini`**

```ini
[alembic]
script_location = alembic
prepend_sys_path = .

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

- [ ] **Step 2: Write `backend/alembic/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 3: Write `backend/alembic/env.py`**

```python
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

from app.config import get_settings
from app.models import Base

config = context.config
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


run_migrations_online()
```

- [ ] **Step 4: Write `backend/alembic/versions/0001_initial.py`**

```python
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
```

- [ ] **Step 5: Bring DB up and run migration**

Run:
```bash
docker compose up -d --wait
cd backend && alembic upgrade head
```

Expected: `Running upgrade  -> 0001, initial schema: docs, chunks, pgvector`.

- [ ] **Step 6: Verify schema**

Run:
```bash
docker compose exec db psql -U docsage -d docsage -c "\d+ chunks"
```

Expected: table with `content_tsv` marked as `GENERATED ALWAYS AS ... STORED`, `embedding` as `vector(1536)`.

- [ ] **Step 7: Commit**

```bash
git add backend/alembic.ini backend/alembic/env.py backend/alembic/script.py.mako backend/alembic/versions/0001_initial.py
git commit -m "feat(backend): add Alembic config and initial schema migration"
```

---

## Phase C — Core pure modules (TDD)

### Task 11: Chunker — types and tests

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/unit/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/unit/test_chunker.py`

- [ ] **Step 1: Create empty package init files**

```bash
mkdir -p backend/tests/unit backend/tests/integration backend/tests/smoke backend/tests/eval backend/tests/fixtures
touch backend/tests/__init__.py backend/tests/unit/__init__.py backend/tests/integration/__init__.py backend/tests/smoke/__init__.py
```

- [ ] **Step 2: Write `backend/tests/conftest.py`**

```python
import os
import sys
from pathlib import Path

# Ensure backend/ is on sys.path so `from app.X import ...` works.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Provide a harmless OPENAI_API_KEY so config loads in unit tests.
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://docsage:docsage@localhost:5432/docsage",
)
```

- [ ] **Step 3: Write failing tests in `backend/tests/unit/test_chunker.py`**

```python
from app.core.chunker import Chunk, chunk_text


def test_single_short_page_produces_one_chunk():
    pages = [(1, "Hello world. This is a short sentence.")]
    chunks = chunk_text(pages, max_tokens=512, overlap_tokens=64)
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].chunk_index == 0
    assert "Hello world" in chunks[0].content


def test_long_page_splits_into_multiple_chunks():
    sentence = "This is a medium-length sentence used for testing. "
    long_page_text = sentence * 80  # ~600+ tokens
    pages = [(1, long_page_text)]
    chunks = chunk_text(pages, max_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 2
    # Indexes are contiguous starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_max_token_budget():
    sentence = "Alpha beta gamma delta epsilon zeta eta theta. "
    pages = [(1, sentence * 50)]
    chunks = chunk_text(pages, max_tokens=50, overlap_tokens=10)
    for c in chunks:
        assert c.token_count <= 50, f"chunk {c.chunk_index} exceeded budget: {c.token_count}"


def test_chunks_have_overlap_between_consecutive():
    sentence_a = "Alpha beta gamma. "
    sentence_b = "Delta epsilon zeta. "
    sentence_c = "Theta iota kappa. "
    text = (sentence_a + sentence_b + sentence_c) * 30
    pages = [(1, text)]
    chunks = chunk_text(pages, max_tokens=40, overlap_tokens=15)
    assert len(chunks) >= 2
    # Overlap: the start of chunk i+1 should share at least one token with the end of chunk i.
    for i in range(len(chunks) - 1):
        tail = chunks[i].content[-80:]
        head = chunks[i + 1].content[:80]
        common = set(tail.split()) & set(head.split())
        assert common, f"no overlap between chunk {i} and {i + 1}"


def test_multi_page_preserves_page_numbers():
    pages = [
        (1, "Page one content. Another sentence on page one."),
        (2, "Page two content. Another sentence on page two."),
        (3, "Page three content. Another sentence on page three."),
    ]
    chunks = chunk_text(pages, max_tokens=512, overlap_tokens=64)
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2, 3}


def test_empty_input_produces_no_chunks():
    assert chunk_text([], max_tokens=512, overlap_tokens=64) == []
    assert chunk_text([(1, "")], max_tokens=512, overlap_tokens=64) == []


def test_chunk_dataclass_fields():
    c = Chunk(page_number=1, chunk_index=0, content="x", token_count=1)
    assert c.page_number == 1
    assert c.chunk_index == 0
    assert c.content == "x"
    assert c.token_count == 1
```

- [ ] **Step 4: Run tests — verify they fail (module not yet created)**

Run:
```bash
cd backend && pytest tests/unit/test_chunker.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.core.chunker'`.

- [ ] **Step 5: Commit failing tests**

```bash
git add backend/tests/
git commit -m "test(backend): add failing chunker tests"
```

---

### Task 12: Chunker — implementation

**Files:**
- Create: `backend/app/core/__init__.py`
- Create: `backend/app/core/chunker.py`

- [ ] **Step 1: Create empty `backend/app/core/__init__.py`**

```python
```

- [ ] **Step 2: Write `backend/app/core/chunker.py`**

```python
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

_ENCODER = tiktoken.get_encoding("cl100k_base")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass(frozen=True)
class Chunk:
    page_number: int
    chunk_index: int
    content: str
    token_count: int


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _token_count(text: str) -> int:
    return len(_ENCODER.encode(text))


def _tail_tokens(text: str, n: int) -> str:
    if n <= 0:
        return ""
    tokens = _ENCODER.encode(text)
    if len(tokens) <= n:
        return text
    return _ENCODER.decode(tokens[-n:])


def chunk_text(
    pages: list[tuple[int, str]],
    max_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Pack sentences into token-budgeted chunks, preserving page boundaries.

    Each output chunk inherits the page number of the sentence that opened it.
    A chunk closes when adding the next sentence would exceed max_tokens;
    the next chunk is seeded with the tail of the previous at overlap_tokens.
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    carry_text = ""
    carry_tokens = 0
    current_page = None
    current_sentences: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal chunk_index, carry_text, carry_tokens, current_sentences, current_tokens, current_page
        if not current_sentences:
            return
        content = " ".join(current_sentences).strip()
        if not content:
            current_sentences = []
            current_tokens = 0
            return
        chunks.append(
            Chunk(
                page_number=current_page if current_page is not None else 1,
                chunk_index=chunk_index,
                content=content,
                token_count=_token_count(content),
            )
        )
        chunk_index += 1
        # Seed the next chunk with the tail of this one.
        carry_text = _tail_tokens(content, overlap_tokens)
        carry_tokens = _token_count(carry_text) if carry_text else 0
        current_sentences = []
        current_tokens = 0
        current_page = None

    for page_num, page_text in pages:
        sentences = _split_sentences(page_text)
        for sent in sentences:
            sent_tokens = _token_count(sent)
            # If a single sentence alone exceeds the budget, emit it as its own chunk.
            if sent_tokens > max_tokens:
                flush()  # flush anything in progress first
                chunks.append(
                    Chunk(
                        page_number=page_num,
                        chunk_index=chunk_index,
                        content=sent,
                        token_count=sent_tokens,
                    )
                )
                chunk_index += 1
                carry_text = _tail_tokens(sent, overlap_tokens)
                carry_tokens = _token_count(carry_text) if carry_text else 0
                continue

            # Starting a new chunk? Open with carry_text for overlap.
            if not current_sentences:
                current_page = page_num
                if carry_text:
                    current_sentences.append(carry_text)
                    current_tokens = carry_tokens
                    carry_text = ""
                    carry_tokens = 0

            # Does adding this sentence blow the budget? Flush and retry.
            if current_tokens + sent_tokens > max_tokens:
                flush()
                current_page = page_num
                if carry_text:
                    current_sentences.append(carry_text)
                    current_tokens = carry_tokens
                    carry_text = ""
                    carry_tokens = 0

            current_sentences.append(sent)
            current_tokens += sent_tokens

    flush()
    return chunks


def parse_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Extract text per page from a PDF. Returns list of (page_number, text)."""
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(filename=str(path), strategy="fast")

    per_page: dict[int, list[str]] = {}
    for el in elements:
        page = getattr(el.metadata, "page_number", None) or 1
        text = str(el).strip()
        if text:
            per_page.setdefault(page, []).append(text)

    return [(p, " ".join(per_page[p])) for p in sorted(per_page.keys())]


def chunk_pdf(
    path: str | Path, max_tokens: int, overlap_tokens: int
) -> tuple[list[Chunk], int]:
    """Parse a PDF and chunk it. Returns (chunks, page_count)."""
    pages = parse_pdf(path)
    page_count = len(pages) if pages else 0
    return chunk_text(pages, max_tokens=max_tokens, overlap_tokens=overlap_tokens), page_count
```

- [ ] **Step 3: Run chunker unit tests**

Run:
```bash
cd backend && pytest tests/unit/test_chunker.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/app/core/__init__.py backend/app/core/chunker.py
git commit -m "feat(backend): implement token-budgeted sentence-aware chunker"
```

---

### Task 13: Citation schema + system prompt (tests)

**Files:**
- Create: `backend/tests/unit/test_citations.py`

- [ ] **Step 1: Write failing tests in `backend/tests/unit/test_citations.py`**

```python
from app.core.citations import (
    RAG_ANSWER_SCHEMA,
    RAG_SYSTEM_PROMPT,
    build_user_prompt,
    verify_citations,
)


def test_schema_has_required_shape():
    assert RAG_ANSWER_SCHEMA["type"] == "object"
    props = RAG_ANSWER_SCHEMA["properties"]
    assert "answer" in props and props["answer"]["type"] == "string"
    assert "citations" in props and props["citations"]["type"] == "array"
    item = props["citations"]["items"]
    for key in ("source", "page", "score"):
        assert key in item["properties"]


def test_system_prompt_mentions_no_invention():
    assert "invent" in RAG_SYSTEM_PROMPT.lower() or "never" in RAG_SYSTEM_PROMPT.lower()


def test_build_user_prompt_includes_question_and_sources():
    prompt = build_user_prompt(
        question="What is X?",
        chunks=[
            {"source": "a.pdf", "page": 1, "content": "X is a thing."},
            {"source": "b.pdf", "page": 3, "content": "Y is another thing."},
        ],
    )
    assert "What is X?" in prompt
    assert "a.pdf" in prompt and "page=1" in prompt
    assert "b.pdf" in prompt and "page=3" in prompt
    assert "X is a thing." in prompt


def test_verify_citations_drops_invalid():
    retrieved = [
        {"source": "a.pdf", "page": 1},
        {"source": "b.pdf", "page": 3},
    ]
    citations = [
        {"source": "a.pdf", "page": 1, "score": 0.9},
        {"source": "c.pdf", "page": 7, "score": 0.8},  # not retrieved
        {"source": "b.pdf", "page": 99, "score": 0.7},  # wrong page
    ]
    verified = verify_citations(citations, retrieved)
    assert len(verified) == 1
    assert verified[0]["source"] == "a.pdf"
    assert verified[0]["page"] == 1


def test_verify_citations_handles_empty():
    assert verify_citations([], []) == []
    assert verify_citations([{"source": "a.pdf", "page": 1, "score": 0.5}], []) == []
```

- [ ] **Step 2: Run tests — verify they fail**

Run:
```bash
cd backend && pytest tests/unit/test_citations.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.core.citations'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add backend/tests/unit/test_citations.py
git commit -m "test(backend): add failing citation schema + verifier tests"
```

---

### Task 14: Citation schema + verifier — implementation

**Files:**
- Create: `backend/app/core/citations.py`

- [ ] **Step 1: Write `backend/app/core/citations.py`**

```python
from typing import Any

RAG_SYSTEM_PROMPT = """You answer strictly from the provided context chunks.

Rules:
- Every factual claim in your answer must be supported by a chunk visible in the context.
- In the `citations` field, include one entry per chunk you actually used. Use the exact `source` filename and `page` shown in the chunk header. Provide a `score` between 0.0 and 1.0 reflecting how much you relied on that chunk.
- If the context does not contain the answer, say so explicitly and return an empty `citations` array. Do NOT guess.
- Never invent sources, page numbers, or facts not present in the context.
"""

RAG_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source": {"type": "string"},
                    "page": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["source", "page", "score"],
            },
        },
    },
    "required": ["answer", "citations"],
}


def build_user_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Render the user message: question + numbered context chunks with source/page headers."""
    lines: list[str] = [f"Question: {question}", "", "Context:"]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[chunk {i} | source=\"{c['source']}\" page={c['page']}]")
        lines.append(str(c["content"]).strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def verify_citations(
    citations: list[dict[str, Any]], retrieved: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Drop any (source, page) pair in citations that is not present in retrieved chunks."""
    allowed = {(r["source"], r["page"]) for r in retrieved}
    return [c for c in citations if (c.get("source"), c.get("page")) in allowed]
```

- [ ] **Step 2: Run citation tests**

Run:
```bash
cd backend && pytest tests/unit/test_citations.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add backend/app/core/citations.py
git commit -m "feat(backend): add citation schema, prompt, and verifier"
```

---

### Task 15: RRF fusion — tests

**Files:**
- Create: `backend/tests/unit/test_rrf.py`

- [ ] **Step 1: Write failing tests in `backend/tests/unit/test_rrf.py`**

```python
from app.core.retriever import rrf_fuse


def test_rrf_single_ranker_preserves_order():
    ranker = [("a", {}), ("b", {}), ("c", {})]
    fused = rrf_fuse([ranker], k=60, top_k=3)
    assert [x[0] for x in fused] == ["a", "b", "c"]


def test_rrf_both_rankers_agree():
    r1 = [("a", {}), ("b", {}), ("c", {})]
    r2 = [("a", {}), ("b", {}), ("c", {})]
    fused = rrf_fuse([r1, r2], k=60, top_k=3)
    assert [x[0] for x in fused] == ["a", "b", "c"]


def test_rrf_combines_disagreeing_rankers():
    r1 = [("a", {}), ("b", {}), ("c", {})]
    r2 = [("b", {}), ("a", {}), ("c", {})]
    fused = rrf_fuse([r1, r2], k=60, top_k=3)
    # a: 1/61 + 1/62 ≈ 0.03236
    # b: 1/62 + 1/61 ≈ 0.03236
    # c: 1/63 + 1/63 ≈ 0.03174
    # a and b tie for top — tie-break by id order doesn't matter, just ensure c is last.
    ids = [x[0] for x in fused]
    assert ids[2] == "c"
    assert set(ids[:2]) == {"a", "b"}


def test_rrf_chunks_only_in_one_ranker_included():
    r1 = [("a", {}), ("b", {})]
    r2 = [("c", {}), ("a", {})]
    fused = rrf_fuse([r1, r2], k=60, top_k=10)
    ids = {x[0] for x in fused}
    assert ids == {"a", "b", "c"}


def test_rrf_top_k_limits_output():
    r1 = [(f"id{i}", {}) for i in range(50)]
    fused = rrf_fuse([r1], k=60, top_k=5)
    assert len(fused) == 5


def test_rrf_preserves_payload():
    r1 = [("a", {"content": "hello", "page": 1})]
    fused = rrf_fuse([r1], k=60, top_k=1)
    assert fused[0][1]["content"] == "hello"
    assert fused[0][1]["page"] == 1
```

- [ ] **Step 2: Run — verify they fail**

Run:
```bash
cd backend && pytest tests/unit/test_rrf.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.core.retriever'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add backend/tests/unit/test_rrf.py
git commit -m "test(backend): add failing RRF fusion tests"
```

---

### Task 16: RRF fusion — implementation (pure helper)

**Files:**
- Create: `backend/app/core/retriever.py` (pure helper only, orchestration added later)

- [ ] **Step 1: Write `backend/app/core/retriever.py` with just the pure fusion helper**

```python
from typing import Any, Hashable


def rrf_fuse(
    ranked_lists: list[list[tuple[Hashable, Any]]],
    k: int,
    top_k: int,
) -> list[tuple[Hashable, Any]]:
    """Reciprocal Rank Fusion.

    Each input list is ranked best-first. Score(item) = Σ 1 / (k + rank_i) across lists where
    item appears (1-indexed rank). Returns items sorted by descending fused score, limited to top_k.
    Item payloads are merged (later lists override earlier keys).
    """
    scores: dict[Hashable, float] = {}
    payloads: dict[Hashable, Any] = {}

    for ranked in ranked_lists:
        for rank, (item_id, payload) in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in payloads:
                payloads[item_id] = payload
            elif isinstance(payload, dict) and isinstance(payloads[item_id], dict):
                payloads[item_id] = {**payloads[item_id], **payload}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(item_id, payloads[item_id]) for item_id, _ in ranked[:top_k]]
```

- [ ] **Step 2: Run RRF tests**

Run:
```bash
cd backend && pytest tests/unit/test_rrf.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git add backend/app/core/retriever.py
git commit -m "feat(backend): add pure RRF fusion helper"
```

---

## Phase D — Core modules with external dependencies

### Task 17: Embeddings wrapper

**Files:**
- Create: `backend/app/core/embeddings.py`
- Create: `backend/tests/unit/test_embeddings.py`

- [ ] **Step 1: Write failing tests in `backend/tests/unit/test_embeddings.py`**

```python
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
```

- [ ] **Step 2: Run — verify they fail**

Run:
```bash
cd backend && pytest tests/unit/test_embeddings.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `backend/app/core/embeddings.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run:
```bash
cd backend && pytest tests/unit/test_embeddings.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/embeddings.py backend/tests/unit/test_embeddings.py
git commit -m "feat(backend): add batched embeddings client with retry"
```

---

### Task 18: OpenAI client factory

**Files:**
- Create: `backend/app/core/openai_client.py`

- [ ] **Step 1: Write `backend/app/core/openai_client.py`**

```python
from functools import lru_cache

from openai import AsyncOpenAI

from app.config import get_settings


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().OPENAI_API_KEY)
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/core/openai_client.py
git commit -m "feat(backend): add cached OpenAI async client factory"
```

---

### Task 19: Retriever orchestration

**Files:**
- Modify: `backend/app/core/retriever.py`

- [ ] **Step 1: Append retrieval orchestration to `backend/app/core/retriever.py`**

After the existing `rrf_fuse` function, add:

```python
import asyncio
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    doc_id: str
    source: str
    page: int
    content: str
    score: float


async def _vector_search(
    session: AsyncSession, query_embedding: list[float], k: int
) -> list[tuple[int, dict]]:
    sql = text(
        """
        SELECT c.id, c.doc_id, d.filename, c.page_number, c.content,
               (c.embedding <=> (:q_emb)::vector) AS distance
        FROM chunks c
        JOIN docs d ON d.id = c.doc_id
        ORDER BY c.embedding <=> (:q_emb)::vector
        LIMIT :k
        """
    )
    result = await session.execute(sql, {"q_emb": str(query_embedding), "k": k})
    rows = result.all()
    return [
        (
            r.id,
            {
                "doc_id": str(r.doc_id),
                "source": r.filename,
                "page": r.page_number,
                "content": r.content,
                "distance": float(r.distance),
            },
        )
        for r in rows
    ]


async def _keyword_search(
    session: AsyncSession, query: str, k: int
) -> list[tuple[int, dict]]:
    sql = text(
        """
        SELECT c.id, c.doc_id, d.filename, c.page_number, c.content,
               ts_rank(c.content_tsv, plainto_tsquery('english', :q)) AS rank
        FROM chunks c
        JOIN docs d ON d.id = c.doc_id
        WHERE c.content_tsv @@ plainto_tsquery('english', :q)
        ORDER BY rank DESC
        LIMIT :k
        """
    )
    result = await session.execute(sql, {"q": query, "k": k})
    rows = result.all()
    return [
        (
            r.id,
            {
                "doc_id": str(r.doc_id),
                "source": r.filename,
                "page": r.page_number,
                "content": r.content,
                "rank": float(r.rank),
            },
        )
        for r in rows
    ]


async def retrieve(
    session: AsyncSession,
    query: str,
    query_embedding: list[float],
    candidate_k: int,
    top_k: int,
    rrf_k: int,
) -> list[RetrievedChunk]:
    vec_task = _vector_search(session, query_embedding, candidate_k)
    kw_task = _keyword_search(session, query, candidate_k)
    vec_hits, kw_hits = await asyncio.gather(vec_task, kw_task)

    fused = rrf_fuse([vec_hits, kw_hits], k=rrf_k, top_k=top_k)
    out: list[RetrievedChunk] = []
    max_score = max((s for _, s in _fused_with_scores([vec_hits, kw_hits], rrf_k)), default=1.0)
    score_map = dict(_fused_with_scores([vec_hits, kw_hits], rrf_k))

    for item_id, payload in fused:
        raw = score_map.get(item_id, 0.0)
        out.append(
            RetrievedChunk(
                chunk_id=int(item_id),
                doc_id=payload["doc_id"],
                source=payload["source"],
                page=payload["page"],
                content=payload["content"],
                score=raw / max_score if max_score > 0 else 0.0,
            )
        )
    return out


def _fused_with_scores(
    ranked_lists: list[list[tuple[int, dict]]], k: int
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return list(scores.items())
```

- [ ] **Step 2: Verify unit tests still pass**

Run:
```bash
cd backend && pytest tests/unit -v
```

Expected: all unit tests still pass (the pure `rrf_fuse` function is unchanged).

- [ ] **Step 3: Commit**

```bash
git add backend/app/core/retriever.py
git commit -m "feat(backend): add parallel vector+keyword retrieval with RRF"
```

---

### Task 20: Generator — tests

**Files:**
- Create: `backend/tests/unit/test_generator.py`

- [ ] **Step 1: Write failing tests in `backend/tests/unit/test_generator.py`**

```python
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
```

- [ ] **Step 2: Run — verify they fail**

Run:
```bash
cd backend && pytest tests/unit/test_generator.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.core.generator'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add backend/tests/unit/test_generator.py
git commit -m "test(backend): add failing generator streaming tests"
```

---

### Task 21: Generator — implementation

**Files:**
- Create: `backend/app/core/generator.py`

- [ ] **Step 1: Write `backend/app/core/generator.py`**

```python
from collections.abc import AsyncIterator
from typing import Any

import json_repair

from app.core.citations import (
    RAG_ANSWER_SCHEMA,
    RAG_SYSTEM_PROMPT,
    build_user_prompt,
    verify_citations,
)

MAX_HISTORY_TURNS = 8


def _truncate_history(
    history: list[dict[str, str]], max_turns: int = MAX_HISTORY_TURNS
) -> list[dict[str, str]]:
    """Keep the last N turns (each turn = one user msg + one assistant msg = 2 items)."""
    return history[-(max_turns * 2) :]


async def generate_stream(
    openai_client: Any,
    model: str,
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]],
) -> AsyncIterator[dict[str, Any]]:
    """Stream a RAG answer from OpenAI. Yields SSE-shaped dicts:
      {"event": "answer_delta", "data": {"text": str}}
      {"event": "citation",     "data": {source, page, score}}
      {"event": "done",         "data": {"verified_citations": [...]}}
      {"event": "error",        "data": {"message": str}}
    """
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    messages.extend(_truncate_history(history))
    messages.append({"role": "user", "content": build_user_prompt(question, chunks)})

    try:
        stream = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_answer",
                    "strict": True,
                    "schema": RAG_ANSWER_SCHEMA,
                },
            },
        )
    except Exception as e:
        yield {"event": "error", "data": {"message": f"upstream: {type(e).__name__}: {e}"}}
        return

    buf = ""
    emitted_answer_len = 0
    emitted_citation_keys: set[tuple[str, int]] = set()

    try:
        async for event in stream:
            delta = event.choices[0].delta.content or ""
            if not delta:
                continue
            buf += delta

            try:
                parsed = json_repair.loads(buf)
            except Exception:
                continue
            if not isinstance(parsed, dict):
                continue

            # 1) Emit new answer characters.
            answer_so_far = parsed.get("answer", "")
            if isinstance(answer_so_far, str) and len(answer_so_far) > emitted_answer_len:
                new_text = answer_so_far[emitted_answer_len:]
                emitted_answer_len = len(answer_so_far)
                yield {"event": "answer_delta", "data": {"text": new_text}}

            # 2) Emit any newly-complete citation entries.
            cits = parsed.get("citations", [])
            if isinstance(cits, list):
                for c in cits:
                    if not isinstance(c, dict):
                        continue
                    src = c.get("source")
                    page = c.get("page")
                    score = c.get("score")
                    if not (isinstance(src, str) and isinstance(page, int) and isinstance(score, (int, float))):
                        continue
                    key = (src, page)
                    if key in emitted_citation_keys:
                        continue
                    emitted_citation_keys.add(key)
                    yield {
                        "event": "citation",
                        "data": {"source": src, "page": page, "score": float(score)},
                    }
    except Exception as e:
        yield {"event": "error", "data": {"message": f"stream: {type(e).__name__}: {e}"}}
        return

    # Final verification pass on the fully accumulated buffer.
    try:
        final = json_repair.loads(buf) if buf else {}
    except Exception:
        final = {}
    final_citations = final.get("citations", []) if isinstance(final, dict) else []
    retrieved_pairs = [{"source": c["source"], "page": c["page"]} for c in chunks]
    verified = verify_citations(final_citations if isinstance(final_citations, list) else [], retrieved_pairs)

    yield {"event": "done", "data": {"verified_citations": verified}}
```

- [ ] **Step 2: Run generator tests**

Run:
```bash
cd backend && pytest tests/unit/test_generator.py -v
```

Expected: both tests pass.

- [ ] **Step 3: Run the full unit suite**

Run:
```bash
cd backend && pytest tests/unit -v
```

Expected: all unit tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/app/core/generator.py
git commit -m "feat(backend): add streaming generator with partial-JSON parsing and citation verification"
```

---

## Phase E — Schemas, app factory, and routes

### Task 22: Pydantic request/response schemas

**Files:**
- Create: `backend/app/schemas.py`

- [ ] **Step 1: Write `backend/app/schemas.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat(backend): add pydantic request/response schemas"
```

---

### Task 23: FastAPI app factory + health route

**Files:**
- Create: `backend/app/main.py`
- Create: `backend/app/routes/__init__.py`
- Create: `backend/app/routes/health.py`

- [ ] **Step 1: Create empty `backend/app/routes/__init__.py`**

```python
```

- [ ] **Step 2: Write `backend/app/routes/health.py`**

```python
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 3: Write `backend/app/main.py`**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import docs, health, ingest, query


def create_app() -> FastAPI:
    settings = get_settings()
    # Swagger moved to /swagger so GET /docs (our docs-list route) is not shadowed.
    app = FastAPI(
        title="DocSage",
        version="0.1.0",
        docs_url="/swagger",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(docs.router, prefix="/docs", tags=["docs"])
    app.include_router(ingest.router, tags=["ingest"])
    app.include_router(query.router, tags=["query"])

    return app


app = create_app()
```

Note: `docs`, `ingest`, `query` imports will resolve in later tasks — this file won't run yet. That's fine; we'll write those next.

- [ ] **Step 4: Commit health route + app factory (imports will fail on run until all routes exist)**

```bash
git add backend/app/main.py backend/app/routes/__init__.py backend/app/routes/health.py
git commit -m "feat(backend): add FastAPI app factory and health route"
```

---

### Task 24: Docs route (list + delete)

**Files:**
- Create: `backend/app/routes/docs.py`

- [ ] **Step 1: Write `backend/app/routes/docs.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/routes/docs.py
git commit -m "feat(backend): add GET /docs and DELETE /docs/{id} routes"
```

---

### Task 25: Ingest route

**Files:**
- Create: `backend/app/routes/ingest.py`

- [ ] **Step 1: Write `backend/app/routes/ingest.py`**

```python
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
```

Note: `Chunk(... content_tsv=...)` is intentionally omitted — the column is `GENERATED ALWAYS AS ... STORED` in Postgres. SQLAlchemy will complain if we try to write to it. On read, it will be populated.

- [ ] **Step 2: Commit**

```bash
git add backend/app/routes/ingest.py
git commit -m "feat(backend): add POST /ingest route with PDF chunking + embedding"
```

---

### Task 26: Query route (SSE)

**Files:**
- Create: `backend/app/routes/query.py`

- [ ] **Step 1: Write `backend/app/routes/query.py`**

```python
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
```

- [ ] **Step 2: Verify app boots**

Run (ensure DB is up + migrated first):
```bash
docker compose up -d --wait
cd backend && alembic upgrade head
cd backend && uvicorn app.main:app --port 8000 &
sleep 2
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok"}`.

- [ ] **Step 3: Stop the background server**

Run:
```bash
kill %1 2>/dev/null || pkill -f "uvicorn app.main:app" || true
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/routes/query.py
git commit -m "feat(backend): add POST /query route with SSE streaming"
```

---

## Phase F — Backend integration tests

### Task 27: Integration test fixtures

**Files:**
- Create: `backend/tests/integration/conftest.py`
- Create: `backend/tests/fixtures/sample.txt`

- [ ] **Step 1: Write `backend/tests/fixtures/sample.txt`** (source of PDF fixture)

```
DocSage Sample Document

Page 1: Introduction
This is a sample document used for testing the ingestion pipeline.
The quick brown fox jumps over the lazy dog.
Retrieval-augmented generation combines retrieval with language models.

Page 2: Details
Pgvector enables efficient similarity search in Postgres.
Hybrid search fuses vector similarity with keyword matching.
Reciprocal rank fusion is a standard technique for combining rankings.
```

- [ ] **Step 2: Create a tiny PDF fixture programmatically (one-time setup script)**

Run:
```bash
cd backend && python -c "
from pypdf import PdfWriter
from pypdf.generic import NameObject, TextStringObject

# Minimal 2-page PDF; pypdf cannot easily write text, so we use reportlab if available,
# falling back to a known-minimal hand-written PDF.
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    c = canvas.Canvas('tests/fixtures/sample.pdf', pagesize=letter)
    c.drawString(72, 720, 'DocSage Sample Document')
    c.drawString(72, 700, 'This is a sample document used for testing ingestion.')
    c.drawString(72, 680, 'Retrieval augmented generation combines retrieval with language models.')
    c.drawString(72, 660, 'The quick brown fox jumps over the lazy dog.')
    c.showPage()
    c.drawString(72, 720, 'Page Two Details')
    c.drawString(72, 700, 'Pgvector enables similarity search in Postgres.')
    c.drawString(72, 680, 'Hybrid search fuses vector similarity with keyword matching.')
    c.drawString(72, 660, 'Reciprocal rank fusion combines rankings from multiple retrievers.')
    c.showPage()
    c.save()
    print('wrote tests/fixtures/sample.pdf via reportlab')
except ImportError:
    pass
"
```

If reportlab is not installed, install it first:
```bash
pip install reportlab
```
Then re-run the command above.

Expected: `tests/fixtures/sample.pdf` is created (2 pages, real text).

- [ ] **Step 3: Write `backend/tests/integration/conftest.py`**

```python
import asyncio
import os
import subprocess
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Ensure env is set before app imports.
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://docsage:docsage@localhost:5432/docsage",
)


@pytest.fixture(scope="session", autouse=True)
def ensure_db_migrated() -> Iterator[None]:
    """Run `alembic upgrade head` once per session."""
    subprocess.run(["alembic", "upgrade", "head"], check=True, cwd=".")
    yield


@pytest_asyncio.fixture
async def clean_db() -> AsyncIterator[None]:
    """Truncate docs + chunks before each integration test for isolation."""
    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE chunks, docs RESTART IDENTITY CASCADE"))
    await engine.dispose()
    yield


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
```

- [ ] **Step 4: Add reportlab to dev requirements**

Append to `backend/requirements-dev.txt`:
```
reportlab==4.1.0
```

Run:
```bash
cd backend && pip install -r requirements-dev.txt
```

- [ ] **Step 5: Commit**

```bash
git add backend/tests/integration/conftest.py backend/tests/fixtures/sample.pdf backend/tests/fixtures/sample.txt backend/requirements-dev.txt
git commit -m "test(backend): add integration test fixtures and conftest"
```

---

### Task 28: Ingest integration test

**Files:**
- Create: `backend/tests/integration/test_ingest_flow.py`

- [ ] **Step 1: Write `backend/tests/integration/test_ingest_flow.py`**

```python
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

    oc.get_openai_client.cache_clear()
    monkeypatch.setattr(oc, "get_openai_client", lambda: fake_openai)

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
```

- [ ] **Step 2: Run integration tests**

Run:
```bash
docker compose up -d --wait
cd backend && pytest tests/integration/test_ingest_flow.py -v
```

Expected: test passes.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/integration/test_ingest_flow.py
git commit -m "test(backend): add ingest-flow integration test with mocked OpenAI"
```

---

### Task 29: Query integration test

**Files:**
- Create: `backend/tests/integration/test_query_flow.py`

- [ ] **Step 1: Write `backend/tests/integration/test_query_flow.py`**

```python
import hashlib
import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


def _deterministic_vector(s: str) -> list[float]:
    h = hashlib.sha256(s.encode()).digest()
    return [((h[i % len(h)] / 255.0) - 0.5) for i in range(1536)]


class _FakeStream:
    def __init__(self, pieces):
        self._pieces = pieces

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._pieces:
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=c))])


@pytest.mark.asyncio
async def test_query_streams_answer_and_citations(client, clean_db, monkeypatch):
    # Seed one doc + one chunk directly in the DB.
    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "INSERT INTO docs (filename, page_count) VALUES (:f, :p) RETURNING id"
            ),
            {"f": "seed.pdf", "p": 1},
        )
        doc_id = result.scalar_one()
        vec = _deterministic_vector("pgvector enables similarity search in postgres")
        await conn.execute(
            text(
                "INSERT INTO chunks (doc_id, page_number, chunk_index, content, embedding) "
                "VALUES (:d, 1, 0, :c, (:e)::vector)"
            ),
            {
                "d": doc_id,
                "c": "Pgvector enables similarity search in Postgres using cosine distance.",
                "e": str(vec),
            },
        )
    await engine.dispose()

    # Mock OpenAI: embeddings return deterministic vectors, chat returns a streamed JSON.
    async def fake_embed_create(*, model, input):
        resp = MagicMock()
        resp.data = [MagicMock(embedding=_deterministic_vector(t)) for t in input]
        return resp

    full = json.dumps(
        {
            "answer": "Pgvector enables similarity search.",
            "citations": [{"source": "seed.pdf", "page": 1, "score": 0.9}],
        }
    )
    pieces = [full[i : i + 10] for i in range(0, len(full), 10)]

    async def fake_chat_create(**_):
        return _FakeStream(pieces)

    fake_openai = MagicMock()
    fake_openai.embeddings = MagicMock()
    fake_openai.embeddings.create = AsyncMock(side_effect=fake_embed_create)
    fake_openai.chat = MagicMock()
    fake_openai.chat.completions = MagicMock()
    fake_openai.chat.completions.create = AsyncMock(side_effect=fake_chat_create)

    from app.core import openai_client as oc

    oc.get_openai_client.cache_clear()
    monkeypatch.setattr(oc, "get_openai_client", lambda: fake_openai)

    resp = await client.post(
        "/query",
        json={"question": "What does pgvector enable?", "history": []},
    )
    assert resp.status_code == 200
    text_out = resp.text

    # sse-starlette serialises events like:  event: answer_delta\ndata: {...}\n\n
    assert "event: answer_delta" in text_out
    assert "event: citation" in text_out
    assert "event: done" in text_out
    assert "seed.pdf" in text_out
```

- [ ] **Step 2: Run integration tests**

Run:
```bash
cd backend && pytest tests/integration/test_query_flow.py -v
```

Expected: test passes.

- [ ] **Step 3: Run the entire backend test suite**

Run:
```bash
cd backend && pytest -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/integration/test_query_flow.py
git commit -m "test(backend): add query-flow integration test (SSE stream)"
```

---

### Task 30: Smoke test + eval scaffold

**Files:**
- Create: `backend/tests/smoke/test_real_openai.py`
- Create: `backend/tests/eval/golden.yaml`
- Create: `backend/tests/eval/run_eval.py`
- Create: `backend/tests/eval/__init__.py`

- [ ] **Step 1: Create `backend/tests/eval/__init__.py`**

```python
```

- [ ] **Step 2: Write `backend/tests/smoke/test_real_openai.py`**

```python
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
```

- [ ] **Step 3: Write `backend/tests/eval/golden.yaml`**

```yaml
# DocSage evaluation golden set.
#
# Populate with (question, expected_source, expected_page) triples against
# a known corpus. Use `make eval` to score answer presence + citation
# precision/recall. Empty at scaffold time.
version: 1
items: []
```

- [ ] **Step 4: Write `backend/tests/eval/run_eval.py`**

```python
"""Evaluation harness scaffold.

Loads tests/eval/golden.yaml, runs each item against the live backend, and
scores answer presence + citation precision/recall. Currently reports zero
because the golden set is empty — populate golden.yaml to use.
"""
from pathlib import Path

import yaml


def main() -> None:
    path = Path(__file__).parent / "golden.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    items = data.get("items", [])
    print(f"loaded {len(items)} golden-set items from {path}")
    if not items:
        print("golden set is empty — add items to evaluate")
        return
    # TODO (future): run each question against POST /query, score citations.
    print("eval harness is a scaffold; full implementation pending roadmap task")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add pyyaml to dev deps**

Append to `backend/requirements-dev.txt`:
```
pyyaml==6.0.1
```

Install:
```bash
cd backend && pip install -r requirements-dev.txt
```

- [ ] **Step 6: Verify eval runs**

Run:
```bash
cd backend && python tests/eval/run_eval.py
```

Expected: prints `loaded 0 golden-set items from ...` and `golden set is empty — add items to evaluate`.

- [ ] **Step 7: Commit**

```bash
git add backend/tests/smoke/test_real_openai.py backend/tests/eval/ backend/requirements-dev.txt
git commit -m "test(backend): add smoke test + eval harness scaffold"
```

---

## Phase G — Frontend

### Task 31: Next.js scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/next.config.mjs`
- Create: `frontend/postcss.config.mjs`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/.gitignore`
- Create: `frontend/.env.example`

- [ ] **Step 1: Verify pnpm is installed**

Run:
```bash
pnpm --version
```

If missing:
```bash
npm install -g pnpm
```

- [ ] **Step 2: Write `frontend/package.json`**

```json
{
  "name": "docsage-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "format": "prettier --write .",
    "test": "vitest"
  },
  "dependencies": {
    "@microsoft/fetch-event-source": "2.0.1",
    "clsx": "2.1.0",
    "lucide-react": "0.372.0",
    "next": "14.2.3",
    "react": "18.3.1",
    "react-dom": "18.3.1",
    "react-dropzone": "14.2.3",
    "tailwind-merge": "2.2.2"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "6.4.2",
    "@testing-library/react": "15.0.2",
    "@types/node": "20.12.7",
    "@types/react": "18.2.79",
    "@types/react-dom": "18.2.25",
    "@vitejs/plugin-react": "4.2.1",
    "autoprefixer": "10.4.19",
    "eslint": "8.57.0",
    "eslint-config-next": "14.2.3",
    "jsdom": "24.0.0",
    "postcss": "8.4.38",
    "prettier": "3.2.5",
    "prettier-plugin-tailwindcss": "0.5.14",
    "tailwindcss": "3.4.3",
    "typescript": "5.4.5",
    "vitest": "1.5.0"
  }
}
```

- [ ] **Step 3: Install**

Run:
```bash
cd frontend && pnpm install
```

Expected: lockfile + node_modules created.

- [ ] **Step 4: Write `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": false,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 5: Write `frontend/next.config.mjs`**

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
```

- [ ] **Step 6: Write `frontend/postcss.config.mjs`**

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

- [ ] **Step 7: Write `frontend/tailwind.config.ts`**

```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f5f7ff",
          100: "#e3e8ff",
          500: "#4f46e5",
          600: "#4338ca",
          700: "#3730a3",
        },
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
```

- [ ] **Step 8: Write `frontend/.gitignore`**

```
node_modules/
.next/
out/
.turbo/
.env.local
.env
coverage/
*.tsbuildinfo
```

- [ ] **Step 9: Write `frontend/.env.example`**

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

- [ ] **Step 10: Commit**

```bash
git add frontend/package.json frontend/pnpm-lock.yaml frontend/tsconfig.json frontend/next.config.mjs frontend/postcss.config.mjs frontend/tailwind.config.ts frontend/.gitignore frontend/.env.example
git commit -m "chore(frontend): Next.js 14 + TypeScript + Tailwind scaffold"
```

---

### Task 32: App layout + global styles

**Files:**
- Create: `frontend/app/layout.tsx`
- Create: `frontend/app/globals.css`
- Create: `frontend/lib/utils.ts`

- [ ] **Step 1: Write `frontend/app/globals.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

html, body {
  height: 100%;
}

body {
  @apply bg-white text-slate-900 antialiased;
}

/* scrollbar polish */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { @apply bg-slate-300 rounded; }
::-webkit-scrollbar-track { @apply bg-transparent; }

@keyframes typing-dot {
  0%, 80%, 100% { opacity: 0.2; }
  40% { opacity: 1; }
}
.typing-dot { animation: typing-dot 1.4s infinite ease-in-out both; }
```

- [ ] **Step 2: Write `frontend/app/layout.tsx`**

```tsx
import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "DocSage",
  description: "RAG chatbot over your PDFs with inline citations.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
```

- [ ] **Step 3: Write `frontend/lib/utils.ts`**

```typescript
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/app/layout.tsx frontend/app/globals.css frontend/lib/utils.ts
git commit -m "feat(frontend): add root layout, global styles, and utils"
```

---

### Task 33: API proxy routes

**Files:**
- Create: `frontend/app/api/ingest/route.ts`
- Create: `frontend/app/api/query/route.ts`
- Create: `frontend/app/api/docs/route.ts`
- Create: `frontend/app/api/docs/[id]/route.ts`

- [ ] **Step 1: Write `frontend/app/api/ingest/route.ts`**

```typescript
import { NextRequest } from "next/server";

export const runtime = "nodejs";

const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const upstream = await fetch(`${BACKEND}/ingest`, {
    method: "POST",
    body: req.body,
    // @ts-expect-error: Node fetch requires duplex for streaming bodies
    duplex: "half",
    headers: {
      "content-type": req.headers.get("content-type") ?? "",
    },
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: upstream.headers,
  });
}
```

- [ ] **Step 2: Write `frontend/app/api/query/route.ts`**

```typescript
import { NextRequest } from "next/server";

export const runtime = "nodejs";

const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const body = await req.text();
  const upstream = await fetch(`${BACKEND}/query`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "text/event-stream",
    },
    body,
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}
```

- [ ] **Step 3: Write `frontend/app/api/docs/route.ts`**

```typescript
const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function GET() {
  const upstream = await fetch(`${BACKEND}/docs`, { cache: "no-store" });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: upstream.headers,
  });
}
```

- [ ] **Step 4: Write `frontend/app/api/docs/[id]/route.ts`**

```typescript
const BACKEND = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function DELETE(_: Request, { params }: { params: { id: string } }) {
  const upstream = await fetch(`${BACKEND}/docs/${encodeURIComponent(params.id)}`, {
    method: "DELETE",
  });
  return new Response(null, { status: upstream.status });
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/app/api
git commit -m "feat(frontend): add Next.js API proxy routes to backend"
```

---

### Task 34: SSE client + typed types

**Files:**
- Create: `frontend/lib/types.ts`
- Create: `frontend/lib/sse.ts`

- [ ] **Step 1: Write `frontend/lib/types.ts`**

```typescript
export interface DocSummary {
  id: string;
  filename: string;
  page_count: number;
  uploaded_at: string;
}

export interface Citation {
  source: string;
  page: number;
  score: number;
}

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  citations?: Citation[];
  streaming?: boolean;
  error?: string;
}

export interface IngestResponse {
  doc_id: string;
  filename: string;
  page_count: number;
  chunk_count: number;
}
```

- [ ] **Step 2: Write `frontend/lib/sse.ts`**

```typescript
import { fetchEventSource } from "@microsoft/fetch-event-source";
import type { Citation } from "./types";

export interface StreamHandlers {
  onAnswerDelta: (text: string) => void;
  onCitation: (c: Citation) => void;
  onDone: (verified: Citation[]) => void;
  onError: (message: string) => void;
}

export async function streamQuery(
  body: { question: string; history: { role: string; content: string }[] },
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  await fetchEventSource("/api/query", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
    openWhenHidden: true,
    onmessage(ev) {
      try {
        const data = JSON.parse(ev.data);
        if (ev.event === "answer_delta") handlers.onAnswerDelta(data.text);
        else if (ev.event === "citation") handlers.onCitation(data);
        else if (ev.event === "done") handlers.onDone(data.verified_citations ?? []);
        else if (ev.event === "error") handlers.onError(data.message ?? "unknown error");
      } catch (e) {
        handlers.onError(`bad event: ${String(e)}`);
      }
    },
    onerror(err) {
      handlers.onError(String(err));
      throw err; // stop reconnection
    },
  });
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/lib/types.ts frontend/lib/sse.ts
git commit -m "feat(frontend): add typed SSE client and shared types"
```

---

### Task 35: PDF drop + doc list components

**Files:**
- Create: `frontend/components/PdfDrop.tsx`
- Create: `frontend/components/DocList.tsx`

- [ ] **Step 1: Write `frontend/components/PdfDrop.tsx`**

```tsx
"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud, Loader2 } from "lucide-react";
import { cn, formatBytes } from "@/lib/utils";
import type { IngestResponse } from "@/lib/types";

interface Props {
  onUploaded: (resp: IngestResponse) => void;
}

export function PdfDrop({ onUploaded }: Props) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    async (files: File[]) => {
      setError(null);
      for (const file of files) {
        const form = new FormData();
        form.append("file", file);
        setUploading(true);
        try {
          const resp = await fetch("/api/ingest", { method: "POST", body: form });
          if (!resp.ok) {
            const t = await resp.text();
            throw new Error(`upload failed: ${resp.status} ${t}`);
          }
          const body: IngestResponse = await resp.json();
          onUploaded(body);
        } catch (e: unknown) {
          setError(e instanceof Error ? e.message : String(e));
        } finally {
          setUploading(false);
        }
      }
    },
    [onUploaded],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    disabled: uploading,
  });

  return (
    <div>
      <div
        {...getRootProps()}
        className={cn(
          "flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 text-center cursor-pointer transition",
          isDragActive ? "border-brand-500 bg-brand-50" : "border-slate-300 hover:border-slate-400",
          uploading && "opacity-60 cursor-not-allowed",
        )}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin text-brand-500" />
            <p className="text-sm text-slate-500">Uploading & indexing…</p>
          </>
        ) : (
          <>
            <UploadCloud className="h-5 w-5 text-slate-400" />
            <p className="text-sm font-medium">Drop a PDF here or click to select</p>
            <p className="text-xs text-slate-400">Max {formatBytes(25 * 1024 * 1024)} per file</p>
          </>
        )}
      </div>
      {error && <p className="mt-2 text-xs text-red-600">{error}</p>}
    </div>
  );
}
```

- [ ] **Step 2: Write `frontend/components/DocList.tsx`**

```tsx
"use client";

import { useEffect, useState, useCallback } from "react";
import { Trash2, FileText } from "lucide-react";
import type { DocSummary } from "@/lib/types";

interface Props {
  refreshKey: number;
}

export function DocList({ refreshKey }: Props) {
  const [docs, setDocs] = useState<DocSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch("/api/docs", { cache: "no-store" });
      if (resp.ok) setDocs(await resp.json());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  const onDelete = async (id: string) => {
    await fetch(`/api/docs/${id}`, { method: "DELETE" });
    await load();
  };

  if (loading) return <p className="text-xs text-slate-400">Loading…</p>;
  if (docs.length === 0) {
    return <p className="text-xs text-slate-400">No documents yet.</p>;
  }

  return (
    <ul className="space-y-1">
      {docs.map((d) => (
        <li key={d.id} className="flex items-center gap-2 rounded px-2 py-1.5 hover:bg-slate-50">
          <FileText className="h-4 w-4 shrink-0 text-slate-400" />
          <span className="flex-1 truncate text-sm" title={d.filename}>
            {d.filename}
          </span>
          <span className="text-xs text-slate-400">{d.page_count}p</span>
          <button
            onClick={() => onDelete(d.id)}
            className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-600"
            aria-label={`Delete ${d.filename}`}
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/components/PdfDrop.tsx frontend/components/DocList.tsx
git commit -m "feat(frontend): add PdfDrop and DocList components"
```

---

### Task 36: CitationChip + Message components

**Files:**
- Create: `frontend/components/CitationChip.tsx`
- Create: `frontend/components/Message.tsx`

- [ ] **Step 1: Write `frontend/components/CitationChip.tsx`**

```tsx
"use client";

import { useState } from "react";
import { BookOpen } from "lucide-react";
import type { Citation } from "@/lib/types";
import { cn } from "@/lib/utils";

export function CitationChip({ citation }: { citation: Citation }) {
  const [expanded, setExpanded] = useState(false);
  const score = Math.round(citation.score * 100);
  return (
    <div className="inline-block">
      <button
        onClick={() => setExpanded((v) => !v)}
        className={cn(
          "inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2 py-0.5 text-xs text-slate-700 hover:bg-slate-50",
        )}
        title={`Confidence ${score}%`}
      >
        <BookOpen className="h-3 w-3 text-brand-500" />
        <span className="font-medium">{citation.source}</span>
        <span className="text-slate-400">p.{citation.page}</span>
        <span className="text-slate-400">·</span>
        <span className="text-slate-500">{score}%</span>
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Write `frontend/components/Message.tsx`**

```tsx
"use client";

import type { ChatMessage } from "@/lib/types";
import { CitationChip } from "./CitationChip";
import { cn } from "@/lib/utils";

export function Message({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex flex-col gap-2", isUser ? "items-end" : "items-start")}>
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap",
          isUser ? "bg-brand-500 text-white" : "bg-slate-100 text-slate-900",
        )}
      >
        {message.content || (message.streaming && (
          <span className="inline-flex gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400 typing-dot" style={{ animationDelay: "0s" }} />
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400 typing-dot" style={{ animationDelay: "0.2s" }} />
            <span className="h-1.5 w-1.5 rounded-full bg-slate-400 typing-dot" style={{ animationDelay: "0.4s" }} />
          </span>
        ))}
      </div>
      {message.error && <p className="text-xs text-red-600">{message.error}</p>}
      {!isUser && message.citations && message.citations.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {message.citations.map((c, i) => (
            <CitationChip key={`${c.source}-${c.page}-${i}`} citation={c} />
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/components/CitationChip.tsx frontend/components/Message.tsx
git commit -m "feat(frontend): add CitationChip and Message components"
```

---

### Task 37: Chat component

**Files:**
- Create: `frontend/components/Chat.tsx`

- [ ] **Step 1: Write `frontend/components/Chat.tsx`**

```tsx
"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { SendHorizontal } from "lucide-react";
import { streamQuery } from "@/lib/sse";
import type { ChatMessage, Citation } from "@/lib/types";
import { Message } from "./Message";

type Action =
  | { type: "APPEND"; msg: ChatMessage }
  | { type: "APPEND_DELTA"; id: string; text: string }
  | { type: "ADD_CITATION"; id: string; citation: Citation }
  | { type: "SET_CITATIONS"; id: string; citations: Citation[] }
  | { type: "FINISH"; id: string }
  | { type: "FAIL"; id: string; error: string }
  | { type: "HYDRATE"; messages: ChatMessage[] }
  | { type: "CLEAR" };

function reducer(state: ChatMessage[], action: Action): ChatMessage[] {
  switch (action.type) {
    case "APPEND":
      return [...state, action.msg];
    case "APPEND_DELTA":
      return state.map((m) =>
        m.id === action.id ? { ...m, content: m.content + action.text } : m,
      );
    case "ADD_CITATION":
      return state.map((m) =>
        m.id === action.id
          ? { ...m, citations: [...(m.citations ?? []), action.citation] }
          : m,
      );
    case "SET_CITATIONS":
      return state.map((m) => (m.id === action.id ? { ...m, citations: action.citations } : m));
    case "FINISH":
      return state.map((m) => (m.id === action.id ? { ...m, streaming: false } : m));
    case "FAIL":
      return state.map((m) =>
        m.id === action.id ? { ...m, streaming: false, error: action.error } : m,
      );
    case "HYDRATE":
      return action.messages;
    case "CLEAR":
      return [];
  }
}

const STORAGE_KEY = "docsage:chat:v1";

function makeId(): string {
  return Math.random().toString(36).slice(2, 10);
}

export function Chat() {
  const [messages, dispatch] = useReducer(reducer, []);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Hydrate from localStorage.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) dispatch({ type: "HYDRATE", messages: JSON.parse(raw) });
    } catch {
      // ignore
    }
  }, []);

  // Persist on every change.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
    } catch {
      // ignore (quota, disabled storage, etc.)
    }
  }, [messages]);

  // Auto-scroll on new content.
  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight });
  }, [messages]);

  const onSubmit = useCallback(async () => {
    const q = input.trim();
    if (!q || streaming) return;

    const userMsg: ChatMessage = { id: makeId(), role: "user", content: q };
    const assistantMsg: ChatMessage = {
      id: makeId(),
      role: "assistant",
      content: "",
      streaming: true,
      citations: [],
    };
    dispatch({ type: "APPEND", msg: userMsg });
    dispatch({ type: "APPEND", msg: assistantMsg });
    setInput("");
    setStreaming(true);

    const history = messages
      .filter((m) => !m.error)
      .map((m) => ({ role: m.role, content: m.content }));

    const abort = new AbortController();
    abortRef.current = abort;

    try {
      await streamQuery(
        { question: q, history },
        {
          onAnswerDelta: (text) =>
            dispatch({ type: "APPEND_DELTA", id: assistantMsg.id, text }),
          onCitation: (c) => dispatch({ type: "ADD_CITATION", id: assistantMsg.id, citation: c }),
          onDone: (verified) =>
            dispatch({ type: "SET_CITATIONS", id: assistantMsg.id, citations: verified }),
          onError: (err) => dispatch({ type: "FAIL", id: assistantMsg.id, error: err }),
        },
        abort.signal,
      );
    } catch (e) {
      dispatch({ type: "FAIL", id: assistantMsg.id, error: String(e) });
    } finally {
      dispatch({ type: "FINISH", id: assistantMsg.id });
      setStreaming(false);
      abortRef.current = null;
    }
  }, [input, streaming, messages]);

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void onSubmit();
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollerRef} className="flex-1 overflow-y-auto space-y-6 p-6">
        {messages.length === 0 && (
          <p className="text-center text-sm text-slate-400 pt-16">
            Upload a PDF and ask a question.
          </p>
        )}
        {messages.map((m) => (
          <Message key={m.id} message={m} />
        ))}
      </div>

      <div className="border-t bg-white p-4">
        <div className="flex items-end gap-2 rounded-2xl border border-slate-300 p-2 focus-within:border-brand-500">
          <textarea
            className="flex-1 resize-none bg-transparent px-2 py-1 text-sm outline-none"
            rows={1}
            placeholder="Ask anything about your documents…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            disabled={streaming}
          />
          <button
            onClick={onSubmit}
            disabled={streaming || !input.trim()}
            className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-brand-500 text-white disabled:opacity-40"
            aria-label="Send"
          >
            <SendHorizontal className="h-4 w-4" />
          </button>
        </div>
        <div className="mt-2 flex justify-between text-xs text-slate-400">
          <span>Enter to send · Shift+Enter for newline</span>
          {messages.length > 0 && (
            <button
              onClick={() => {
                dispatch({ type: "CLEAR" });
              }}
              className="hover:text-slate-700"
            >
              Clear chat
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/components/Chat.tsx
git commit -m "feat(frontend): add Chat component with SSE streaming and localStorage"
```

---

### Task 38: Main page

**Files:**
- Create: `frontend/app/page.tsx`

- [ ] **Step 1: Write `frontend/app/page.tsx`**

```tsx
"use client";

import { useState } from "react";
import { PdfDrop } from "@/components/PdfDrop";
import { DocList } from "@/components/DocList";
import { Chat } from "@/components/Chat";

export default function Home() {
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <main className="h-dvh flex">
      <aside className="w-72 shrink-0 border-r bg-slate-50/50 p-4 overflow-y-auto">
        <div className="mb-6">
          <h1 className="font-semibold text-lg">DocSage</h1>
          <p className="text-xs text-slate-500">
            RAG chatbot over your PDFs. Answers cite their sources.
          </p>
        </div>

        <div className="mb-4">
          <PdfDrop onUploaded={() => setRefreshKey((k) => k + 1)} />
        </div>

        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
            Documents
          </h2>
          <DocList refreshKey={refreshKey} />
        </div>
      </aside>

      <section className="flex-1">
        <Chat />
      </section>
    </main>
  );
}
```

- [ ] **Step 2: Copy `.env.example` to `.env.local`**

Run:
```bash
cp frontend/.env.example frontend/.env.local
```

- [ ] **Step 3: Verify frontend builds**

Run:
```bash
cd frontend && pnpm build
```

Expected: build succeeds with no TypeScript errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/app/page.tsx
git commit -m "feat(frontend): add main page with sidebar and chat layout"
```

---

### Task 39: Frontend test for Chat

**Files:**
- Create: `frontend/vitest.config.ts`
- Create: `frontend/tests/Chat.test.tsx`
- Create: `frontend/tests/setup.ts`

- [ ] **Step 1: Write `frontend/vitest.config.ts`**

```typescript
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./") },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    globals: true,
  },
});
```

- [ ] **Step 2: Write `frontend/tests/setup.ts`**

```typescript
import "@testing-library/jest-dom/vitest";
```

- [ ] **Step 3: Write `frontend/tests/Chat.test.tsx`**

```tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock the SSE client so the component can be driven by a test script.
vi.mock("@/lib/sse", () => ({
  streamQuery: vi.fn(async (_body, handlers) => {
    handlers.onAnswerDelta("Hello ");
    handlers.onAnswerDelta("world");
    handlers.onCitation({ source: "a.pdf", page: 1, score: 0.9 });
    handlers.onDone([{ source: "a.pdf", page: 1, score: 0.9 }]);
  }),
}));

import { Chat } from "@/components/Chat";

describe("Chat", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("streams an answer and renders a citation chip", async () => {
    render(<Chat />);
    const input = screen.getByPlaceholderText(/ask anything/i) as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "What does X do?" } });
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => {
      expect(screen.getByText("Hello world")).toBeInTheDocument();
    });
    expect(screen.getByText("a.pdf")).toBeInTheDocument();
    expect(screen.getByText(/p\.1/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 4: Run frontend test**

Run:
```bash
cd frontend && pnpm test --run
```

Expected: 1 test passes.

- [ ] **Step 5: Commit**

```bash
git add frontend/vitest.config.ts frontend/tests
git commit -m "test(frontend): add Chat component streaming test with Vitest"
```

---

## Phase H — CI, smoke, and README updates

### Task 40: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  backend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: backend/requirements-dev.txt
      - run: pip install -r requirements-dev.txt
      - run: ruff check app tests
      - run: pytest tests/unit -v
        env:
          OPENAI_API_KEY: sk-test
          DATABASE_URL: postgresql+asyncpg://docsage:docsage@localhost:5432/docsage

  frontend:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with:
          version: 9
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: pnpm
          cache-dependency-path: frontend/pnpm-lock.yaml
      - run: pnpm install --frozen-lockfile
      - run: pnpm lint
      - run: pnpm build
      - run: pnpm test --run

  secrets-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add backend, frontend, and secrets-scan workflows"
```

---

### Task 41: README updates

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README and update the Quick start section**

Replace the existing "Quick start" section (bash block starting with `git clone`) with:

```markdown
## Quick start

**One-command setup** (requires Docker, Python 3.11, Node 20, and pnpm):

```bash
git clone https://github.com/bright-nwokoro/docsage-rag
cd docsage-rag

cp backend/.env.example backend/.env       # fill OPENAI_API_KEY
cp frontend/.env.example frontend/.env.local

make install     # backend pip + frontend pnpm
make up          # start Postgres with pgvector
make migrate     # apply schema
make dev         # run backend (8000) + frontend (3000)
```

Open http://localhost:3000, upload a PDF, start asking questions.

**Under the hood** (what `make dev` actually does):

```bash
# Backend
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000     # http://localhost:8000

# Frontend (separate terminal)
cd frontend
pnpm dev                                      # http://localhost:3000
```
```

- [ ] **Step 2: Add a note about partial-JSON streaming in the "How it works" section**

Find the "### Generation with forced citations" heading, and after the code block, replace the paragraph that starts "The backend then verifies..." with:

```markdown
The backend streams the partial JSON from OpenAI as it arrives: every time the
`answer` field grows, we emit an SSE `answer_delta` event; every time a complete
citation entry appears in `citations[]`, we emit an SSE `citation` event. When
the stream ends, the backend verifies each cited `(source, page)` pair actually
exists in the retrieved chunks and emits a final `done` event with only the
verified set — the frontend reconciles and drops any previously-shown chips that
failed verification. This makes hallucinated citations structurally impossible
while keeping the UX fully streamed.
```

- [ ] **Step 3: Update the Project structure block**

Replace the existing project-structure tree with:

```markdown
## Project structure

```
docsage-rag/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app factory
│   │   ├── config.py               # pydantic-settings
│   │   ├── db.py                   # async SQLAlchemy engine
│   │   ├── models.py               # Doc + Chunk models (pgvector)
│   │   ├── schemas.py              # pydantic API schemas
│   │   ├── routes/
│   │   │   ├── ingest.py           # POST /ingest
│   │   │   ├── query.py            # POST /query (SSE)
│   │   │   ├── docs.py             # GET /docs, DELETE /docs/{id}
│   │   │   └── health.py           # GET /health
│   │   └── core/
│   │       ├── chunker.py          # token-budgeted sentence-aware chunker
│   │       ├── embeddings.py       # batched OpenAI embeddings with retry
│   │       ├── retriever.py        # parallel vector+keyword + RRF
│   │       ├── citations.py        # schema + prompt + verifier
│   │       ├── generator.py        # streaming + partial JSON parsing
│   │       └── openai_client.py    # cached async client factory
│   ├── alembic/                    # schema migrations
│   ├── tests/{unit,integration,smoke,eval}/
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   └── api/                    # thin proxy to backend
│   ├── components/
│   │   ├── Chat.tsx
│   │   ├── Message.tsx
│   │   ├── CitationChip.tsx
│   │   ├── PdfDrop.tsx
│   │   └── DocList.tsx
│   ├── lib/{sse.ts,types.ts,utils.ts}
│   └── package.json
├── docker-compose.yml              # Postgres + pgvector
├── Makefile                        # install / up / migrate / dev / test
├── .github/workflows/ci.yml
├── docs/
│   ├── DEPLOY.md
│   └── superpowers/{specs,plans}/
└── README.md
```
```

- [ ] **Step 4: Add a note under the roadmap about deferred seed mode**

In the "Roadmap" section, add this item near the top of the list:

```markdown
- [ ] Seed mode: ingest preloaded Next.js / Stripe / Kubernetes docs on deploy
```

(Move the existing streaming-citations item below it if present.)

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update README with Makefile quick-start and streaming details"
```

---

### Task 42: DEPLOY stub

**Files:**
- Create: `docs/DEPLOY.md`

- [ ] **Step 1: Write `docs/DEPLOY.md`**

```markdown
# Deployment

> **Status:** placeholder — full deploy guide is a future roadmap item.

## Target

- **Frontend:** Vercel (Next.js native).
- **Backend + Postgres:** Railway (FastAPI service + managed Postgres plugin with pgvector).

## Outline

1. **Railway: Postgres.** Provision the Postgres plugin. Connect, run `CREATE EXTENSION IF NOT EXISTS vector;`. Copy the connection string.
2. **Railway: backend service.** Deploy from GitHub. Set env vars from `backend/.env.example`: `OPENAI_API_KEY`, `DATABASE_URL` (the Railway Postgres URL), `ALLOWED_ORIGINS=https://your-frontend.vercel.app`. Build command: `pip install -r requirements.txt`. Start command: `alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. **Vercel: frontend.** Import the GitHub repo, set root to `frontend/`. Add env var `NEXT_PUBLIC_API_URL=https://your-backend.up.railway.app`. Deploy.

Detailed walkthrough (screenshots, rollback, cost monitoring, rate limiting) to come.
```

- [ ] **Step 2: Commit**

```bash
git add docs/DEPLOY.md
git commit -m "docs: add DEPLOY.md placeholder"
```

---

### Task 43: Seed script placeholder

**Files:**
- Create: `backend/scripts/seed.py`
- Create: `backend/scripts/__init__.py`

- [ ] **Step 1: Create `backend/scripts/__init__.py`**

```python
```

- [ ] **Step 2: Write `backend/scripts/seed.py`**

```python
"""Seed script placeholder.

Seed mode (preloading Next.js / Stripe / Kubernetes docs) is deferred. When
implemented, this script will iterate PDFs in ./seeds and call the ingest
pipeline for each one. Today it is a no-op.
"""


def main() -> None:
    print("seed mode deferred — see docs/superpowers/specs/2026-04-21-docsage-rag-design.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add backend/scripts
git commit -m "chore(backend): add seed.py placeholder for deferred seed mode"
```

---

## Phase I — End-to-end verification

### Task 44: Full local smoke run

**Files:** none modified — verification only.

- [ ] **Step 1: Clean slate**

Run:
```bash
make clean
make up
make migrate
```

Expected: `make up` reports healthy; `make migrate` runs migration `0001`.

- [ ] **Step 2: Start the backend in the background**

Run:
```bash
cd backend && source .venv/bin/activate && uvicorn app.main:app --port 8000 &
```

Wait a few seconds, then verify:
```bash
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok"}`.

- [ ] **Step 3: Upload a PDF**

Run:
```bash
curl -s -X POST http://localhost:8000/ingest \
  -F "file=@backend/tests/fixtures/sample.pdf"
```

Expected: JSON like `{"doc_id":"...","filename":"sample.pdf","page_count":2,"chunk_count":N}`.

- [ ] **Step 4: Query via SSE**

Run:
```bash
curl -N -s -X POST http://localhost:8000/query \
  -H "content-type: application/json" \
  -d '{"question":"What does pgvector enable?","history":[]}'
```

Expected: a stream of `event: answer_delta`, `event: citation`, and a final `event: done` with `verified_citations` — the answer should reference pgvector, citation should list `sample.pdf`.

- [ ] **Step 5: Stop the backend**

Run:
```bash
pkill -f "uvicorn app.main:app" || true
```

- [ ] **Step 6: Run frontend**

Run:
```bash
cd frontend && pnpm dev
```

Open `http://localhost:3000` in a browser.

- [ ] **Step 7: Manual UI checks**

In the browser:
- Upload `backend/tests/fixtures/sample.pdf` via the dropzone. Expect it to appear in the docs list.
- Ask: "What does pgvector enable?". Expect streamed answer with a citation chip showing `sample.pdf p.X`.
- Refresh the page. Expect the conversation to persist.
- Click Clear chat. Expect the conversation to reset.
- Delete the doc via the trash icon. Expect the list to empty.

- [ ] **Step 8: Stop everything and commit the verification note**

Run:
```bash
pkill -f "next dev" || true
```

No code changes — nothing to commit for this task.

---

### Task 45: Run the full test suite

**Files:** none.

- [ ] **Step 1: Run backend unit + integration tests**

Run:
```bash
docker compose up -d --wait
cd backend && source .venv/bin/activate
alembic upgrade head
pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run frontend tests**

Run:
```bash
cd frontend && pnpm test --run
```

Expected: all tests pass.

- [ ] **Step 3: Run lint**

Run:
```bash
make lint
```

Expected: no errors.

- [ ] **Step 4: Final commit (if needed)**

If any fixes were required, commit them with descriptive messages. Otherwise, no action.

---

## Summary

This plan builds DocSage end-to-end locally in 45 sequenced tasks across nine phases:

- **A (1–4):** repo bootstrap, docker-compose, Makefile, `.gitignore`.
- **B (5–10):** backend skeleton, config, DB session, models, Alembic init, initial migration.
- **C (11–16):** pure core modules (chunker, citations, RRF) with full TDD.
- **D (17–21):** core modules with external deps (embeddings, OpenAI client, retriever orchestration, generator).
- **E (22–26):** schemas, FastAPI app, routes (health, docs, ingest, query SSE).
- **F (27–30):** integration tests, smoke test, eval scaffold.
- **G (31–39):** Next.js frontend — scaffold, API proxy, SSE client, components, Chat, main page, test.
- **H (40–43):** CI workflow, README updates, DEPLOY stub, seed placeholder.
- **I (44–45):** end-to-end verification + full test suite run.

Frequent commits, TDD on pure logic, integration tests for the external-dep boundaries, and an explicit manual-verification pass at the end.
