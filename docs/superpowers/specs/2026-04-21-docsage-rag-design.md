# DocSage — Implementation Design

**Date:** 2026-04-21
**Status:** Approved — ready for implementation plan
**Source spec:** `[README.md](../../../README.md)`

The [README](../../../README.md) is the user-facing product spec. This document resolves the implementation decisions the README leaves open and locks in the defaults we will build against.

## Goal & scope

Build DocSage end-to-end locally so `make up && make migrate && make dev` gives a working RAG chatbot at `localhost:3000`. Public repo, portfolio quality. No deployment in this pass — code is structured to deploy later without refactoring.

**In scope:**

- FastAPI backend: PDF ingestion, hybrid retrieval, structured-output generation with partial-JSON streaming.
- Next.js 14 frontend: upload UI, streaming chat with live citation chips, doc list with delete.
- Postgres 16 + pgvector 0.7 via Docker Compose.
- Alembic migrations, pytest unit + integration tests, GitHub Actions CI, Makefile for one-command dev loop.

**Deferred (roadmap, not blockers):**

- Deployment to Vercel + Railway (roadmap item, needs its own spec).
- Seed mode preloading (Next.js / Stripe / Kubernetes docs).
- Multi-tenant isolation, rate limiting, auth.
- Re-ranker stage, MMR diversity, non-PDF inputs, Ollama mode.
- Full evaluation harness (scaffold only in this pass).

## Resolved decisions


| Question                      | Decision                                                               |
| ----------------------------- | ---------------------------------------------------------------------- |
| Build target                  | Full end-to-end working locally (backend + frontend + Docker Postgres) |
| OpenAI key                    | Real API calls during development                                      |
| Postgres setup                | Docker Compose with `pgvector/pgvector:pg16`                           |
| Seed mode                     | Defer — placeholder script only                                        |
| Streaming + structured output | Partial-JSON streaming, server parses incrementally, emits SSE events  |
| Workspace model               | Single pool, no auth, no per-user isolation                            |
| Chat memory                   | Stateless backend, client passes history                               |
| Package managers              | `pip` + `requirements.txt` (backend), `pnpm` (frontend)                |
| Streaming transport           | SSE (Server-Sent Events), not WebSockets                               |


## Repo layout

```
docsage-rag/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app factory, CORS, lifespan
│   │   ├── routes/
│   │   │   ├── ingest.py           # POST /ingest
│   │   │   ├── query.py            # POST /query (SSE)
│   │   │   └── docs.py             # GET /docs, DELETE /docs/{id}
│   │   ├── core/
│   │   │   ├── chunker.py
│   │   │   ├── embeddings.py
│   │   │   ├── retriever.py
│   │   │   ├── citations.py
│   │   │   └── generator.py
│   │   ├── db.py                   # async SQLAlchemy engine + session
│   │   ├── models.py
│   │   ├── config.py               # pydantic-settings
│   │   └── schemas.py              # pydantic request/response models
│   ├── alembic/
│   │   ├── env.py
│   │   └── versions/
│   │       └── 0001_initial.py
│   ├── tests/
│   │   ├── unit/
│   │   │   ├── test_chunker.py
│   │   │   ├── test_rrf.py
│   │   │   └── test_citations.py
│   │   ├── integration/
│   │   │   ├── test_ingest_flow.py
│   │   │   └── test_query_flow.py
│   │   ├── smoke/
│   │   │   └── test_real_openai.py  # @pytest.mark.smoke, opt-in
│   │   ├── eval/
│   │   │   ├── golden.yaml
│   │   │   └── run_eval.py
│   │   ├── fixtures/                # sample PDFs for tests
│   │   └── conftest.py
│   ├── scripts/
│   │   └── seed.py                  # placeholder no-op
│   ├── pyproject.toml               # ruff config
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                 # upload + chat
│   │   ├── globals.css
│   │   └── api/
│   │       ├── ingest/route.ts
│   │       ├── query/route.ts
│   │       └── docs/route.ts
│   ├── components/
│   │   ├── Chat.tsx
│   │   ├── Message.tsx
│   │   ├── CitationChip.tsx
│   │   ├── PdfDrop.tsx
│   │   ├── DocList.tsx
│   │   └── ui/                      # shadcn/ui generated components
│   ├── lib/
│   │   ├── api.ts                   # typed client
│   │   └── sse.ts                   # @microsoft/fetch-event-source wrapper
│   ├── tests/
│   │   └── Chat.test.tsx
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   └── .env.example
├── docker-compose.yml
├── Makefile
├── .github/workflows/ci.yml
├── .gitignore
├── .python-version                  # 3.11
├── docs/
│   ├── DEPLOY.md                    # stub for roadmap
│   └── superpowers/specs/
│       └── 2026-04-21-docsage-rag-design.md
└── README.md
```

## Backend architecture

Every core module has either zero external dependencies (pure) or exactly one (OpenAI or DB). Route handlers are pure orchestration.

### Modules


| Module               | Responsibility                                                                         | External deps                    |
| -------------------- | -------------------------------------------------------------------------------------- | -------------------------------- |
| `core/chunker.py`    | `chunk_pdf(path) -> list[Chunk]`. Sentence-aware, token-budgeted.                      | None (pure)                      |
| `core/embeddings.py` | `embed_batch(list[str]) -> list[list[float]]`. Batched, retried.                       | OpenAI                           |
| `core/retriever.py`  | `retrieve(query, top_k) -> list[RetrievedChunk]`. Parallel vector+keyword, RRF fusion. | DB, OpenAI (for query embedding) |
| `core/citations.py`  | Schema, system prompt, post-hoc verifier.                                              | None (pure)                      |
| `core/generator.py`  | Streams OpenAI completion, parses partial JSON, yields SSE events.                     | OpenAI                           |
| `routes/ingest.py`   | Orchestrates chunk → embed → DB insert in one transaction.                             | DB, core modules                 |
| `routes/query.py`    | Orchestrates retrieve → generate, streams SSE.                                         | DB, core modules                 |
| `routes/docs.py`     | List/delete operations.                                                                | DB                               |
| `models.py`          | SQLAlchemy 2.0 models for `docs` and `chunks`.                                         | DB                               |


### Database schema

Extends the README's schema with the columns the `docs` table needs (README underspecifies it):

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE docs (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename       TEXT NOT NULL,
  page_count     INT NOT NULL,
  uploaded_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
  id             BIGSERIAL PRIMARY KEY,
  doc_id         UUID NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
  page_number    INT NOT NULL,
  chunk_index    INT NOT NULL,
  content        TEXT NOT NULL,
  content_tsv    TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  embedding      VECTOR(1536) NOT NULL
);

CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX chunks_tsv_idx       ON chunks USING GIN (content_tsv);
CREATE INDEX chunks_doc_id_idx    ON chunks (doc_id);
```

`ON DELETE CASCADE` on `chunks.doc_id` so `DELETE /docs/{id}` is a single statement.

### API surface


| Method   | Path         | Request                                       | Response                                      |
| -------- | ------------ | --------------------------------------------- | --------------------------------------------- |
| `POST`   | `/ingest`    | `multipart/form-data` with `file`             | `{doc_id, filename, chunk_count, page_count}` |
| `POST`   | `/query`     | `{question: str, history: [{role, content}]}` | `text/event-stream`                           |
| `GET`    | `/docs`      | —                                             | `[{id, filename, page_count, uploaded_at}]`   |
| `DELETE` | `/docs/{id}` | —                                             | `204`                                         |
| `GET`    | `/health`    | —                                             | `{status: "ok"}`                              |


## Ingestion pipeline

1. **Parse.** `unstructured.partition.pdf.partition_pdf(strategy="fast")` — returns elements annotated with `page_number`. Fast strategy (no OCR) is sufficient for text PDFs.
2. **Sentence-split.** Concatenate elements per page, split into sentences with a lightweight regex (avoids NLTK/spacy dependency). Preserves page boundaries.
3. **Pack into chunks.** Greedy pack sentences up to `MAX_CHUNK_TOKENS` (512) measured by tiktoken `cl100k_base`. When a chunk closes, seed the next with the tail of the previous at `CHUNK_OVERLAP_TOKENS` (64). A chunk spanning two pages inherits the starting page.
4. **Embed.** `embed_batch(chunks)` calls OpenAI `text-embedding-3-small` in batches of 100 with tenacity-based exponential backoff on 429/5xx.
5. **Insert.** `docs` row + all `chunks` rows in a single SQLAlchemy transaction. If embedding fails partway, the whole thing rolls back.

**Why a custom packer over `unstructured.chunk_by_title`:** deterministic token counts, ~40 lines of code, unit-testable without heavy deps.

**Idempotency:** out of scope. Uploading the same PDF twice produces two `docs` rows. `DELETE /docs/{id}` is the escape hatch.

## Retrieval

Two queries run concurrently via `asyncio.gather`, each returning top-20 candidates:

**Vector:**

```sql
SELECT id, doc_id, page_number, content, embedding <=> :q_emb AS distance
FROM chunks
ORDER BY embedding <=> :q_emb
LIMIT 20;
```

**Keyword:**

```sql
SELECT id, doc_id, page_number, content,
       ts_rank(content_tsv, plainto_tsquery('english', :q)) AS rank
FROM chunks
WHERE content_tsv @@ plainto_tsquery('english', :q)
ORDER BY rank DESC
LIMIT 20;
```

**Fusion (RRF, `k=60`):**

```
score(chunk) = Σ over rankers:  1 / (k + rank_in_that_ranker)
```

Sort by combined score, take top `TOP_K` (5, env-configurable). Chunks are joined with their parent `docs.filename` for the prompt.

**Why RRF over weighted-sum:** cosine distance (bounded [0,2]) and `ts_rank` (unbounded, corpus-dependent) are not comparable scales. RRF uses only rank position and is the standard in Elasticsearch / Vespa / Weaviate.

**Index tuning note:** `ivfflat` `lists = 100` is the starting value (pgvector rule of thumb: `lists ≈ sqrt(rows)`). Documented as "revisit once corpus > 10k chunks."

## Generation with streaming + forced citations

### The call

```python
async for delta in openai_client.chat.completions.create(
    model=settings.OPENAI_CHAT_MODEL,
    messages=[
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": build_prompt(question, chunks)},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "rag_answer",
            "strict": True,
            "schema": RAG_ANSWER_SCHEMA,
        },
    },
    stream=True,
):
    ...
```

`strict: true` guarantees schema conformance on the final output. `stream=True` gives token-by-token deltas.

### Server-side incremental parsing

Deltas form a growing JSON string that is almost-always-invalid until the last token. Parsed incrementally with a partial-JSON parser (pick the maintained package at implementation: `partial-json-parser` or equivalent). On each parser advance:

- `**answer` grew:** emit `event: answer_delta` with the new character diff.
- **New complete citation in `citations[]`:** emit `event: citation` with `{source, page, score}`.
- **Stream end:** run the verifier (drop any `(source, page)` not in the retrieved set), emit `event: done` with `{verified_citations: [...]}`.

Frontend reconciles: any previously-streamed citation not in `verified_citations` is removed.

### Prompt structure

```
SYSTEM:
You answer strictly from the provided context chunks. Every claim must cite the
chunk it came from. If the context doesn't contain the answer, say so explicitly
and return an empty citations array. Never invent sources or page numbers.

USER:
Question: {question}

Context:
[chunk 1 | source="contracts.pdf" page=4]
<content>

[chunk 2 | source="manual.pdf" page=12]
<content>
...
```

The model can only cite `(source, page)` pairs visible in the context block — this is the mechanical guarantee behind "no hallucinated citations." The post-hoc verifier is belt-and-suspenders.

### Multi-turn

Last 8 `{role, content}` turns from the client are inserted between the system prompt and the current user message. Bound prevents token creep over long conversations.

### Error handling

If OpenAI errors or the stream aborts mid-answer, emit `event: error` with `{message}`. Frontend shows an inline error chip and retains the partial answer.

## Frontend architecture

Single-page app, App Router, no auth.

### Components


| Component          | Responsibility                                                                                           |
| ------------------ | -------------------------------------------------------------------------------------------------------- |
| `Chat.tsx`         | Owns conversation state (reducer), opens SSE connection on submit, dispatches events to update messages. |
| `Message.tsx`      | One chat bubble. Renders streaming answer text, citation chips, error state.                             |
| `CitationChip.tsx` | Pill showing `[{source} p.{page}]`. Hover reveals score. Click expands chunk content inline.             |
| `PdfDrop.tsx`      | react-dropzone wrapper. Multipart upload to `/api/ingest`. Per-file progress + success/error.            |
| `DocList.tsx`      | Lists uploaded docs, delete button per row. Polls `/api/docs` on mount + after uploads.                  |


### API proxy layer

Thin Next.js route handlers under `app/api/` forward to the FastAPI backend:

- Keeps backend URL behind one env var (`NEXT_PUBLIC_API_URL`).
- Future: edge-layer rate limiting / auth without backend changes.
- SSE: `new Response(upstream.body)` pipes through natively on edge runtime.

### Styling

Tailwind + `shadcn/ui`. Two-column layout: 280px docs panel left, chat fills right. Mobile: docs panel collapses behind a sheet.

### State

React `useReducer` for chat. Persisted to `localStorage` on every update so refresh keeps the conversation. No Zustand/Redux.

### SSE client

`@microsoft/fetch-event-source` — supports POST (native `EventSource` does not), MIT-licensed, widely used.

## Dev infrastructure

### `docker-compose.yml`

Single service: `pgvector/pgvector:pg16`. Healthcheck on `pg_isready`. Named volume for persistence. Backend and frontend run on the host for fast iteration — dockerize later for deploy.

### `Makefile`

```
make up        # docker compose up -d + wait for healthcheck
make migrate   # alembic upgrade head
make dev       # uvicorn --reload + pnpm dev in parallel
make test      # pytest + pnpm test
make eval      # run golden-set eval harness (scaffold)
make seed      # placeholder no-op
make clean     # docker compose down -v
```

### Environment

`backend/.env.example` and `frontend/.env.example` committed. First-run: `cp .env.example .env && <fill OPENAI_API_KEY>`.

### `.gitignore`

Covers `.env`, `.env.local`, `*.pyc`, `__pycache__`, `.venv`, `node_modules`, `.next`, `dist`, `.pytest_cache`, `.ruff_cache`.

### CI (`.github/workflows/ci.yml`)

Runs on PR:

- **Backend:** Python 3.11, `ruff check`, `pytest tests/unit` (no OpenAI calls).
- **Frontend:** Node 20, `pnpm install --frozen-lockfile`, `pnpm lint`, `pnpm build`.
- **Secrets scan:** `gitleaks-action`.

## Testing approach

### Unit (fast, no external deps)

- `test_chunker.py` — synthetic multi-page text, assert chunk count, token counts, overlap, page propagation.
- `test_rrf.py` — two synthetic ranked lists, assert fused order matches hand-computed RRF scores.
- `test_citations.py` — retrieved chunk set + model output with one valid + one invalid citation, assert verifier drops the invalid one.

### Integration (real Postgres via Compose, mocked OpenAI)

- `test_ingest_flow.py` — upload fixture PDF, assert `docs`/`chunks` rows, 1536-dim embeddings. Fake OpenAI returns deterministic hash-based vectors.
- `test_query_flow.py` — pre-seed DB, call `/query`, assert SSE event sequence. Fake OpenAI returns a fixed streamed response.

### Smoke (real OpenAI, opt-in)

- `tests/smoke/test_real_openai.py` — `@pytest.mark.smoke`, skipped by default. Run locally before releasing.

### Eval harness scaffold

- `tests/eval/golden.yaml` — empty list + schema documentation.
- `tests/eval/run_eval.py` — loads golden set, runs each question, scores answer-presence + citation precision/recall.
- Wired into `make eval`, not into CI.

### Frontend tests

- One Vitest + RTL test on `Chat.tsx` verifying SSE event reduction into messages. No Playwright.

## README updates

The README will be amended to reflect:

- Quick-start uses `make up && make migrate && make dev` (replaces the manual pip/pnpm sequence for first-run, keeps the manual sequence as "what it does under the hood").
- Acknowledgement that seed mode is deferred.
- Note that the pipeline uses partial-JSON streaming (not two-phase) — worth calling out since it's the most technically interesting detail.

No other README changes — the product description, architecture diagram, and roadmap stay as they are.

## Out of scope (explicit)

- Authentication / multi-user.
- Rate limiting / cost caps (the live-demo concern is a deployment concern, not a code concern).
- Non-PDF inputs (roadmap).
- Re-ranker stage (roadmap).
- Ollama / self-hosted mode (roadmap).
- Real seed docs (deferred; script placeholder only).
- Production Dockerfile for the backend (deploy-time concern).

