# DocSage

> A RAG chatbot over any set of PDFs. Ask questions, get answers with inline citations — never hallucinated references.

**🔗 Live demo:** [https://docsage.brightnwokoro.dev](https://docsage.brightnwokoro.dev)
**👤 Built by:** [Bright Nwokoro](https://brightnwokoro.dev) · [hello@brightnwokoro.dev](mailto:hello@brightnwokoro.dev)

DocSage demo

---

## Why this exists

Founders with large document sets — contracts, manuals, policies, API docs, research — want AI that cites its sources, not hallucinates them. Generic "chat with PDF" tools are fast to demo but slow to trust: they invent page numbers, cite the wrong documents, or silently drop context when a question spans multiple chunks.

DocSage is the opinionated, production version. Every answer is forced through a strict citation schema at the model layer. Retrieval is hybrid (semantic + keyword) so recall holds up across document types — code, prose, tables, bullet lists. Infrastructure is one Postgres instance with pgvector, so cost stays under a dollar a day for small-to-medium corpora.

## What it does

- Upload 1–N PDFs via the web UI
- Automatic semantic chunking (sentence-boundary aware) + OpenAI embeddings
- Hybrid retrieval combining vector similarity and Postgres full-text search
- Every answer returns a structured `{ answer, citations[] }` payload — no free-form hallucinations
- Inline source chunks rendered under each answer with page number and confidence score
- Streaming responses
- Seed mode: ships preloaded with Next.js, Stripe, and Kubernetes documentation so visitors can try it without uploading anything

## Architecture

```
  ┌──────────┐        ┌────────────┐        ┌──────────────┐
  │  Next.js │───────▶│  FastAPI   │───────▶│  OpenAI API  │
  │   (UI)   │  JSON  │  backend   │  REST  │  chat+embed  │
  └──────────┘        └─────┬──────┘        └──────────────┘
                            │
                     ┌──────▼──────┐
                     │  Postgres   │
                     │  + pgvector │
                     └─────────────┘
```

- **Frontend** (Next.js on Vercel) renders the upload UI, chat, and streaming citations. It's a thin shell — all intelligence lives in the backend.
- **Backend** (FastAPI on Railway) owns ingestion (PDF → chunks → embeddings) and query-time retrieval + generation. Stateless; horizontal-scales cleanly.
- **Postgres + pgvector** stores chunks, embeddings, and document metadata in one database. No separate vector store to pay for or keep in sync.
- **OpenAI** provides `text-embedding-3-small` for ingestion and `gpt-4o-mini` for generation. Both are cheap enough to keep per-query cost under $0.002.

## Stack


| Layer            | Tech                                                  |
| ---------------- | ----------------------------------------------------- |
| Frontend         | Next.js 14, React 18, TypeScript, Tailwind, shadcn/ui |
| Backend          | FastAPI 0.110, Python 3.11, uvicorn                   |
| LLM — generation | OpenAI `gpt-4o-mini` (with structured output)         |
| LLM — embeddings | OpenAI `text-embedding-3-small`                       |
| Vector store     | Postgres 16 + pgvector 0.7                            |
| PDF parsing      | `pypdf` + `unstructured`                              |
| Retrieval        | pgvector cosine + Postgres `tsvector` full-text       |
| Hosting          | Vercel (frontend), Railway (backend + Postgres)       |


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

Open [http://localhost:3000](http://localhost:3000), upload a PDF, start asking questions.

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

## Environment variables

**Backend (`backend/.env`)**

```bash
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/docsage
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
MAX_CHUNK_TOKENS=512
CHUNK_OVERLAP_TOKENS=64
TOP_K=5
```

**Frontend (`frontend/.env.local`)**

```bash
API_URL=http://localhost:8000
```

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

## How it works

### Ingestion pipeline

1. User uploads a PDF via the frontend dropzone.
2. Backend extracts text per page (`pypdf`), then runs a semantic chunker that respects sentence boundaries — default ~400–500 tokens per chunk with ~64 token overlap.
3. Each chunk is embedded with `text-embedding-3-small` (1536-dimensional vectors).
4. Chunks are stored in Postgres:
  ```sql
   CREATE TABLE chunks (
     id              BIGSERIAL PRIMARY KEY,
     doc_id          UUID NOT NULL REFERENCES docs(id),
     page_number     INT NOT NULL,
     chunk_index     INT NOT NULL,
     content         TEXT NOT NULL,
     content_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
     embedding       VECTOR(1536) NOT NULL
   );
   CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
   CREATE INDEX ON chunks USING GIN (content_tsv);
  ```

### Retrieval

Each question runs both retrievers in parallel, then fuses:

- **Vector:** `ORDER BY embedding <=> $embedding` (cosine distance, top-k)
- **Keyword:** `ts_rank(content_tsv, plainto_tsquery($query))` (top-k)
- **Fusion:** Reciprocal Rank Fusion (RRF) with `k = 60` combines both rankings. Chunks that show up high in both win.

The top 5 fused chunks go into the prompt context.

### Generation with forced citations

The LLM call uses OpenAI's **structured output** mode so the model cannot return an answer without citations:

```python
response = openai.chat.completions.create(
    model=OPENAI_CHAT_MODEL,
    messages=[
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user",   "content": build_prompt(question, chunks)},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "rag_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "page":   {"type": "integer"},
                                "score":  {"type": "number"},
                            },
                            "required": ["source", "page", "score"],
                        },
                    },
                },
                "required": ["answer", "citations"],
            },
        },
    },
)
```

The backend streams the partial JSON from OpenAI as it arrives: every time the
`answer` field grows, we emit an SSE `answer_delta` event; every time a complete
citation entry appears in `citations[]`, we emit an SSE `citation` event. When
the stream ends, the backend verifies each cited `(source, page)` pair actually
exists in the retrieved chunks and emits a final `done` event with only the
verified set — the frontend reconciles and drops any previously-shown chips that
failed verification. This makes hallucinated citations structurally impossible
while keeping the UX fully streamed.

## Cost profile

Measured on the preloaded demo corpus (~300 pages across three docs):


| Operation           | Cost (USD)             |
| ------------------- | ---------------------- |
| Ingestion           | ~$0.01 per 100 pages   |
| Query — embedding   | ~$0.00002 per question |
| Query — chat        | ~$0.0018 per answer    |
| **Per query total** | **~$0.002**            |


Postgres storage is negligible (~50MB per 1,000 pages including embeddings).

## Deployment

**Vercel (frontend):** one-click deploy from GitHub. Set `API_URL` to the Railway backend URL.

**Railway (backend + db):** provision a Postgres plugin, enable the `pgvector` extension, set env vars, deploy from the GitHub repo. See `[docs/DEPLOY.md](docs/DEPLOY.md)` for the step-by-step.

## Roadmap

- Seed mode: ingest preloaded Next.js / Stripe / Kubernetes docs on deploy
- Support for DOCX, TXT, MD, and HTML inputs
- Multi-tenant isolation (one pgvector schema per client workspace)
- Evaluation harness (RAGAS + custom golden set)
- Re-ranker stage (Cohere Rerank or a local cross-encoder)
- Self-hosted Ollama + pgvector mode for air-gapped clients

## Contributing

PRs welcome. Please include a test for retrieval logic changes — the evaluation harness runs against a small golden set in `backend/tests/eval/`.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Freelance AI engineering — RAG, chat widgets, AI copilots, end-to-end.
**Email:** [hello@brightnwokoro.dev](mailto:hello@brightnwokoro.dev)
**Portfolio:** [https://brightnwokoro.dev](https://brightnwokoro.dev)
**Book a call:** [https://calendly.com/brightnwokoro/30min](https://calendly.com/brightnwokoro/30min)