# Deployment

> **Status:** placeholder — full deploy guide is a future roadmap item.

## Target

- **Frontend:** Vercel (Next.js native).
- **Backend + Postgres:** Railway (FastAPI service + managed Postgres plugin with pgvector).

## Outline

1. **Railway: Postgres.** Provision the Postgres plugin. Connect, run `CREATE EXTENSION IF NOT EXISTS vector;`. Copy the connection string.
2. **Railway: backend service.** Deploy from GitHub. Set env vars from `backend/.env.example`: `OPENAI_API_KEY`, `DATABASE_URL` (the Railway Postgres URL), `ALLOWED_ORIGINS=https://your-frontend.vercel.app`. Build command: `pip install -r requirements.txt`. Start command: `alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
3. **Vercel: frontend.** Import the GitHub repo, set root to `frontend/`. Add env var `API_URL=https://your-backend.up.railway.app`. Deploy.

Detailed walkthrough (screenshots, rollback, cost monitoring, rate limiting) to come.
