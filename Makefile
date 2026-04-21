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
