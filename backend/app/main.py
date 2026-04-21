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
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(docs.router, prefix="/docs", tags=["docs"])
    app.include_router(ingest.router, tags=["ingest"])
    app.include_router(query.router, tags=["query"])

    return app


app = create_app()
