import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import NullPool, text
from sqlalchemy.ext.asyncio import create_async_engine

# Ensure env is set before app imports.
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://docsage:docsage@localhost:5432/docsage",
)


@pytest.fixture(scope="session", autouse=True)
def ensure_db_migrated() -> Iterator[None]:
    """Run `alembic upgrade head` once per session.

    Resolves the alembic CLI from the same venv as the running Python so the
    fixture works whether or not the user has the venv activated in their shell.
    """
    alembic_bin = os.path.join(os.path.dirname(sys.executable), "alembic")
    subprocess.run([alembic_bin, "upgrade", "head"], check=True, cwd=".")
    yield


@pytest_asyncio.fixture
async def clean_db() -> AsyncIterator[None]:
    """Truncate docs + chunks before each integration test for isolation."""
    # NullPool prevents asyncpg connections from being held across event-loop
    # boundaries (each pytest-asyncio test gets its own loop).
    engine = create_async_engine(os.environ["DATABASE_URL"], poolclass=NullPool)
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
