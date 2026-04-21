import asyncio
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

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


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Session-scoped event loop so the app's SQLAlchemy connection pool is
    created and reused within a single loop for all async tests.  Without this
    each test would get its own loop, causing 'Future attached to a different
    loop' errors from asyncpg's connection pool.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()
