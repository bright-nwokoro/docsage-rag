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
