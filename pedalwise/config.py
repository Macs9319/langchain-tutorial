"""Settings, read once from the environment (and .env)."""

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    fallback_model: str = os.getenv("FALLBACK_MODEL", "gpt-5-nano")
    data_dir: Path = ROOT / "data"
    index_dir: Path = ROOT / ".rag_index"
    db_path: str = os.getenv("PEDALWISE_DB", str(ROOT / "pedalwise.sqlite"))
    # Staff must send this token to approve refunds. If it isn't set, a random
    # one is made at startup and printed, so the staff page is never open by
    # accident. Real apps use proper logins instead of a shared token.
    staff_token: str = field(default_factory=lambda: os.getenv("STAFF_TOKEN")
                             or secrets.token_urlsafe(16))
    host: str = "127.0.0.1"
    port: int = int(os.getenv("PORT", "8001"))
