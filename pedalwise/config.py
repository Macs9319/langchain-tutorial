"""Settings, read from the environment (and .env) when Settings() is created.

Every setting can come from an environment variable, so the same code runs
on your laptop and in a container without edits: only the environment
changes (Tutorial 21).
"""

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def env(name: str, default: str = "") -> str:
    """A dataclass default read from the environment each time Settings() is made."""
    return field(default_factory=lambda: os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # "development" on your machine, "production" when deployed. Production
    # refuses to start with unsafe settings (see problems()) and hides /docs.
    environment: str = env("PEDALWISE_ENV", "development")
    model: str = env("OPENAI_MODEL", "gpt-5-mini")
    fallback_model: str = env("FALLBACK_MODEL", "gpt-5-nano")
    data_dir: Path = ROOT / "data"
    index_dir: Path = field(default_factory=lambda: Path(
        os.getenv("PEDALWISE_INDEX_DIR", str(ROOT / ".rag_index"))))
    db_path: str = env("PEDALWISE_DB", str(ROOT / "pedalwise.sqlite"))
    # 127.0.0.1 accepts connections from this computer only. A container
    # needs 0.0.0.0 so traffic from outside the container reaches it.
    host: str = env("HOST", "127.0.0.1")
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8001")))
    log_level: str = env("LOG_LEVEL", "INFO")
    # Runs keep customers' questions and answers; delete them after this long (22).
    telemetry_retention_days: int = field(default_factory=lambda: int(
        os.getenv("TELEMETRY_RETENTION_DAYS", "30")))
    # Secrets: repr=False keeps them out of logs and error messages that
    # print the settings object.
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
                                repr=False)
    # Staff must send this token to approve refunds. If it isn't set, a random
    # one is made and logged, so the staff page is never open by accident.
    # Real apps use proper logins instead of a shared token.
    staff_token: str = field(default_factory=lambda: os.getenv("STAFF_TOKEN", ""), repr=False)
    staff_token_generated: bool = field(default=False, init=False)

    def __post_init__(self):
        if not self.staff_token:
            # The dataclass is frozen, so set the fields the low-level way.
            object.__setattr__(self, "staff_token", secrets.token_urlsafe(16))
            object.__setattr__(self, "staff_token_generated", True)

    @property
    def production(self) -> bool:
        return self.environment == "production"

    def problems(self) -> list[str]:
        """What's wrong with these settings. Production won't start unless empty."""
        found = []
        if not self.openai_api_key:
            found.append("OPENAI_API_KEY is not set")
        if self.environment not in ("development", "production"):
            found.append(f"PEDALWISE_ENV must be development or production, not {self.environment!r}")
        if self.production:
            # A generated token changes on every restart and only appears in
            # the logs: staff would be locked out after each deploy.
            if self.staff_token_generated:
                found.append("STAFF_TOKEN must be set in production")
            elif len(self.staff_token) < 24:
                found.append("STAFF_TOKEN must be at least 24 characters in production")
        return found
