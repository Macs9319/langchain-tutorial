"""Settings, read from the environment (and .env) when Settings() is created.

Every setting can come from an environment variable, so the same code runs
on your laptop and in a container without edits: only the environment
changes (Tutorial 21).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .models import API_KEY_VARS, deepseek_thinking_from_env, default_model_spec, split_spec

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
    # "provider:model", e.g. "deepseek:deepseek-flash" (Tutorial 23). A bare
    # name means OpenAI, so OPENAI_MODEL from earlier tutorials still works.
    model: str = field(default_factory=default_model_spec)
    # "none" turns the fallback off, e.g. when you only have a DeepSeek key.
    fallback_model: str = env("FALLBACK_MODEL", "gpt-5-nano")
    deepseek_thinking: bool = field(default_factory=deepseek_thinking_from_env)
    # "openai": hybrid search (keywords + OpenAI embeddings).
    # "none": keywords only. For providers without embeddings, like DeepSeek.
    embeddings: str = env("EMBEDDINGS", "openai")
    data_dir: Path = ROOT / "data"
    index_dir: Path = field(default_factory=lambda: Path(
        os.getenv("PEDALWISE_INDEX_DIR", str(ROOT / ".rag_index"))))
    db_path: str = env("PEDALWISE_DB", str(ROOT / "pedalwise.sqlite"))
    # A Postgres URL (postgresql://...) shared by every copy of the app
    # (Tutorial 26). Unset: the SQLite file at db_path, for a single copy.
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""),
                              repr=False)  # it may contain a password
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
    # People log in (Tutorial 27). This token is only for scripts and tools
    # that call the staff API (X-Staff-Token header). Unset: no token access.
    staff_token: str = field(default_factory=lambda: os.getenv("STAFF_TOKEN", ""), repr=False)
    # If set, demo accounts (Sam, Dana, staff@example.com) are created at
    # startup with this password. For tutorials and tests: leave it unset
    # for a real deployment and create accounts with 27_auth.py instead.
    demo_password: str = field(default_factory=lambda: os.getenv("PEDALWISE_DEMO_PASSWORD", ""),
                               repr=False)
    # Send the session cookie over HTTPS only. On by default in production;
    # COOKIE_SECURE=false for a local stack on plain http://127.0.0.1.
    cookie_secure: bool | None = field(default_factory=lambda: {
        "true": True, "false": False}.get(os.getenv("COOKIE_SECURE", "").lower()))

    @property
    def secure_cookies(self) -> bool:
        return self.production if self.cookie_secure is None else self.cookie_secure

    @property
    def db_url(self) -> str:
        return self.database_url or self.db_path

    @property
    def production(self) -> bool:
        return self.environment == "production"

    def problems(self) -> list[str]:
        """What's wrong with these settings. Production won't start unless empty."""
        found = []
        # Every provider in use needs its key: the chat model, the fallback,
        # and OpenAI again if it makes the embeddings.
        providers = {split_spec(self.model)[0]}
        if self.fallback_model != "none":
            providers.add(split_spec(self.fallback_model)[0])
        if self.embeddings == "openai":
            providers.add("openai")
        elif self.embeddings != "none":
            found.append(f"EMBEDDINGS must be openai or none, not {self.embeddings!r}")
        for provider in sorted(providers):
            var = API_KEY_VARS.get(provider)
            key = self.openai_api_key if provider == "openai" else os.getenv(var or "", "")
            if var is None:
                found.append(f"Unknown provider {provider!r} (known: {', '.join(API_KEY_VARS)})")
            elif not key:
                found.append(f"{var} is not set (needed for {provider})")
        if self.environment not in ("development", "production"):
            found.append(f"PEDALWISE_ENV must be development or production, not {self.environment!r}")
        if self.production:
            if self.staff_token and len(self.staff_token) < 24:
                found.append("STAFF_TOKEN must be at least 24 characters in production")
            if self.demo_password and len(self.demo_password) < 12:
                found.append("PEDALWISE_DEMO_PASSWORD must be at least 12 characters")
        return found
