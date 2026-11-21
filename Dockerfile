# Tutorial 21: the Pedalwise support app as a container image.
#
#   docker build -t pedalwise .
#   docker run --rm -p 127.0.0.1:8001:8001 --env-file .env \
#              -v pedalwise-data:/var/lib/pedalwise pedalwise
#
# Or simply: docker compose up --build

# --- Stage 1: install dependencies ------------------------------------------
# A separate build stage keeps uv and its cache out of the final image.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Dependencies first, code second: Docker caches each step, so editing the
# code doesn't reinstall every package. --locked uses exactly the versions
# in uv.lock (fails if it's out of date); --no-dev leaves out pytest.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY pedalwise/ pedalwise/
COPY data/ data/
# The server, and the account tool (Tutorial 27), so accounts can be made
# inside the container, against the database it actually uses:
#   docker compose exec pedalwise python 27_auth.py --add-user you@example.com --role staff
# the migration tool (Tutorial 28), which the scaled stack runs before the app,
# and the backup tool (Tutorial 29), which the "backup" service runs.
COPY 21_deployment.py 27_auth.py 28_migrations.py 29_backups.py ./

# --- Stage 2: the image that runs -------------------------------------------
FROM python:3.12-slim AS app

# Run as an ordinary user: if someone breaks into the app, they don't get root.
RUN useradd --create-home --uid 10001 pedalwise \
 && mkdir -p /var/lib/pedalwise /var/lib/pedalwise-backups \
 && chown pedalwise /var/lib/pedalwise /var/lib/pedalwise-backups

WORKDIR /app
COPY --from=build /app /app

# Settings for the container. Secrets (OPENAI_API_KEY, STAFF_TOKEN) are NOT
# here: anything in the image can be read by anyone who gets the image. They
# arrive at run time from --env-file or the platform's secret store.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PEDALWISE_ENV=production \
    HOST=0.0.0.0 \
    PORT=8001 \
    PEDALWISE_DB=/var/lib/pedalwise/pedalwise.sqlite \
    PEDALWISE_INDEX_DIR=/var/lib/pedalwise/rag_index \
    PEDALWISE_BACKUP_DIR=/var/lib/pedalwise-backups

# Everything that must survive a restart lives here: conversations, the
# approval queue and the search index (so documents are embedded once).
VOLUME /var/lib/pedalwise
USER pedalwise
EXPOSE 8001

# Docker marks the container unhealthy if this fails three times in a row.
# start-period: the first start embeds the documents, which takes a moment.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/readyz', timeout=4)"

# The exec form (a JSON list) makes Python process 1, so it receives
# SIGTERM from `docker stop` and shuts down gracefully.
CMD ["python", "21_deployment.py"]

# --- Stage 3: the same, plus the Postgres client (Tutorial 29) ----------------
# For the scaled stack's backup service: pg_dump must be the server's major
# version (18) or newer, and Debian's own package is older, so it comes from
# the PostgreSQL project's repository. Only the backup service uses this:
#   build: {context: ., target: backup}
FROM app AS backup
USER root
# env -u HOST: the setup script reads $HOST as the repository's address, and
# this image sets HOST=0.0.0.0 for the server.
RUN apt-get update \
 && apt-get install -y --no-install-recommends postgresql-common \
 && env -u HOST /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y \
 && apt-get install -y --no-install-recommends postgresql-client-18 \
 && rm -rf /var/lib/apt/lists/*
USER pedalwise
HEALTHCHECK NONE

# --- The default image -------------------------------------------------------
# `docker build` builds the last stage, so end with the app itself.
FROM app
