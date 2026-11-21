"""Logging: readable text on your machine, one JSON object per line in production.

Log collectors (Docker, CloudWatch, Loki, Datadog...) read a container's
standard output. JSON lines let them filter on fields such as thread_id
instead of searching free text (Tutorial 21).

Add fields to a log line with extra=:
    logger.info("refund decided", extra={"thread_id": tid, "approved": True})
"""

import json
import logging
import sys
from datetime import UTC, datetime

# Attributes every LogRecord has; anything else came from extra=. Uvicorn
# also adds color_message, a copy of the message with terminal colour codes.
_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "color_message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD})
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO", json_lines: bool = False) -> None:
    """Send all logs, including uvicorn's, to stdout in one format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_lines else logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Let uvicorn's loggers flow into the handler above instead of their own.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    # The HTTP client logs every OpenAI request at INFO; that's noise here.
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
