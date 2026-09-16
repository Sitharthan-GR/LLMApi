"""JSONL request log. Redact secrets. Never write raw API keys."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

SECRET_RE = re.compile(r"(gsk_[A-Za-z0-9_-]+|sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]+|Bearer\s+\S+)", re.I)


def redact(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text)


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = json.loads(redact(json.dumps(row)))
    safe["ts"] = datetime.now(timezone.utc).isoformat()
    with path.open("a") as fh:
        fh.write(json.dumps(safe) + "\n")
