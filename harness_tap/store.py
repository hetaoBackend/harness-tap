from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class TraceStore(Protocol):
    def append(self, record: dict[str, Any]) -> str:
        """Append a trace record and return its session id."""

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return trace session summaries."""

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Load all records for a session."""


class JsonlTraceStore:
    def __init__(self, base_dir: Path | str, *, session_id: str | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.session_id = session_id or f"session-{uuid.uuid4().hex}"

    def append(self, record: dict[str, Any]) -> str:
        session_id = str(record.get("session_id") or self.session_id)
        timestamp = _record_timestamp(record)
        date_key = timestamp.date().isoformat()
        path = self.base_dir / date_key / f"{_safe_session_id(session_id)}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        stored_record = dict(record)
        stored_record["session_id"] = session_id
        with path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(stored_record, ensure_ascii=False, separators=(",", ":")))
            file_obj.write("\n")
        return session_id

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in self.base_dir.glob("*/*.jsonl"):
            records = _read_jsonl(path)
            if not records:
                continue
            session_id = str(records[0].get("session_id") or path.stem)
            timestamps = [record.get("timestamp") for record in records if isinstance(record.get("timestamp"), str)]
            started_at = timestamps[0] if timestamps else ""
            updated_at = timestamps[-1] if timestamps else ""
            sessions.append(
                {
                    "id": session_id,
                    "path": str(path),
                    "record_count": len(records),
                    "started_at": started_at,
                    "updated_at": updated_at,
                }
            )
        sessions.sort(key=lambda item: (item["updated_at"], item["started_at"], item["id"]), reverse=True)
        return sessions

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path_for_session(session_id)
        if path is None:
            return []
        return _read_jsonl(path)

    def _path_for_session(self, session_id: str) -> Path | None:
        safe_id = _safe_session_id(session_id)
        for path in sorted(self.base_dir.glob(f"*/{safe_id}.jsonl")):
            return path
        for path in self.base_dir.glob("*/*.jsonl"):
            records = _read_jsonl(path)
            if records and records[0].get("session_id") == session_id:
                return path
        return None


def _safe_session_id(session_id: str) -> str:
    return session_id.replace("/", "_").replace("\\", "_")


def _record_timestamp(record: dict[str, Any]) -> datetime:
    timestamp = record.get("timestamp")
    if isinstance(timestamp, str) and timestamp:
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.now(timezone.utc)
    else:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records
