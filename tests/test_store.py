from __future__ import annotations

import json

from harness_tap.store import JsonlTraceStore


def test_append_writes_jsonl_record_under_date_and_session(tmp_path):
    store = JsonlTraceStore(tmp_path)
    record = {
        "timestamp": "2026-07-02T12:34:56+00:00",
        "session_id": "session-1",
        "turn": 1,
        "request": {"body": {"model": "gpt-test"}},
    }

    session_id = store.append(record)

    assert session_id == "session-1"
    trace_path = tmp_path / "2026-07-02" / "session-1.jsonl"
    assert trace_path.exists()
    assert json.loads(trace_path.read_text(encoding="utf-8"))["turn"] == 1


def test_load_session_returns_records_in_append_order(tmp_path):
    store = JsonlTraceStore(tmp_path)
    store.append({"timestamp": "2026-07-02T00:00:00+00:00", "session_id": "session-1", "turn": 1})
    store.append({"timestamp": "2026-07-02T00:00:01+00:00", "session_id": "session-1", "turn": 2})

    records = store.load_session("session-1")

    assert [record["turn"] for record in records] == [1, 2]


def test_list_sessions_returns_metadata_newest_first(tmp_path):
    store = JsonlTraceStore(tmp_path)
    store.append({"timestamp": "2026-07-02T00:00:00+00:00", "session_id": "older", "turn": 1})
    store.append({"timestamp": "2026-07-03T00:00:00+00:00", "session_id": "newer", "turn": 1})
    store.append({"timestamp": "2026-07-03T00:00:01+00:00", "session_id": "newer", "turn": 2})

    sessions = store.list_sessions()

    assert [session["id"] for session in sessions] == ["newer", "older"]
    assert sessions[0]["record_count"] == 2
    assert sessions[0]["started_at"] == "2026-07-03T00:00:00+00:00"
    assert sessions[0]["updated_at"] == "2026-07-03T00:00:01+00:00"
