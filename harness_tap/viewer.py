from __future__ import annotations

from pathlib import Path

from aiohttp import web

from harness_tap.context import session_summary, turn_context_summary


VIEWER_HTML = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")


def install_viewer_routes(app: web.Application, trace_store_key: web.AppKey) -> None:
    async def viewer_index(request: web.Request) -> web.Response:
        return web.Response(text=VIEWER_HTML, content_type="text/html")

    async def api_sessions(request: web.Request) -> web.Response:
        store = request.app[trace_store_key]
        sessions = []
        for summary in store.list_sessions():
            records = store.load_session(summary["id"])
            sessions.append({**summary, **session_summary(summary["id"], records)})
        return web.json_response({"sessions": sessions})

    async def api_session_detail(request: web.Request) -> web.Response:
        session_id = request.match_info["session_id"]
        store = request.app[trace_store_key]
        records = store.load_session(session_id)
        if not records:
            return web.json_response({"error": "session not found", "session_id": session_id}, status=404)
        return web.json_response(
            {
                "session": session_summary(session_id, records),
                "turns": [turn_context_summary(record, index) for index, record in enumerate(records)],
                "records": records,
            }
        )

    app.router.add_get("/", viewer_index)
    app.router.add_get("/viewer", viewer_index)
    app.router.add_get("/api/sessions", api_sessions)
    app.router.add_get("/api/sessions/{session_id}", api_session_detail)
