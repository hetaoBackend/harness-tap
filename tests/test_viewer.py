from __future__ import annotations

from aiohttp import ClientSession, web

from harness_tap.proxy import create_app
from harness_tap.store import JsonlTraceStore


async def test_viewer_index_and_session_apis(tmp_path):
    store = JsonlTraceStore(tmp_path, session_id="session-viewer")
    store.append(
        {
            "timestamp": "2026-07-02T09:00:00+00:00",
            "turn": 1,
            "duration_ms": 123,
            "transport": "http",
            "upstream_base_url": "https://api.minimaxi.com/v1",
            "request": {
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": {"Authorization": "***"},
                "body": {
                    "model": "MiniMax-M3",
                    "messages": [
                        {"role": "system", "content": "You are Harness."},
                        {"role": "user", "content": "Show trace."},
                    ],
                    "tools": [{"type": "function", "function": {"name": "read_file"}}],
                    "stream": False,
                },
            },
            "response": {
                "status": 200,
                "headers": {},
                "body": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Trace ready.",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                },
            },
            "capture": {"client": "harness", "protocol": "openai-chat-completions"},
        }
    )
    runner, base_url = await _start_app(create_app(upstream_base_url="https://api.example.test/v1", trace_store=store))

    try:
        async with ClientSession() as session:
            viewer_response = await session.get(f"{base_url}/viewer")
            viewer_html = await viewer_response.text()
            sessions_response = await session.get(f"{base_url}/api/sessions")
            sessions_body = await sessions_response.json()
            detail_response = await session.get(f"{base_url}/api/sessions/session-viewer")
            detail_body = await detail_response.json()

        assert viewer_response.status == 200
        assert "Harness Tap Trace Viewer" in viewer_html
        assert "trace-shell" in viewer_html
        assert sessions_response.status == 200
        assert sessions_body["sessions"][0]["id"] == "session-viewer"
        assert sessions_body["sessions"][0]["record_count"] == 1
        assert detail_response.status == 200
        assert detail_body["session"]["id"] == "session-viewer"
        assert detail_body["records"][0]["request"]["body"]["messages"][1]["content"] == "Show trace."
    finally:
        await runner.cleanup()


async def test_session_api_returns_grouped_turn_context_for_viewer(tmp_path):
    store = JsonlTraceStore(tmp_path, session_id="session-context")
    store.append(
        {
            "timestamp": "2026-07-02T09:00:00+00:00",
            "turn": 4,
            "duration_ms": 456,
            "transport": "http",
            "upstream_base_url": "https://api.minimaxi.com/v1",
            "request": {
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": {"Authorization": "***"},
                "body": {
                    "model": "MiniMax-M3",
                    "messages": [
                        {"role": "system", "content": "You are Harness."},
                        {"role": "user", "content": "<system-reminder>Available skills</system-reminder>"},
                        {"role": "user", "content": "Build a hello world."},
                        {"role": "assistant", "content": "I will inspect the repo."},
                        {"role": "tool", "content": "README.md\npyproject.toml"},
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {"name": "Bash", "description": "Run a shell command."},
                        },
                        {
                            "type": "function",
                            "function": {"name": "Edit", "description": "Edit a file."},
                        },
                    ],
                    "tool_choice": "auto",
                    "stream": False,
                },
            },
            "response": {
                "status": 200,
                "headers": {},
                "body": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Created hello world.",
                            }
                        }
                    ],
                },
            },
            "capture": {"client": "harness", "protocol": "openai-chat-completions"},
        }
    )
    runner, base_url = await _start_app(create_app(upstream_base_url="https://api.example.test/v1", trace_store=store))

    try:
        async with ClientSession() as session:
            detail_response = await session.get(f"{base_url}/api/sessions/session-context")
            detail_body = await detail_response.json()

        assert detail_response.status == 200
        turn = detail_body["turns"][0]
        sections = {section["kind"]: section for section in turn["sections"]}
        assert sections["system"]["label"] == "System prompts"
        assert sections["system"]["items"][0]["text"] == "You are Harness."
        assert sections["tool_schemas"]["count"] == 2
        assert [item["title"] for item in sections["tool_schemas"]["items"]] == ["Bash", "Edit"]
        assert "Build a hello world." in [item["text"] for item in sections["user"]["items"]]
        assert sections["tool_results"]["items"][0]["text"] == "README.md\npyproject.toml"
        assert sections["response"]["items"][0]["text"] == "Created hello world."
    finally:
        await runner.cleanup()


async def test_viewer_html_supports_expandable_turn_cards(tmp_path):
    runner, base_url = await _start_app(
        create_app(upstream_base_url="https://api.example.test/v1", trace_store=JsonlTraceStore(tmp_path))
    )

    try:
        async with ClientSession() as session:
            viewer_response = await session.get(f"{base_url}/viewer")
            viewer_html = await viewer_response.text()

        assert viewer_response.status == 200
        assert "expandedRecords" in viewer_html
        assert "data-context-section" in viewer_html
        assert "aria-expanded" in viewer_html
    finally:
        await runner.cleanup()


async def test_viewer_html_supports_structured_request_response_inspector(tmp_path):
    runner, base_url = await _start_app(
        create_app(upstream_base_url="https://api.example.test/v1", trace_store=JsonlTraceStore(tmp_path))
    )

    try:
        async with ClientSession() as session:
            viewer_response = await session.get(f"{base_url}/viewer")
            viewer_html = await viewer_response.text()

        assert viewer_response.status == 200
        assert "renderRequestInspector" in viewer_html
        assert "renderResponseInspector" in viewer_html
        assert "data-inspector-section" in viewer_html
        assert "Prompt messages" in viewer_html
        assert "Tool schemas" in viewer_html
    finally:
        await runner.cleanup()


async def test_viewer_html_lazily_loads_long_inspector_text(tmp_path):
    runner, base_url = await _start_app(
        create_app(upstream_base_url="https://api.example.test/v1", trace_store=JsonlTraceStore(tmp_path))
    )

    try:
        async with ClientSession() as session:
            viewer_response = await session.get(f"{base_url}/viewer")
            viewer_html = await viewer_response.text()

        assert viewer_response.status == 200
        assert "TEXT_PREVIEW_LIMIT" in viewer_html
        assert "data-fulltext-ref" in viewer_html
        assert "bindInspectorDetailToggles" in viewer_html
        assert "previewText" in viewer_html
    finally:
        await runner.cleanup()


async def test_viewer_html_preserves_inspector_detail_state_on_refresh(tmp_path):
    runner, base_url = await _start_app(
        create_app(upstream_base_url="https://api.example.test/v1", trace_store=JsonlTraceStore(tmp_path))
    )

    try:
        async with ClientSession() as session:
            viewer_response = await session.get(f"{base_url}/viewer")
            viewer_html = await viewer_response.text()

        assert viewer_response.status == 200
        assert "openInspectorDetails" in viewer_html
        assert "closedInspectorDetails" in viewer_html
        assert "data-detail-key" in viewer_html
        assert "inspectorDetailKey" in viewer_html
        assert "sessionFingerprint" in viewer_html
    finally:
        await runner.cleanup()


async def test_viewer_html_keeps_inspector_sections_in_scroll_layout(tmp_path):
    runner, base_url = await _start_app(
        create_app(upstream_base_url="https://api.example.test/v1", trace_store=JsonlTraceStore(tmp_path))
    )

    try:
        async with ClientSession() as session:
            viewer_response = await session.get(f"{base_url}/viewer")
            viewer_html = await viewer_response.text()

        assert viewer_response.status == 200
        inspector_panel_css = viewer_html.split(".inspector-panel {", 1)[1].split("}", 1)[0]
        assert "display: grid" not in inspector_panel_css
        inspector_section_css = viewer_html.split(".inspector-section {", 1)[1].split("}", 1)[0]
        assert "content-visibility" not in inspector_section_css
        assert "contain-intrinsic-size" not in inspector_section_css
    finally:
        await runner.cleanup()


async def _start_app(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"
