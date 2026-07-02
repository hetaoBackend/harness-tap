from __future__ import annotations

import json
from typing import Any

from aiohttp import web


def install_viewer_routes(app: web.Application, trace_store_key: web.AppKey) -> None:
    async def viewer_index(request: web.Request) -> web.Response:
        return web.Response(text=VIEWER_HTML, content_type="text/html")

    async def api_sessions(request: web.Request) -> web.Response:
        store = request.app[trace_store_key]
        sessions = store.list_sessions()
        return web.json_response({"sessions": sessions})

    async def api_session_detail(request: web.Request) -> web.Response:
        session_id = request.match_info["session_id"]
        store = request.app[trace_store_key]
        records = store.load_session(session_id)
        if not records:
            return web.json_response({"error": "session not found", "session_id": session_id}, status=404)
        return web.json_response(
            {
                "session": _session_summary(session_id, records),
                "turns": [_turn_context_summary(record, index) for index, record in enumerate(records)],
                "records": records,
            }
        )

    app.router.add_get("/", viewer_index)
    app.router.add_get("/viewer", viewer_index)
    app.router.add_get("/api/sessions", api_sessions)
    app.router.add_get("/api/sessions/{session_id}", api_session_detail)


def _session_summary(session_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [record.get("timestamp") for record in records if isinstance(record.get("timestamp"), str)]
    return {
        "id": session_id,
        "record_count": len(records),
        "started_at": timestamps[0] if timestamps else "",
        "updated_at": timestamps[-1] if timestamps else "",
        "models": sorted(
            {
                model
                for record in records
                for model in [_record_model(record)]
                if model
            }
        ),
        "total_duration_ms": sum(
            int(record.get("duration_ms") or 0)
            for record in records
            if isinstance(record.get("duration_ms"), int)
        ),
        "total_tokens": sum(_record_total_tokens(record) for record in records),
        "error_count": sum(1 for record in records if _record_status(record) >= 400),
    }


def _record_model(record: dict[str, Any]) -> str:
    request = record.get("request")
    body = request.get("body") if isinstance(request, dict) else {}
    model = body.get("model") if isinstance(body, dict) else ""
    return model if isinstance(model, str) else ""


def _record_status(record: dict[str, Any]) -> int:
    response = record.get("response")
    status = response.get("status") if isinstance(response, dict) else 0
    return status if isinstance(status, int) else 0


def _record_total_tokens(record: dict[str, Any]) -> int:
    response = record.get("response")
    body = response.get("body") if isinstance(response, dict) else {}
    usage = body.get("usage") if isinstance(body, dict) else {}
    if not isinstance(usage, dict):
        return 0
    value = usage.get("total_tokens")
    if isinstance(value, int):
        return value
    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    return (input_tokens if isinstance(input_tokens, int) else 0) + (output_tokens if isinstance(output_tokens, int) else 0)


def _turn_context_summary(record: dict[str, Any], index: int) -> dict[str, Any]:
    request_body = _request_body(record)
    response_body = _response_body(record)
    messages = request_body.get("messages")
    tools = request_body.get("tools")
    sections = [
        _section("system", "System prompts", _message_items(messages, "system")),
        _section("tool_schemas", "Tool schemas", _tool_schema_items(tools)),
        _section("user", "User prompts", _message_items(messages, "user")),
        _section("assistant", "Assistant messages", _message_items(messages, "assistant")),
        _section("tool_results", "Tool results", _message_items(messages, "tool")),
        _section("response", "Upstream response", _response_items(response_body)),
    ]
    return {
        "index": index,
        "turn": record.get("turn", index + 1),
        "model": request_body.get("model", ""),
        "status": _record_status(record),
        "duration_ms": record.get("duration_ms", 0),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "tool_schema_count": len(tools) if isinstance(tools, list) else 0,
        "sections": [section for section in sections if section["count"]],
    }


def _request_body(record: dict[str, Any]) -> dict[str, Any]:
    request = record.get("request")
    body = request.get("body") if isinstance(request, dict) else {}
    return body if isinstance(body, dict) else {}


def _response_body(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response")
    body = response.get("body") if isinstance(response, dict) else {}
    return body if isinstance(body, dict) else {}


def _section(kind: str, label: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": kind, "label": label, "count": len(items), "items": items}


def _message_items(messages: Any, role: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        text = _content_text(message.get("content"))
        tool_calls = _tool_call_text(message.get("tool_calls"))
        items.append(
            {
                "title": f"{role} #{len(items) + 1}",
                "text": text or tool_calls,
            }
        )
    return items


def _tool_schema_items(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    items: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        items.append(
            {
                "title": _tool_name(tool),
                "text": _tool_description(tool),
            }
        )
    return items


def _response_items(response_body: dict[str, Any]) -> list[dict[str, Any]]:
    assistant = _assistant_message(response_body)
    if not assistant:
        return []
    text = _content_text(assistant.get("content")) or _tool_call_text(assistant.get("tool_calls"))
    if not text:
        return []
    return [{"title": "assistant", "text": text}]


def _assistant_message(response_body: dict[str, Any]) -> dict[str, Any]:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    return message if isinstance(message, dict) else {}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(item.get("type"), str):
                    parts.append(f"[{item['type']}]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    name = tool.get("name")
    if isinstance(name, str):
        return name
    tool_type = tool.get("type")
    return tool_type if isinstance(tool_type, str) else "tool"


def _tool_description(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict):
        description = function.get("description")
        if isinstance(description, str):
            return description
    description = tool.get("description")
    return description if isinstance(description, str) else ""


def _tool_call_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    parts: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name") or tool_call.get("id") or "tool"
            arguments = function.get("arguments") or ""
            parts.append(f"{name}({arguments})")
        else:
            parts.append(str(tool_call.get("id") or "tool"))
    return "\n".join(parts)


VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Harness Tap Trace Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --ink: #e8edf2;
      --muted: #8fa0ad;
      --faint: #5e6b76;
      --line: #27343f;
      --panel: #10181f;
      --panel-2: #0c1218;
      --page: #060a0e;
      --amber: #e1b65f;
      --cyan: #71c7d9;
      --green: #7fca8b;
      --red: #e17878;
      --violet: #b9a4ff;
      --chip: #17232d;
      --focus: #f2ce7d;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--ink);
      min-height: 100vh;
      overflow: hidden;
    }
    button, input, select {
      font: inherit;
    }
    button:focus-visible, input:focus-visible, select:focus-visible {
      outline: 2px solid var(--focus);
      outline-offset: 2px;
    }
    .trace-shell {
      display: grid;
      grid-template-columns: 300px minmax(440px, 1fr) minmax(360px, 42vw);
      height: 100vh;
      min-width: 980px;
    }
    .sessions, .timeline, .inspector {
      min-height: 0;
      border-right: 1px solid var(--line);
      background: var(--panel-2);
    }
    .inspector { border-right: 0; background: #081016; }
    .brand {
      height: 68px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .brand h1 {
      margin: 0;
      font-size: 15px;
      line-height: 1.1;
      letter-spacing: 0;
      font-weight: 760;
    }
    .brand p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.2;
    }
    .pulse {
      width: 11px;
      height: 11px;
      border-radius: 999px;
      background: var(--green);
      box-shadow: 0 0 0 4px rgba(127, 202, 139, .12);
      flex: 0 0 auto;
    }
    .toolbar {
      padding: 12px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 8px;
    }
    .search {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      background: #071016;
      color: var(--ink);
      padding: 0 10px;
      border-radius: 6px;
    }
    .meta-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }
    .chip {
      border: 1px solid var(--line);
      background: var(--chip);
      color: var(--muted);
      border-radius: 999px;
      padding: 3px 8px;
      min-height: 22px;
    }
    .list {
      height: calc(100vh - 139px);
      overflow: auto;
      padding: 8px;
    }
    .session-item, .turn-item {
      width: 100%;
      text-align: left;
      border: 1px solid transparent;
      background: transparent;
      color: var(--ink);
      border-radius: 7px;
      padding: 10px;
      display: grid;
      gap: 6px;
      cursor: pointer;
    }
    .session-item:hover, .turn-item:hover { background: rgba(255,255,255,.035); }
    .session-item.active, .turn-item.active {
      border-color: rgba(113, 199, 217, .45);
      background: rgba(113, 199, 217, .09);
    }
    .item-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
      font-size: 13px;
      font-weight: 700;
    }
    .item-sub {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .status-ok { color: var(--green); }
    .status-error { color: var(--red); }
    .main-head, .inspect-head {
      min-height: 68px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 6px;
      background: var(--panel);
    }
    .main-title, .inspect-title {
      margin: 0;
      font-size: 16px;
      font-weight: 780;
      line-height: 1.2;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #0a141b;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px;
      min-height: 58px;
      background: #0c171f;
    }
    .metric b {
      display: block;
      font-size: 18px;
      line-height: 1.15;
    }
    .metric span {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    .turns {
      height: calc(100vh - 183px);
      overflow: auto;
      padding: 12px 14px 28px;
      display: grid;
      gap: 12px;
      align-content: start;
    }
    .turn-card {
      border: 1px solid var(--line);
      background: #0b151c;
      border-radius: 8px;
      overflow: hidden;
      content-visibility: auto;
      contain-intrinsic-size: 172px;
    }
    .turn-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #101b24;
      cursor: pointer;
    }
    .turn-top strong { font-size: 13px; }
    .turn-top span { color: var(--muted); font-size: 12px; }
    .turn-top-left, .turn-top-right {
      display: flex;
      gap: 8px;
      align-items: baseline;
      min-width: 0;
      flex-wrap: wrap;
    }
    .expand-state {
      color: var(--cyan);
      font-size: 12px;
      font-weight: 700;
    }
    .context-stack { padding: 8px 12px 12px; display: grid; gap: 10px; }
    .context-section {
      display: grid;
      grid-template-columns: 118px minmax(0, 1fr);
      gap: 12px;
      padding-top: 8px;
      border-top: 1px solid rgba(39, 52, 63, .65);
    }
    .context-section:first-child { border-top: 0; padding-top: 0; }
    .context-label {
      display: flex;
      gap: 6px;
      align-items: baseline;
      flex-wrap: wrap;
      min-width: 0;
    }
    .context-label b {
      font-size: 11px;
      font-weight: 850;
      text-transform: uppercase;
    }
    .context-label span { color: var(--muted); font-size: 11px; }
    .context-items { display: grid; gap: 7px; min-width: 0; }
    .context-item {
      display: grid;
      grid-template-columns: minmax(82px, 118px) minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      min-width: 0;
    }
    .context-title {
      color: var(--muted);
      font-size: 11px;
      font-weight: 740;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .context-text {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #d8e2ea;
      line-height: 1.45;
      font-size: 13px;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .turn-card.expanded .context-text {
      display: block;
      -webkit-line-clamp: unset;
    }
    .context-more { color: var(--muted); font-size: 12px; }
    .context-section.system .context-label b { color: var(--amber); }
    .context-section.user .context-label b { color: var(--cyan); }
    .context-section.assistant .context-label b,
    .context-section.response .context-label b { color: var(--green); }
    .context-section.tool_schemas .context-label b,
    .context-section.tool_results .context-label b { color: var(--violet); }
    .message-stack { padding: 10px 12px; display: grid; gap: 8px; }
    .message {
      display: grid;
      grid-template-columns: 82px minmax(0, 1fr);
      gap: 10px;
      align-items: start;
      font-size: 13px;
    }
    .role {
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      color: var(--muted);
      padding-top: 2px;
    }
    .role.system { color: var(--amber); }
    .role.user { color: var(--cyan); }
    .role.assistant { color: var(--green); }
    .role.tool { color: var(--violet); }
    .text {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      color: #d8e2ea;
      line-height: 1.45;
    }
    .text.dim { color: var(--muted); }
    .inspect-body {
      height: calc(100vh - 68px);
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      min-height: 0;
    }
    .tabs {
      display: flex;
      gap: 6px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      overflow-x: auto;
    }
    .tab {
      border: 1px solid var(--line);
      background: #0c151d;
      color: var(--muted);
      border-radius: 6px;
      padding: 6px 9px;
      cursor: pointer;
      min-width: max-content;
    }
    .tab.active {
      color: var(--ink);
      border-color: rgba(225, 182, 95, .5);
      background: rgba(225, 182, 95, .1);
    }
    .inspect-actions {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: flex;
      gap: 8px;
      justify-content: space-between;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }
    .inspector-panel {
      height: 100%;
      overflow: auto;
      padding: 12px;
      background: #071017;
    }
    #payload {
      min-height: 0;
      overflow: hidden;
    }
    .inspector-section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0b151c;
      overflow: hidden;
    }
    .inspector-section + .inspector-section {
      margin-top: 12px;
    }
    .inspector-section h3 {
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 12px;
      line-height: 1.2;
      text-transform: uppercase;
      color: var(--muted);
      background: #101b24;
    }
    .inspector-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      padding: 10px 12px;
    }
    .inspector-stat {
      border: 1px solid rgba(39, 52, 63, .75);
      border-radius: 7px;
      padding: 8px;
      min-width: 0;
      background: #08131a;
    }
    .inspector-stat b {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-size: 13px;
    }
    .inspector-stat span {
      color: var(--muted);
      display: block;
      font-size: 11px;
      margin-top: 3px;
      text-transform: uppercase;
    }
    .prompt-list, .tool-list, .response-list {
      padding: 10px 12px;
      display: grid;
      gap: 8px;
    }
    .prompt-card, .tool-card, .response-card {
      border: 1px solid rgba(39, 52, 63, .85);
      border-radius: 7px;
      background: #08131a;
      overflow: hidden;
    }
    .prompt-card summary, .tool-card summary, .response-card summary {
      cursor: pointer;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 10px;
      color: var(--ink);
      font-size: 12px;
      font-weight: 760;
    }
    .prompt-card summary::-webkit-details-marker,
    .tool-card summary::-webkit-details-marker,
    .response-card summary::-webkit-details-marker {
      display: none;
    }
    .prompt-card summary span, .tool-card summary span, .response-card summary span {
      color: var(--muted);
      font-weight: 600;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .prompt-card.system summary { color: var(--amber); }
    .prompt-card.user summary { color: var(--cyan); }
    .prompt-card.assistant summary, .response-card.assistant summary { color: var(--green); }
    .prompt-card.tool summary, .tool-card summary, .response-card.tool_calls summary { color: var(--violet); }
    .inspector-text {
      border-top: 1px solid rgba(39, 52, 63, .75);
      margin: 0;
      padding: 10px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      height: auto;
      overflow: visible;
      color: #d8e2ea;
      font: 12px/1.5 "SFMono-Regular", Consolas, ui-monospace, monospace;
    }
    .tool-param {
      border-top: 1px solid rgba(39, 52, 63, .75);
      color: var(--muted);
      padding: 8px 10px;
      font-size: 12px;
    }
    .lazy-text-slot {
      min-height: 0;
    }
    .copy-btn {
      border: 1px solid var(--line);
      background: #12202a;
      color: var(--ink);
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
    }
    pre {
      margin: 0;
      padding: 14px;
      height: 100%;
      overflow: auto;
      color: #dbe6ee;
      background: #071017;
      font: 12px/1.5 "SFMono-Regular", Consolas, ui-monospace, monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .empty {
      color: var(--muted);
      padding: 18px;
      line-height: 1.45;
    }
    .delta-list {
      padding: 12px;
      display: grid;
      gap: 10px;
      overflow: auto;
    }
    .delta-item {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      background: #0a141b;
    }
    .delta-item b { display: block; margin-bottom: 4px; }
    @media (max-width: 980px) {
      body { overflow: auto; }
      .trace-shell {
        min-width: 0;
        height: auto;
        grid-template-columns: 1fr;
      }
      .sessions, .timeline, .inspector { border-right: 0; border-bottom: 1px solid var(--line); }
      .list, .turns, .inspect-body { height: auto; max-height: none; }
      .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .inspector-grid { grid-template-columns: 1fr; }
      .message, .context-section, .context-item { grid-template-columns: 1fr; gap: 4px; }
    }
  </style>
</head>
<body>
  <main class="trace-shell">
    <aside class="sessions">
      <header class="brand">
        <div>
          <h1>Harness Tap Trace Viewer</h1>
          <p>Local context evidence</p>
        </div>
        <span class="pulse" title="Viewer is live"></span>
      </header>
      <div class="toolbar">
        <input id="search" class="search" type="search" placeholder="Search current trace">
        <div class="meta-row">
          <span class="chip" id="session-count">0 sessions</span>
          <span class="chip" id="refresh-label">auto refresh</span>
        </div>
      </div>
      <nav id="sessions" class="list" aria-label="Trace sessions"></nav>
    </aside>

    <section class="timeline">
      <header class="main-head">
        <h2 class="main-title" id="session-title">No trace selected</h2>
        <div class="meta-row" id="session-meta"></div>
      </header>
      <section class="summary-grid" id="summary"></section>
      <section class="turns" id="turns"></section>
    </section>

    <aside class="inspector">
      <header class="inspect-head">
        <h2 class="inspect-title" id="inspect-title">Inspector</h2>
        <div class="meta-row" id="inspect-meta"></div>
      </header>
      <section class="inspect-body">
        <nav class="tabs" id="tabs" aria-label="Inspector tabs"></nav>
        <div class="inspect-actions">
          <span id="payload-label">Select a turn</span>
          <button id="copy" class="copy-btn" type="button">Copy JSON</button>
        </div>
        <div id="payload"></div>
      </section>
    </aside>
  </main>
  <script>
    const state = {
      sessions: [],
      detail: null,
      selectedSession: "",
      selectedRecordIndex: 0,
      expandedRecords: new Set(),
      openInspectorDetails: new Set(),
      closedInspectorDetails: new Set(),
      fullTextRefs: new Map(),
      fullTextCounter: 0,
      detailFingerprint: "",
      tab: "overview",
      query: ""
    };
    const $ = (id) => document.getElementById(id);
    const tabs = ["overview", "request", "response", "events", "delta", "raw"];
    const TEXT_PREVIEW_LIMIT = 900;
    const TIMELINE_PREVIEW_LIMIT = 260;

    $("search").addEventListener("input", (event) => {
      state.query = event.target.value.toLowerCase();
      renderSessions();
      renderTurns();
    });
    $("copy").addEventListener("click", async () => {
      const payload = currentPayload();
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      $("payload-label").textContent = "Copied";
      setTimeout(renderInspector, 700);
    });

    async function loadSessions() {
      const response = await fetch("/api/sessions", {cache: "no-store"});
      const body = await response.json();
      state.sessions = body.sessions || [];
      $("session-count").textContent = `${state.sessions.length} sessions`;
      if (!state.selectedSession && state.sessions.length) {
        state.selectedSession = state.sessions[0].id;
        await loadDetail(state.selectedSession);
      } else if (state.selectedSession) {
        const selectedSummary = state.sessions.find((session) => session.id === state.selectedSession);
        if (selectedSummary && sessionFingerprint(selectedSummary) !== state.detailFingerprint) {
          await loadDetail(state.selectedSession, true);
        }
      }
      renderSessions();
    }

    async function loadDetail(sessionId, keepSelection = false) {
      const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {cache: "no-store"});
      if (!response.ok) return;
      state.detail = await response.json();
      state.selectedSession = sessionId;
      state.detailFingerprint = sessionFingerprint(state.detail?.session);
      if (!keepSelection) {
        state.selectedRecordIndex = 0;
        state.expandedRecords = new Set([0]);
        state.openInspectorDetails = new Set();
        state.closedInspectorDetails = new Set();
      }
      renderAll();
    }

    function renderAll() {
      renderSessions();
      renderSessionHead();
      renderSummary();
      renderTurns();
      renderInspector();
    }

    function renderSessions() {
      const list = $("sessions");
      const sessions = state.sessions;
      list.innerHTML = sessions.length ? sessions.map((session) => `
        <button class="session-item ${session.id === state.selectedSession ? "active" : ""}" data-session="${escapeAttr(session.id)}">
          <span class="item-title"><span>${escapeHtml(shortId(session.id))}</span><span>${session.record_count}</span></span>
          <span class="item-sub">${escapeHtml(session.updated_at || session.started_at || "no timestamp")}</span>
        </button>
      `).join("") : `<div class="empty">No trace sessions yet. Send a request through the proxy and this list will fill in.</div>`;
      list.querySelectorAll("[data-session]").forEach((button) => {
        button.addEventListener("click", () => loadDetail(button.dataset.session));
      });
    }

    function renderSessionHead() {
      const detail = state.detail;
      if (!detail) {
        $("session-title").textContent = "No trace selected";
        $("session-meta").innerHTML = "";
        return;
      }
      const summary = detail.session;
      $("session-title").textContent = shortId(summary.id);
      $("session-meta").innerHTML = [
        `${summary.record_count} turns`,
        `${summary.models?.join(", ") || "model unknown"}`,
        `${summary.error_count || 0} errors`
      ].map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("");
    }

    function renderSummary() {
      const summary = state.detail?.session;
      const host = $("summary");
      if (!summary) {
        host.innerHTML = metric("0", "Turns") + metric("0", "Tokens") + metric("0ms", "Duration") + metric("-", "Model");
        return;
      }
      host.innerHTML = [
        metric(summary.record_count, "Turns"),
        metric(summary.total_tokens || 0, "Tokens"),
        metric(`${summary.total_duration_ms || 0}ms`, "Duration"),
        metric(summary.models?.[0] || "-", "Model"),
      ].join("");
    }

    function renderTurns() {
      const records = filteredRecords();
      const host = $("turns");
      if (!state.detail) {
        host.innerHTML = `<div class="empty">Choose a session to preview the full conversation context.</div>`;
        return;
      }
      if (!records.length) {
        host.innerHTML = `<div class="empty">No turns match the current search.</div>`;
        return;
      }
      host.innerHTML = records.map(({record, index}) => turnCard(record, index)).join("");
      host.querySelectorAll("[data-record]").forEach((button) => {
        button.addEventListener("click", () => {
          const index = Number(button.dataset.record);
          state.selectedRecordIndex = index;
          if (state.expandedRecords.has(index)) {
            state.expandedRecords.delete(index);
          } else {
            state.expandedRecords.add(index);
          }
          renderTurns();
          renderInspector();
        });
      });
    }

    function turnCard(record, index) {
      const req = record.request?.body || {};
      const status = record.response?.status || 0;
      const selected = index === state.selectedRecordIndex ? "active" : "";
      const expanded = state.expandedRecords.has(index);
      const summary = turnSummary(record, index);
      const contextRows = summary.sections?.length
        ? summary.sections.map((section) => contextSection(section, expanded)).join("")
        : `<div class="text dim">No messages captured.</div>`;
      return `
        <article class="turn-card ${selected} ${expanded ? "expanded" : ""}">
          <button class="turn-top turn-item ${selected}" data-record="${index}" aria-expanded="${expanded ? "true" : "false"}">
            <span class="turn-top-left">
              <strong>Turn ${escapeHtml(record.turn ?? index + 1)} - ${escapeHtml(req.model || "model unknown")}</strong>
              <span class="expand-state">${expanded ? "Collapse" : "Expand"}</span>
            </span>
            <span class="turn-top-right">
              <span>${summary.message_count || 0} messages</span>
              <span>${summary.tool_schema_count || 0} tools</span>
              <span class="${status >= 400 ? "status-error" : "status-ok"}">${status || "-"} / ${record.duration_ms ?? 0}ms</span>
            </span>
          </button>
          <div class="context-stack">${contextRows}</div>
        </article>
      `;
    }

    function contextSection(section, expanded) {
      const items = Array.isArray(section.items) ? section.items : [];
      const visibleItems = expanded ? items : items.slice(0, 2);
      const remainder = Math.max(0, items.length - visibleItems.length);
      return `
        <section class="context-section ${escapeAttr(section.kind || "context")}" data-context-section="${escapeAttr(section.kind || "context")}">
          <div class="context-label">
            <b>${escapeHtml(section.label || section.kind || "Context")}</b>
            <span>${section.count || items.length}</span>
          </div>
          <div class="context-items">
            ${visibleItems.map(contextItem).join("")}
            ${remainder ? `<div class="context-more">+${remainder} more in this turn</div>` : ""}
          </div>
        </section>
      `;
    }

    function contextItem(item) {
      return `
        <div class="context-item">
          <div class="context-title">${escapeHtml(item.title || "")}</div>
          <div class="context-text">${escapeHtml(previewText(item.text || "", TIMELINE_PREVIEW_LIMIT))}</div>
        </div>
      `;
    }

    function turnSummary(record, index) {
      const fromApi = state.detail?.turns?.[index];
      if (fromApi) return fromApi;
      const req = record.request?.body || {};
      const res = record.response?.body || {};
      const messages = Array.isArray(req.messages) ? req.messages : [];
      const tools = Array.isArray(req.tools) ? req.tools : [];
      const sections = [
        sectionFromMessages("system", "System prompts", messages, "system"),
        {kind: "tool_schemas", label: "Tool schemas", count: tools.length, items: tools.map((tool) => ({title: toolName(tool), text: toolDescription(tool)}))},
        sectionFromMessages("user", "User prompts", messages, "user"),
        sectionFromMessages("assistant", "Assistant messages", messages, "assistant"),
        sectionFromMessages("tool_results", "Tool results", messages, "tool"),
      ];
      const assistant = assistantMessage(res);
      if (assistant) {
        sections.push({kind: "response", label: "Upstream response", count: 1, items: [{title: "assistant", text: contentText(assistant.content) || toolCallText(assistant.tool_calls)}]});
      }
      return {
        message_count: messages.length,
        tool_schema_count: tools.length,
        sections: sections.filter((section) => section.count)
      };
    }

    function sectionFromMessages(kind, label, messages, role) {
      const items = messages.filter((message) => message?.role === role).map((message, index) => ({
        title: `${role} #${index + 1}`,
        text: contentText(message.content) || toolCallText(message.tool_calls)
      }));
      return {kind, label, count: items.length, items};
    }

    function renderInspector() {
      const record = selectedRecord();
      $("tabs").innerHTML = tabs.map((tab) => `
        <button class="tab ${state.tab === tab ? "active" : ""}" data-tab="${tab}">${tab}</button>
      `).join("");
      $("tabs").querySelectorAll("[data-tab]").forEach((button) => {
        button.addEventListener("click", () => {
          state.tab = button.dataset.tab;
          renderInspector();
        });
      });
      if (!record) {
        $("inspect-title").textContent = "Inspector";
        $("inspect-meta").innerHTML = "";
        $("payload-label").textContent = "No payload";
        $("payload").innerHTML = `<div class="empty">Select a trace turn to inspect request, response, raw SSE events, and context delta.</div>`;
        return;
      }
      $("inspect-title").textContent = `Turn ${record.turn ?? state.selectedRecordIndex + 1}`;
      $("inspect-meta").innerHTML = [
        record.transport || "http",
        record.request?.path || "",
        record.upstream_base_url || ""
      ].filter(Boolean).map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("");
      $("payload-label").textContent = state.tab;
      if (state.tab === "overview") {
        renderOverview(record);
      } else if (state.tab === "request") {
        renderRequestInspector(record);
      } else if (state.tab === "response") {
        renderResponseInspector(record);
      } else if (state.tab === "delta") {
        renderDelta(record, previousRecord());
      } else {
        $("payload").innerHTML = `<pre>${escapeHtml(JSON.stringify(currentPayload(), null, 2))}</pre>`;
      }
    }

    function renderRequestInspector(record) {
      resetFullTextRefs();
      const req = record.request?.body || {};
      const messages = Array.isArray(req.messages) ? req.messages : [];
      const tools = Array.isArray(req.tools) ? req.tools : [];
      const promptSections = turnSummary(record, state.selectedRecordIndex).sections?.filter((section) => section.kind !== "response" && section.kind !== "tool_schemas") || [];
      $("payload").innerHTML = `
        <div class="inspector-panel" data-inspector-view="request">
          <section class="inspector-section" data-inspector-section="request-summary">
            <h3>Request summary</h3>
            <div class="inspector-grid">
              ${inspectorStat(req.model || "-", "Model")}
              ${inspectorStat(messages.length, "Prompt messages")}
              ${inspectorStat(tools.length, "Tool schemas")}
              ${inspectorStat(req.tool_choice || "auto", "Tool choice")}
              ${inspectorStat(String(Boolean(req.stream)), "Stream")}
              ${inspectorStat(record.request?.path || "-", "Path")}
            </div>
          </section>
          <section class="inspector-section" data-inspector-section="prompt-messages">
            <h3>Prompt messages</h3>
            <div class="prompt-list">
              ${promptSections.length ? promptSections.map(inspectorPromptSection).join("") : `<div class="empty">No prompt messages captured.</div>`}
            </div>
          </section>
          <section class="inspector-section" data-inspector-section="tool-schemas">
            <h3>Tool schemas</h3>
            <div class="tool-list">
              ${tools.length ? tools.map((tool, index) => toolSchemaCard(tool, index)).join("") : `<div class="empty">No tool schemas captured.</div>`}
            </div>
          </section>
        </div>
      `;
      bindInspectorDetailToggles();
    }

    function renderResponseInspector(record) {
      resetFullTextRefs();
      const response = record.response || {};
      const body = response.body || {};
      const assistant = assistantMessage(body);
      const usage = body.usage || {};
      const toolCalls = assistant?.tool_calls || [];
      const error = body.error || null;
      $("payload").innerHTML = `
        <div class="inspector-panel" data-inspector-view="response">
          <section class="inspector-section" data-inspector-section="response-summary">
            <h3>Response summary</h3>
            <div class="inspector-grid">
              ${inspectorStat(response.status || "-", "Status")}
              ${inspectorStat(record.duration_ms ?? 0, "Duration ms")}
              ${inspectorStat(tokenValue(usage, "prompt_tokens", "input_tokens"), "Input tokens")}
              ${inspectorStat(tokenValue(usage, "completion_tokens", "output_tokens"), "Output tokens")}
              ${inspectorStat(usage.total_tokens ?? "-", "Total tokens")}
              ${inspectorStat(body.id || body.request_id || "-", "Response id")}
            </div>
          </section>
          <section class="inspector-section" data-inspector-section="assistant-response">
            <h3>Assistant response</h3>
            <div class="response-list">
              ${error ? responseErrorCard(error) : ""}
              ${assistant ? responseMessageCard(assistant) : ""}
              ${toolCalls.length ? responseToolCallsCard(toolCalls) : ""}
              ${!error && !assistant && !toolCalls.length ? `<div class="empty">No assistant response body captured.</div>` : ""}
            </div>
          </section>
        </div>
      `;
      bindInspectorDetailToggles();
    }

    function renderOverview(record) {
      const req = record.request?.body || {};
      const res = record.response?.body || {};
      const usage = res.usage || {};
      const data = {
        status: record.response?.status,
        duration_ms: record.duration_ms,
        model: req.model,
        messages: Array.isArray(req.messages) ? req.messages.length : 0,
        tools: Array.isArray(req.tools) ? req.tools.map(toolName).filter(Boolean) : [],
        usage,
        assistant: assistantMessage(res)
      };
      $("payload").innerHTML = `<pre>${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
    }

    function renderDelta(record, previous) {
      const currentMessages = record.request?.body?.messages || [];
      const previousMessages = previous?.request?.body?.messages || [];
      const currentTools = record.request?.body?.tools || [];
      const previousTools = previous?.request?.body?.tools || [];
      const addedMessages = currentMessages.slice(previousMessages.length);
      const toolNames = currentTools.map(toolName).filter(Boolean);
      const previousToolNames = previousTools.map(toolName).filter(Boolean);
      const addedTools = toolNames.filter((name) => !previousToolNames.includes(name));
      $("payload").innerHTML = `
        <div class="delta-list">
          <div class="delta-item"><b>Messages</b>${addedMessages.length} added since previous turn, ${currentMessages.length} total.</div>
          <div class="delta-item"><b>Tools</b>${addedTools.length ? escapeHtml(addedTools.join(", ")) : "No new tool schemas."}</div>
          <div class="delta-item"><b>Context size</b>${JSON.stringify(record.request?.body || {}).length.toLocaleString()} request characters.</div>
          <div class="delta-item"><b>Previous turn</b>${previous ? `Turn ${escapeHtml(previous.turn ?? "")}` : "None"}</div>
        </div>
      `;
    }

    function inspectorStat(value, label) {
      return `<div class="inspector-stat"><b>${escapeHtml(String(value ?? "-"))}</b><span>${escapeHtml(label)}</span></div>`;
    }

    function inspectorPromptSection(section) {
      const items = Array.isArray(section.items) ? section.items : [];
      return items.map((item, index) => promptCard(section.kind, item, index)).join("");
    }

    function promptCard(kind, item, index) {
      const role = promptRole(kind);
      const text = item.text || "";
      const detailKey = inspectorDetailKey("prompt", kind, index);
      return `
        <details class="prompt-card ${escapeAttr(role)}" data-lazy-detail data-detail-key="${escapeAttr(detailKey)}"${inspectorDetailOpenAttr(detailKey)}>
          <summary>${escapeHtml(role)} <span>${escapeHtml(summaryLine(item.title || `#${index + 1}`, text))}</span></summary>
          ${lazyText(text)}
        </details>
      `;
    }

    function promptRole(kind) {
      if (kind === "tool_schemas") return "tool";
      if (kind === "tool_results") return "tool";
      return kind || "message";
    }

    function toolSchemaCard(tool, index) {
      const name = toolName(tool) || `tool #${index + 1}`;
      const description = toolDescription(tool);
      const parameters = tool?.function?.parameters || tool?.parameters || null;
      const parametersText = parameters ? JSON.stringify(parameters, null, 2) : "";
      const detailKey = inspectorDetailKey("tool", index, name);
      return `
        <details class="tool-card" data-lazy-detail data-detail-key="${escapeAttr(detailKey)}"${inspectorDetailOpenAttr(detailKey)}>
          <summary>${escapeHtml(name)} <span>${escapeHtml(description || "No description")}</span></summary>
          ${description ? lazyText(description) : ""}
          ${parameters ? `<div class="tool-param">parameters</div>${lazyText(parametersText)}` : ""}
        </details>
      `;
    }

    function responseMessageCard(message) {
      const text = contentText(message.content) || toolCallText(message.tool_calls);
      const detailKey = inspectorDetailKey("response", "assistant");
      return `
        <details class="response-card assistant" data-lazy-detail data-detail-key="${escapeAttr(detailKey)}"${inspectorDetailOpenAttr(detailKey, true)}>
          <summary>assistant <span>${escapeHtml(message.finish_reason || "message")}</span></summary>
          ${lazyText(text || "")}
        </details>
      `;
    }

    function responseToolCallsCard(toolCalls) {
      const detailKey = inspectorDetailKey("response", "tool_calls");
      return `
        <details class="response-card tool_calls" data-lazy-detail data-detail-key="${escapeAttr(detailKey)}"${inspectorDetailOpenAttr(detailKey, true)}>
          <summary>tool calls <span>${toolCalls.length}</span></summary>
          ${lazyText(toolCallText(toolCalls))}
        </details>
      `;
    }

    function responseErrorCard(error) {
      const message = typeof error === "string" ? error : JSON.stringify(error, null, 2);
      const detailKey = inspectorDetailKey("response", "error");
      return `
        <details class="response-card error" data-lazy-detail data-detail-key="${escapeAttr(detailKey)}"${inspectorDetailOpenAttr(detailKey, true)}>
          <summary>error <span>upstream</span></summary>
          ${lazyText(message)}
        </details>
      `;
    }

    function tokenValue(usage, primary, fallback) {
      return usage?.[primary] ?? usage?.[fallback] ?? "-";
    }

    function resetFullTextRefs() {
      state.fullTextRefs = new Map();
      state.fullTextCounter = 0;
    }

    function rememberFullText(text) {
      const key = `text-${state.fullTextCounter++}`;
      state.fullTextRefs.set(key, String(text ?? ""));
      return key;
    }

    function lazyText(text, limit = TEXT_PREVIEW_LIMIT) {
      const key = rememberFullText(text);
      return `<div class="lazy-text-slot" data-fulltext-ref="${key}" data-preview-limit="${limit}"></div>`;
    }

    function bindInspectorDetailToggles() {
      $("payload").querySelectorAll("[data-lazy-detail]").forEach((details) => {
        const sync = () => {
          const detailKey = details.dataset.detailKey;
          if (detailKey) {
            if (details.open) {
              state.openInspectorDetails.add(detailKey);
              state.closedInspectorDetails.delete(detailKey);
            } else if (state.openInspectorDetails.has(detailKey) || state.closedInspectorDetails.has(detailKey)) {
              state.openInspectorDetails.delete(detailKey);
              state.closedInspectorDetails.add(detailKey);
            }
          }
          details.querySelectorAll("[data-fulltext-ref]").forEach((node) => {
            const fullText = state.fullTextRefs.get(node.dataset.fulltextRef) || "";
            node.innerHTML = details.open
              ? `<pre class="inspector-text">${escapeHtml(fullText)}</pre>`
              : "";
          });
        };
        details.addEventListener("toggle", sync);
        sync();
      });
    }

    function inspectorDetailKey(...parts) {
      return [
        state.selectedSession || "",
        state.selectedRecordIndex,
        state.tab,
        ...parts
      ].map((part) => encodeURIComponent(String(part ?? ""))).join("|");
    }

    function inspectorDetailOpenAttr(detailKey, defaultOpen = false) {
      const open = state.openInspectorDetails.has(detailKey)
        || (defaultOpen && !state.closedInspectorDetails.has(detailKey));
      return open ? " open" : "";
    }

    function sessionFingerprint(session) {
      if (!session) return "";
      return [
        session.id || "",
        session.record_count ?? "",
        session.started_at || "",
        session.updated_at || ""
      ].join("|");
    }

    function previewText(value, limit) {
      const text = String(value ?? "");
      if (text.length <= limit) return text;
      return `${text.slice(0, limit)}\n\n... ${text.length - limit} chars hidden; use the inspector for full text.`;
    }

    function summaryLine(title, text) {
      const compact = String(text || "").replace(/\s+/g, " ").trim();
      const preview = compact.length > 90 ? `${compact.slice(0, 90)}...` : compact;
      return preview ? `${title} - ${preview}` : title;
    }

    function currentPayload() {
      const record = selectedRecord();
      if (!record) return {};
      if (state.tab === "request") return record.request || {};
      if (state.tab === "response") return record.response || {};
      if (state.tab === "events") return record.response?.sse_events || [];
      if (state.tab === "raw") return record;
      if (state.tab === "delta") return {record, previous: previousRecord()};
      return {
        request: record.request?.body,
        response: record.response?.body,
        capture: record.capture,
        duration_ms: record.duration_ms
      };
    }

    function filteredRecords() {
      const records = state.detail?.records || [];
      const query = state.query;
      return records.map((record, index) => ({record, index})).filter(({record}) => {
        if (!query) return true;
        return JSON.stringify(record).toLowerCase().includes(query);
      });
    }
    function selectedRecord() { return state.detail?.records?.[state.selectedRecordIndex] || null; }
    function previousRecord() { return state.detail?.records?.[state.selectedRecordIndex - 1] || null; }
    function assistantMessage(body) {
      const choice = body?.choices?.[0];
      return choice?.message || null;
    }
    function messageRow(role, text, extra = "") {
      return `<div class="message"><div class="role ${escapeAttr(role)}">${escapeHtml(role)}</div><div class="text ${extra}">${escapeHtml(text || "")}</div></div>`;
    }
    function metric(value, label) {
      return `<div class="metric"><b>${escapeHtml(String(value))}</b><span>${escapeHtml(label)}</span></div>`;
    }
    function contentText(content) {
      if (typeof content === "string") return content;
      if (Array.isArray(content)) return content.map((item) => item?.text || item?.type || "").join("\n");
      if (content == null) return "";
      return JSON.stringify(content, null, 2);
    }
    function toolName(tool) { return tool?.function?.name || tool?.name || ""; }
    function toolDescription(tool) { return tool?.function?.description || tool?.description || ""; }
    function toolCallText(toolCalls) {
      if (!Array.isArray(toolCalls)) return "";
      return toolCalls.map((call) => `${call?.function?.name || call?.id || "tool"}(${call?.function?.arguments || ""})`).join("\n");
    }
    function shortId(id) { return id && id.length > 28 ? `${id.slice(0, 12)}...${id.slice(-8)}` : id; }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));
    }
    function escapeAttr(value) { return escapeHtml(value).replace(/\s+/g, "-"); }

    loadSessions();
    setInterval(loadSessions, 3000);
  </script>
</body>
</html>
"""
