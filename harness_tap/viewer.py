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
    button:focus-visible, input:focus-visible, select:focus-visible, summary:focus-visible {
      outline: 2px solid var(--focus);
      outline-offset: 2px;
    }
    .trace-shell {
      display: grid;
      grid-template-columns: 232px minmax(320px, .95fr) minmax(0, 1.2fr);
      height: 100vh;
      min-width: 0;
    }
    .sessions, .timeline, .inspector {
      min-height: 0;
      min-width: 0;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      border-right: 1px solid var(--line);
      background: var(--panel-2);
    }
    .timeline { grid-template-rows: auto auto auto minmax(0, 1fr); }
    .inspector { grid-template-rows: auto minmax(0, 1fr); border-right: 0; background: #081016; }
    .trace-shell * { min-width: 0; }
    .chip, .main-title, .metric b, .item-title, .turn-top strong { overflow-wrap: anywhere; }
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
      min-height: 0;
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
      font-size: 16px;
      line-height: 1.15;
    }
    .metric span {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }
    .turns {
      min-height: 0;
      overflow: auto;
      padding: 12px 14px 28px;
      display: block;
      overflow-anchor: none;
    }
    .turn-card {
      border: 1px solid var(--line);
      background: #0b151c;
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 10px;
    }
    .turn-card.active { border-color: var(--cyan); }
    .turn-top { width: 100%; text-align: left; color: var(--ink); border: 0; border-radius: 0; }
    .turn-controls { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--line); }
    .turn-controls select { flex: 1; width: 100%; background: var(--panel); color: var(--ink); border: 1px solid var(--line); padding: 6px; border-radius: 6px; }
    .turn-controls button[aria-pressed="true"] { border-color: var(--amber); color: var(--amber); }
    .turn-count { width: 100%; color: var(--muted); font-size: 12px; }
    .turn-preview { margin: 8px 12px; color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .context-toggle { margin: 0 12px 10px; background: transparent; border: 0; color: var(--cyan); padding: 4px 0; cursor: pointer; font-size: 12px; }
    button:disabled { opacity: .45; cursor: default; }
    .pulse.offline { background: var(--red); box-shadow: none; }
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
    .turn-top .status-error { color: var(--red); }
    .turn-top .status-ok { color: var(--green); }
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
      padding: 12px;
      background: #071017;
    }
    #payload {
      min-height: 0;
      overflow: auto;
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
    .prompt-card summary::before, .tool-card summary::before, .response-card summary::before { content: "▸"; flex: 0 0 auto; }
    details[open] > summary::before { content: "▾"; }
    .prompt-card summary { justify-content: flex-start; }
    .message-label { flex: 0 0 auto; white-space: nowrap; }
    .prompt-card summary span { flex: 1; }
    .prompt-card summary span, .tool-card summary span, .response-card summary span {
      color: var(--muted);
      font-weight: 600;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .prompt-card.system summary, .prompt-card.developer summary { color: var(--amber); }
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
      height: auto;
      overflow: visible;
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
      .sessions { grid-template-rows: auto auto auto; }
      .list { max-height: 180px; }
      .timeline { grid-template-rows: auto auto auto auto; }
      .turns { height: 340px; }
      .inspector { height: 85vh; min-height: 520px; }
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
        <span class="pulse" id="connection-status" title="Connecting"></span>
      </header>
      <div class="toolbar">
        <input id="search" class="search" type="search" aria-label="Search turns in current session" placeholder="Search turns in this session">
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
      <nav class="turn-controls" aria-label="Turn navigation">
        <button class="copy-btn" id="previous-turn" aria-label="Previous turn">←</button>
        <select id="jump-turn" aria-label="Jump to turn"></select>
        <button class="copy-btn" id="next-turn" aria-label="Next turn">→</button>
        <button class="copy-btn" id="errors-only" aria-pressed="false">Errors only</button>
        <span class="turn-count" id="turn-count" role="status"></span>
      </nav>
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
        <div id="payload" tabindex="0" aria-label="Turn payload"></div>
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
      detailRequestId: 0,
      detailLoading: false,
      sessionsLoading: false,
      errorsOnly: false,
      tab: "request",
      query: ""
    };
    const $ = (id) => document.getElementById(id);
    const tabs = ["overview", "request", "response", "events", "delta", "raw"];
    const TEXT_PREVIEW_LIMIT = 900;
    const TIMELINE_PREVIEW_LIMIT = 260;

    $("search").addEventListener("input", (event) => {
      state.query = event.target.value.toLowerCase();
      renderTurns();
    });
    $("previous-turn").addEventListener("click", () => moveTurn(-1));
    $("next-turn").addEventListener("click", () => moveTurn(1));
    $("jump-turn").addEventListener("change", (event) => selectTurn(Number(event.target.value), true));
    $("errors-only").addEventListener("click", () => {
      state.errorsOnly = !state.errorsOnly;
      $("errors-only").setAttribute("aria-pressed", String(state.errorsOnly));
      renderTurns();
    });
    $("copy").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(JSON.stringify(currentPayload(), null, 2));
        $("payload-label").textContent = "Copied";
      } catch {
        $("payload-label").textContent = "Copy failed. Select and copy the raw JSON.";
      }
    });

    async function loadSessions() {
      if (state.sessionsLoading) return;
      state.sessionsLoading = true;
      try {
        const response = await fetch("/api/sessions", {cache: "no-store"});
        if (!response.ok) throw new Error(`Sessions: HTTP ${response.status}`);
        const body = await response.json();
        state.sessions = body.sessions || [];
        $("session-count").textContent = `${state.sessions.length} sessions`;
        if (!state.selectedSession && state.sessions.length) {
          await loadDetail(state.sessions[0].id);
        } else if (state.selectedSession) {
          const selectedSummary = state.sessions.find((session) => session.id === state.selectedSession);
          if (selectedSummary && !state.detailLoading && (!state.detail || sessionFingerprint(selectedSummary) !== state.detailFingerprint)) {
            await loadDetail(state.selectedSession, Boolean(state.detail));
          } else if (!selectedSummary) {
            ++state.detailRequestId;
            state.selectedSession = "";
            state.detail = null;
            renderAll();
            if (state.sessions.length) await loadDetail(state.sessions[0].id);
          }
        }
        setConnectionStatus(true);
        renderSessions();
      } catch (error) {
        setConnectionStatus(false, error.message);
      } finally {
        state.sessionsLoading = false;
      }
    }

    async function loadDetail(sessionId, keepSelection = false) {
      const requestId = ++state.detailRequestId;
      state.detailLoading = true;
      state.selectedSession = sessionId;
      if (!keepSelection) {
        state.selectedRecordIndex = 0;
        state.detail = null;
        state.expandedRecords = new Set();
        state.openInspectorDetails = new Set();
        state.closedInspectorDetails = new Set();
        state.query = "";
        state.errorsOnly = false;
        $("search").value = "";
        $("errors-only").setAttribute("aria-pressed", "false");
        renderAll();
        $("session-title").textContent = "Loading session…";
      }
      try {
        const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {cache: "no-store"});
        if (!response.ok) throw new Error(`Session: HTTP ${response.status}`);
        const detail = await response.json();
        if (requestId !== state.detailRequestId) return;
        state.detail = detail;
        state.detailFingerprint = sessionFingerprint(detail.session);
        state.selectedRecordIndex = Math.max(0, Math.min(state.selectedRecordIndex, detail.records.length - 1));
        renderAll();
        setConnectionStatus(true);
      } catch (error) {
        if (requestId !== state.detailRequestId) return;
        if (!keepSelection) $("session-title").textContent = "Unable to load session. Select it to retry.";
        setConnectionStatus(false, error.message);
        throw error;
      } finally {
        if (requestId === state.detailRequestId) state.detailLoading = false;
      }
    }

    function setConnectionStatus(ok, message = "") {
      $("refresh-label").textContent = ok ? "auto refresh · 3s" : "refresh failed · retrying";
      $("connection-status").classList.toggle("offline", !ok);
      $("connection-status").title = ok ? "Connected" : message;
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
      const focusedSession = list.contains(document.activeElement) ? document.activeElement.dataset.session : null;
      const sessions = state.sessions;
      list.innerHTML = sessions.length ? sessions.map((session) => `
        <button class="session-item ${session.id === state.selectedSession ? "active" : ""}" data-session="${escapeAttr(session.id)}" title="${escapeAttr(session.id)}">
          <span class="item-title"><span>${escapeHtml(shortId(session.id))}</span><span>${session.record_count}</span></span>
          <span class="item-sub">${escapeHtml(session.updated_at || session.started_at || "no timestamp")}</span>
        </button>
      `).join("") : `<div class="empty">No trace sessions yet. Send a request through the proxy and this list will fill in.</div>`;
      list.querySelectorAll("[data-session]").forEach((button) => {
        button.addEventListener("click", () => loadDetail(button.dataset.session).catch(() => {}));
        if (button.dataset.session === focusedSession) button.focus({preventScroll: true});
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
      $("session-title").title = summary.id;
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
        metric(summary.record_count.toLocaleString(), "Turns"),
        metric((summary.total_tokens || 0).toLocaleString(), "Tokens"),
        metric(formatDuration(summary.total_duration_ms || 0), "Total duration"),
        metric(summary.models?.[0] || "-", "Model"),
      ].join("");
    }

    function renderTurns() {
      const records = filteredRecords();
      const host = $("turns");
      const scrollTop = host.scrollTop;
      const focused = host.contains(document.activeElement) ? document.activeElement.dataset : null;
      $("turn-count").textContent = `${records.length} / ${state.detail?.records?.length || 0} turns${state.detail && !records.some(({index}) => index === state.selectedRecordIndex) ? " · selected turn is filtered out" : ""}`;
      $("jump-turn").innerHTML = `<option value="" disabled ${records.some(({index}) => index === state.selectedRecordIndex) ? "" : "selected"}>Jump to turn</option>` + records.map(({record, index}) => `<option value="${index}" ${index === state.selectedRecordIndex ? "selected" : ""}>Turn ${escapeHtml(record.turn ?? index + 1)}</option>`).join("");
      const position = records.findIndex(({index}) => index === state.selectedRecordIndex);
      $("previous-turn").disabled = position <= 0;
      $("next-turn").disabled = !records.length || position === records.length - 1;
      $("jump-turn").disabled = !records.length;
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
          selectTurn(Number(button.dataset.record));
        });
      });
      host.querySelectorAll("[data-expand-record]").forEach((button) => {
        button.addEventListener("click", () => {
          const index = Number(button.dataset.expandRecord);
          if (state.expandedRecords.has(index)) {
            state.expandedRecords.delete(index);
          } else {
            state.expandedRecords.add(index);
          }
          renderTurns();
        });
      });
      host.scrollTop = scrollTop;
      if (focused) {
        const selector = focused.expandRecord !== undefined ? `[data-expand-record="${focused.expandRecord}"]` : `[data-record="${focused.record}"]`;
        host.querySelector(selector)?.focus({preventScroll: true});
      }
    }

    function moveTurn(direction) {
      const records = filteredRecords();
      const position = records.findIndex(({index}) => index === state.selectedRecordIndex);
      const target = records[position + direction];
      if (target) selectTurn(target.index, true);
    }

    function selectTurn(index, reveal = false) {
      state.selectedRecordIndex = index;
      renderTurns();
      $("payload").scrollTop = 0;
      renderInspector();
      if (reveal) $("turns").querySelector(`[data-record="${index}"]`)?.closest(".turn-card").scrollIntoView({block: "nearest"});
    }

    function turnCard(record, index) {
      const req = record.request?.body || {};
      const status = record.response?.status || 0;
      const selected = index === state.selectedRecordIndex ? "active" : "";
      const expanded = state.expandedRecords.has(index);
      const summary = expanded ? turnSummary(record) : {
        message_count: Array.isArray(req.messages) ? req.messages.length : 0,
        tool_schema_count: Array.isArray(req.tools) ? req.tools.length : 0
      };
      const contextRows = !expanded ? "" : summary.sections?.length
        ? summary.sections.map((section) => contextSection(section, expanded)).join("")
        : `<div class="text dim">No messages captured.</div>`;
      return `
        <article class="turn-card ${selected} ${expanded ? "expanded" : ""}">
          <button class="turn-top turn-item ${selected}" data-record="${index}" aria-pressed="${selected ? "true" : "false"}">
            <span class="turn-top-left">
              <strong>Turn ${escapeHtml(record.turn ?? index + 1)} - ${escapeHtml(req.model || "model unknown")}</strong>
            </span>
            <span class="turn-top-right">
              <span>${summary.message_count || 0} messages</span>
              <span>${summary.tool_schema_count || 0} tools</span>
              <span class="${status >= 400 ? "status-error" : "status-ok"}">${status || "-"} / ${formatDuration(record.duration_ms ?? 0)}</span>
            </span>
          </button>
          <p class="turn-preview">${escapeHtml(turnPreview(record))}</p>
          <button class="context-toggle" data-expand-record="${index}" aria-expanded="${expanded}" aria-controls="context-${index}">${expanded ? "▾ Hide context" : "▸ Show context"}</button>
          <div id="context-${index}" ${expanded ? 'class="context-stack"' : 'hidden'}>${contextRows}</div>
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

    function turnSummary(record) {
      const req = record.request?.body || {};
      const res = record.response?.body || {};
      const messages = Array.isArray(req.messages) ? req.messages : [];
      const tools = Array.isArray(req.tools) ? req.tools : [];
      const sections = [
        {kind: "messages", label: "Request order", count: messages.length, items: messages.map((message, index) => ({title: `#${index + 1} ${message?.role || "message"}`, text: messageText(message)}))},
        {kind: "tool_schemas", label: "Tool schemas", count: tools.length, items: tools.map((tool) => ({title: toolName(tool), text: toolDescription(tool)}))},
      ];
      const assistant = assistantMessage(res);
      if (assistant) {
        sections.push({kind: "response", label: "Upstream response", count: 1, items: [{title: "assistant", text: messageText(assistant)}]});
      }
      return {
        message_count: messages.length,
        tool_schema_count: tools.length,
        sections: sections.filter((section) => section.count)
      };
    }

    function turnPreview(record) {
      const body = record.response?.body || {};
      if (body.error) return typeof body.error === "string" ? body.error : JSON.stringify(body.error);
      const assistant = assistantMessage(body);
      if (assistant) return summaryLine("Response", messageText(assistant));
      const messages = record.request?.body?.messages;
      const last = Array.isArray(messages) ? messages.at(-1) : null;
      return last ? summaryLine(last.role || "message", messageText(last)) : "No message content captured";
    }

    function renderInspector() {
      const record = selectedRecord();
      const scrollTop = $("payload").scrollTop;
      const focusedTab = $("tabs").contains(document.activeElement) ? document.activeElement.dataset.tab : null;
      $("copy").disabled = !record;
      $("tabs").innerHTML = tabs.map((tab) => `
        <button class="tab ${state.tab === tab ? "active" : ""}" data-tab="${tab}" aria-pressed="${state.tab === tab}">${tab}</button>
      `).join("");
      $("tabs").querySelectorAll("[data-tab]").forEach((button) => {
        button.addEventListener("click", () => {
          state.tab = button.dataset.tab;
          $("payload").scrollTop = 0;
          renderInspector();
        });
        if (button.dataset.tab === focusedTab) button.focus({preventScroll: true});
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
      $("payload").scrollTop = scrollTop;
    }

    function renderRequestInspector(record) {
      resetFullTextRefs();
      const req = record.request?.body || {};
      const messages = Array.isArray(req.messages) ? req.messages : [];
      const tools = Array.isArray(req.tools) ? req.tools : [];
      $("payload").innerHTML = `
        <div class="inspector-panel" data-inspector-view="request">
          <section class="inspector-section" data-inspector-section="request-summary">
            <h3>Request summary</h3>
            <div class="inspector-grid">
              ${inspectorStat(req.model || "-", "Model")}
              ${inspectorStat(messages.length, "Prompt messages")}
              ${inspectorStat(tools.length, "Tool schemas")}
              ${inspectorStat(req.tool_choice ?? "auto", "Tool choice")}
              ${inspectorStat(String(Boolean(req.stream)), "Stream")}
              ${inspectorStat(record.request?.path || "-", "Path")}
            </div>
          </section>
          <section class="inspector-section" data-inspector-section="prompt-messages">
            <h3>Prompt messages · original request order</h3>
            <div class="prompt-list">
              ${messages.length ? messages.map((message, index) => promptCard(message, index)).join("") : `<div class="empty">No prompt messages captured.</div>`}
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
      const asArray = (value) => Array.isArray(value) ? value : [];
      const currentMessages = asArray(record.request?.body?.messages);
      const previousMessages = asArray(previous?.request?.body?.messages);
      let prefix = 0;
      while (prefix < Math.min(currentMessages.length, previousMessages.length) && stableJson(currentMessages[prefix]) === stableJson(previousMessages[prefix])) prefix++;
      const messageDelta = prefix === previousMessages.length
        ? `${currentMessages.length - prefix} added since previous turn, ${currentMessages.length} total.`
        : `Context replaced or truncated after ${prefix} unchanged messages: ${previousMessages.length - prefix} previous messages → ${currentMessages.length - prefix} current messages. This is not an append-only change.`;
      const currentTools = asArray(record.request?.body?.tools);
      const previousTools = asArray(previous?.request?.body?.tools);
      const toolsChanged = stableJson(currentTools) !== stableJson(previousTools);
      $("payload").innerHTML = `
        <div class="delta-list">
          <div class="delta-item"><b>Messages</b>${messageDelta}</div>
          <div class="delta-item"><b>Tools</b>${toolsChanged ? `Tool schemas changed (definitions, order, additions or removals). ${previousTools.length} → ${currentTools.length} schemas.` : "Tool schemas unchanged."}</div>
          <div class="delta-item"><b>Context size</b>${JSON.stringify(record.request?.body || {}).length.toLocaleString()} request characters.</div>
          <div class="delta-item"><b>Previous turn</b>${previous ? `Turn ${escapeHtml(previous.turn ?? "")}` : "None"}</div>
        </div>
      `;
    }

    function inspectorStat(value, label) {
      const text = typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "-");
      return `<div class="inspector-stat"><b title="${escapeAttr(text)}">${escapeHtml(text)}</b><span>${escapeHtml(label)}</span></div>`;
    }

    function promptCard(message, index) {
      const role = message?.role || "message";
      const text = messageText(message);
      const detailKey = inspectorDetailKey("prompt", index);
      return `
        <details class="prompt-card ${escapeAttr(role)}" data-message-index="${index}" data-lazy-detail data-detail-key="${escapeAttr(detailKey)}"${inspectorDetailOpenAttr(detailKey)}>
          <summary><b class="message-label">#${index + 1} ${escapeHtml(role)}</b><span>${escapeHtml(summaryLine(`messages[${index}]`, text))}</span></summary>
          ${lazyText(text)}
        </details>
      `;
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
      const {tool_calls, ...content} = message;
      const text = messageText(content);
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
          ${lazyText(JSON.stringify(toolCalls, null, 2))}
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
          if (!details.isConnected) return;
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
        if (state.errorsOnly && !(record.response?.status >= 400)) return false;
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
      if (Array.isArray(content)) return content.map((item) => typeof item === "string" ? item : JSON.stringify(item, null, 2)).join("\n\n");
      if (content == null) return "";
      return JSON.stringify(content, null, 2);
    }
    function toolName(tool) { return tool?.function?.name || tool?.name || ""; }
    function toolDescription(tool) { return tool?.function?.description || tool?.description || ""; }
    function messageText(message) {
      if (!message || typeof message !== "object") return JSON.stringify(message) ?? "";
      const parts = [contentText(message.content)];
      const metadata = Object.fromEntries(Object.entries(message).filter(([key]) => key !== "role" && key !== "content"));
      if (Object.keys(metadata).length) parts.push(JSON.stringify(metadata, null, 2));
      return parts.filter(Boolean).join("\n\n");
    }
    function stableJson(value) {
      if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
      if (value && typeof value === "object") return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(",")}}`;
      return JSON.stringify(value);
    }
    function formatDuration(ms) {
      if (ms < 1000) return `${ms}ms`;
      if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
      return `${Math.floor(ms / 60000)}m ${Math.floor(ms % 60000 / 1000)}s`;
    }
    function shortId(id) { return id && id.length > 28 ? `${id.slice(0, 12)}...${id.slice(-8)}` : id; }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));
    }
    function escapeAttr(value) { return escapeHtml(value); }

    renderAll();
    loadSessions();
    setInterval(loadSessions, 3000);
  </script>
</body>
</html>
"""
