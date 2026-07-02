from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from aiohttp import ClientError, ClientSession, web

from harness_tap.headers import filter_headers
from harness_tap.sse import ChatCompletionSSEReassembler
from harness_tap.store import TraceStore
from harness_tap.viewer import install_viewer_routes

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
UPSTREAM_BASE_URL_KEY = web.AppKey("upstream_base_url", str)
TRACE_STORE_KEY = web.AppKey("trace_store", TraceStore)
STATE_KEY = web.AppKey("state", dict[str, int])


def create_app(*, upstream_base_url: str, trace_store: TraceStore) -> web.Application:
    app = web.Application()
    app[UPSTREAM_BASE_URL_KEY] = upstream_base_url.rstrip("/")
    app[TRACE_STORE_KEY] = trace_store
    app[STATE_KEY] = {"turn_counter": 0}
    install_viewer_routes(app, TRACE_STORE_KEY)
    app.router.add_route("*", "/{tail:.*}", proxy_handler)
    return app


async def run_proxy(
    *,
    local_host: str,
    local_port: int,
    upstream_base_url: str,
    trace_store: TraceStore,
) -> tuple[web.AppRunner, int]:
    app = create_app(upstream_base_url=upstream_base_url, trace_store=trace_store)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, local_host, local_port)
    await site.start()
    if site._server is None:
        return runner, local_port
    actual_port = int(site._server.sockets[0].getsockname()[1])
    return runner, actual_port


async def proxy_handler(request: web.Request) -> web.StreamResponse:
    if request.path != CHAT_COMPLETIONS_PATH:
        return web.Response(status=404, text="Not Found")
    if request.method != "POST":
        return web.Response(status=405, text="Method Not Allowed")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    state = request.app[STATE_KEY]
    state["turn_counter"] += 1
    turn = state["turn_counter"]
    trace_store: TraceStore = request.app[TRACE_STORE_KEY]
    upstream_base_url: str = request.app[UPSTREAM_BASE_URL_KEY]

    try:
        request_body = await request.json()
    except json.JSONDecodeError:
        record = _base_record(
            request=request,
            request_id=request_id,
            turn=turn,
            timestamp=started_at,
            duration_ms=_duration_ms(started),
            upstream_base_url=upstream_base_url,
            request_body=None,
        )
        record["response"] = {"status": 400, "headers": {}, "body": {"error": "Invalid JSON request body"}}
        record["capture"]["rejected"] = True
        trace_store.append(record)
        return web.json_response({"error": "Invalid JSON request body"}, status=400)

    upstream_url = f"{upstream_base_url}/chat/completions"
    forward_headers = filter_headers(request.headers, redact=False)
    try:
        async with ClientSession() as session:
            async with session.post(upstream_url, headers=forward_headers, json=request_body) as upstream_response:
                if _is_stream_response(request_body, upstream_response.headers.get("Content-Type", "")):
                    return await _relay_stream_response(
                        request=request,
                        upstream_response=upstream_response,
                        request_id=request_id,
                        turn=turn,
                        timestamp=started_at,
                        started=started,
                        upstream_base_url=upstream_base_url,
                        request_body=request_body,
                        trace_store=trace_store,
                    )
                response_bytes = await upstream_response.read()
                response_body = _parse_response_body(response_bytes)
                record = _base_record(
                    request=request,
                    request_id=request_id,
                    turn=turn,
                    timestamp=started_at,
                    duration_ms=_duration_ms(started),
                    upstream_base_url=upstream_base_url,
                    request_body=request_body,
                )
                record["response"] = {
                    "status": upstream_response.status,
                    "headers": filter_headers(upstream_response.headers, redact=True),
                    "body": response_body,
                }
                trace_store.append(record)
                return web.Response(
                    body=response_bytes,
                    status=upstream_response.status,
                    headers=filter_headers(upstream_response.headers, redact=False),
                )
    except ClientError as exc:
        record = _base_record(
            request=request,
            request_id=request_id,
            turn=turn,
            timestamp=started_at,
            duration_ms=_duration_ms(started),
            upstream_base_url=upstream_base_url,
            request_body=request_body,
        )
        record["response"] = {"status": 502, "headers": {}, "body": {"error": str(exc)}}
        trace_store.append(record)
        return web.json_response({"error": str(exc)}, status=502)


def _base_record(
    *,
    request: web.Request,
    request_id: str,
    turn: int,
    timestamp: str,
    duration_ms: int,
    upstream_base_url: str,
    request_body: Any,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "request_id": request_id,
        "turn": turn,
        "duration_ms": duration_ms,
        "transport": "http",
        "upstream_base_url": upstream_base_url,
        "request": {
            "method": request.method,
            "path": request.path,
            "headers": filter_headers(request.headers, redact=True),
            "body": request_body,
        },
        "response": {},
        "capture": {
            "client": "harness",
            "protocol": "openai-chat-completions",
        },
    }


def _duration_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _parse_response_body(response_bytes: bytes) -> Any:
    if not response_bytes:
        return None
    try:
        return json.loads(response_bytes)
    except json.JSONDecodeError:
        return response_bytes.decode("utf-8", errors="replace")


def _is_stream_response(request_body: Any, content_type: str) -> bool:
    return content_type.lower().split(";", 1)[0].strip() == "text/event-stream"


async def _relay_stream_response(
    *,
    request: web.Request,
    upstream_response: Any,
    request_id: str,
    turn: int,
    timestamp: str,
    started: float,
    upstream_base_url: str,
    request_body: Any,
    trace_store: TraceStore,
) -> web.StreamResponse:
    reassembler = ChatCompletionSSEReassembler()
    downstream = web.StreamResponse(
        status=upstream_response.status,
        headers=filter_headers(upstream_response.headers, redact=False),
    )
    await downstream.prepare(request)
    partial = False
    try:
        async for chunk in upstream_response.content.iter_chunked(8192):
            reassembler.feed_bytes(chunk)
            await downstream.write(chunk)
    except Exception:
        partial = True
        raise
    finally:
        record = _base_record(
            request=request,
            request_id=request_id,
            turn=turn,
            timestamp=timestamp,
            duration_ms=_duration_ms(started),
            upstream_base_url=upstream_base_url,
            request_body=request_body,
        )
        record["transport"] = "http-sse"
        record["response"] = {
            "status": upstream_response.status,
            "headers": filter_headers(upstream_response.headers, redact=True),
            "body": reassembler.final_response(),
            "sse_events": reassembler.events,
        }
        if partial or not reassembler.done:
            record["capture"]["partial"] = True
        trace_store.append(record)
    await downstream.write_eof()
    return downstream
