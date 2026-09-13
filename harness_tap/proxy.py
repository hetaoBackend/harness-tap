from __future__ import annotations

import asyncio
import json
import time
import uuid
import zlib
from datetime import datetime, timezone
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, DummyCookieJar, web
from yarl import URL

from harness_tap.headers import filter_headers
from harness_tap.protocols import CHAT, MESSAGES, RESPONSES, ROUTES
from harness_tap.sse import SSEReassembler, create_reassembler
from harness_tap.store import TraceStore
from harness_tap.viewer import install_viewer_routes

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
UPSTREAM_BASE_URL_KEY = web.AppKey("upstream_base_url", str)
UPSTREAMS_KEY = web.AppKey("upstreams", dict)
TRACE_STORE_KEY = web.AppKey("trace_store", TraceStore)
STATE_KEY = web.AppKey("state", dict[str, int])
SESSION_KEY = web.AppKey("http_session", ClientSession)
CAPTURE_LIMIT_KEY = web.AppKey("capture_limit", int)
DEFAULT_CAPTURE_LIMIT = 16 * 1024 * 1024
DEFAULT_REQUEST_LIMIT = 32 * 1024 * 1024


def _api_base(value: str) -> str:
    url = URL(value)
    if url.scheme not in {"http", "https"} or not url.host or url.query_string or url.fragment or url.raw_user is not None:
        raise ValueError("Upstream must be an HTTP(S) API directory without credentials, query, or fragment")
    return value.rstrip("/")


def create_app(
    *, upstream_base_url: str, trace_store: TraceStore,
    responses_upstream: str | None = None, messages_upstream: str | None = None,
    max_capture_bytes: int = DEFAULT_CAPTURE_LIMIT,
    max_request_bytes: int = DEFAULT_REQUEST_LIMIT, read_timeout: float = 300,
) -> web.Application:
    if min(max_capture_bytes, max_request_bytes, read_timeout) <= 0:
        raise ValueError("Capture limit, request limit, and read timeout must be positive")
    base = _api_base(upstream_base_url)
    app = web.Application(client_max_size=max_request_bytes, handler_args={"auto_decompress": False})
    app[UPSTREAM_BASE_URL_KEY] = base
    app[UPSTREAMS_KEY] = {
        CHAT: base,
        RESPONSES: _api_base(responses_upstream) if responses_upstream else base,
        MESSAGES: _api_base(messages_upstream) if messages_upstream else base,
    }
    app[TRACE_STORE_KEY] = trace_store
    app[STATE_KEY] = {"turn_counter": 0}
    app[CAPTURE_LIMIT_KEY] = max_capture_bytes

    async def http_session(application: web.Application):
        async with ClientSession(
            timeout=ClientTimeout(total=None, sock_connect=30, sock_read=read_timeout),
            auto_decompress=False, cookie_jar=DummyCookieJar(),
            skip_auto_headers={"Content-Type", "Accept-Encoding"},
        ) as session:
            application[SESSION_KEY] = session
            yield

    app.cleanup_ctx.append(http_session)
    install_viewer_routes(app, TRACE_STORE_KEY)
    app.router.add_route("*", "/{tail:.*}", proxy_handler)
    return app


async def run_proxy(
    *, local_host: str, local_port: int, upstream_base_url: str, trace_store: TraceStore,
    responses_upstream: str | None = None, messages_upstream: str | None = None,
    max_capture_bytes: int = DEFAULT_CAPTURE_LIMIT,
    max_request_bytes: int = DEFAULT_REQUEST_LIMIT, read_timeout: float = 300,
) -> tuple[web.AppRunner, int]:
    app = create_app(
        upstream_base_url=upstream_base_url, trace_store=trace_store,
        responses_upstream=responses_upstream, messages_upstream=messages_upstream,
        max_capture_bytes=max_capture_bytes, max_request_bytes=max_request_bytes, read_timeout=read_timeout,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, local_host, local_port)
    await site.start()
    actual_port = int(site._server.sockets[0].getsockname()[1]) if site._server else local_port
    return runner, actual_port


class BodyCapture:
    """Decode an observation copy with a bounded *decompressed* byte budget."""

    def __init__(self, encoding: str, limit: int, reassembler: SSEReassembler | None = None) -> None:
        self.limit = limit
        self.size = 0
        self.truncated = False
        self.errors: list[str] = []
        self.buffer = bytearray()
        self.reassembler = reassembler
        self.decoder = None
        encoding = encoding.lower().strip()
        if encoding in {"gzip", "deflate"}:
            self.decoder = zlib.decompressobj(31 if encoding == "gzip" else 15)
        elif encoding not in {"", "identity"}:
            self.errors.append(f"Unsupported capture content encoding: {encoding}")

    def feed(self, chunk: bytes) -> None:
        if self.truncated or self.errors:
            return
        try:
            remaining = self.limit - self.size
            data = self.decoder.decompress(chunk, remaining + 1) if self.decoder else chunk
            if len(data) > remaining:
                data = data[:remaining]
                self.truncated = True
            self.size += len(data)
            if self.reassembler:
                self.reassembler.feed_bytes(data)
            else:
                self.buffer.extend(data)
        except Exception as exc:
            self.errors.append(f"Capture decoding failed: {type(exc).__name__}")

    def finish(self) -> None:
        if self.decoder and not self.truncated and not self.errors:
            if not self.decoder.eof or self.decoder.unused_data:
                self.errors.append("Incomplete or trailing compressed body")
        if self.reassembler and not self.truncated and not self.errors:
            try:
                self.reassembler.finish()
            except Exception as exc:
                self.errors.append(f"Capture finalization failed: {type(exc).__name__}")

    def body(self) -> Any:
        if self.reassembler:
            try:
                return self.reassembler.final_response()
            except Exception as exc:
                self.errors.append(f"Capture reconstruction failed: {type(exc).__name__}")
                return None
        return _parse_response_body(bytes(self.buffer))


def _parse_response_body(data: bytes) -> Any:
    if not data:
        return None
    try:
        return json.loads(data)
    except (ValueError, UnicodeError):
        return data.decode("utf-8", errors="replace")


async def proxy_handler(request: web.Request) -> web.StreamResponse:
    route = ROUTES.get(request.path)
    if route is None:
        return web.Response(status=404, text="Not Found")
    if request.method != "POST":
        return web.Response(status=405, text="Method Not Allowed", headers={"Allow": "POST"})

    started = time.monotonic()
    state = request.app[STATE_KEY]
    state["turn_counter"] += 1
    base = request.app[UPSTREAMS_KEY][route.protocol]
    query = request.rel_url.raw_query_string
    upstream_url = URL(base + route.suffix + ("?" + query if query else ""), encoded=True)
    record: dict[str, Any] = {
        "schema_version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": f"req_{uuid.uuid4().hex[:12]}",
        "turn": state["turn_counter"], "transport": "http", "upstream_base_url": base,
        "request": {"method": request.method, "path": request.path, "query_string": query,
                    "headers": filter_headers(request.headers, redact=True), "body": None},
        "response": {}, "capture": {"client": "harness", "protocol": route.protocol},
    }
    downstream: web.StreamResponse | None = None
    observation: BodyCapture | None = None
    reassembler = None
    try:
        raw_request = await request.read()
        request_capture = BodyCapture(request.headers.get("Content-Encoding", ""), request.app[CAPTURE_LIMIT_KEY])
        request_capture.feed(raw_request)
        request_capture.finish()
        record["request"]["body"] = request_capture.body()
        if request_capture.errors:
            record["capture"]["request_parse_errors"] = request_capture.errors
        if request_capture.truncated:
            record["capture"]["request_truncated"] = True
        # Preserve the existing invalid-JSON rejection for fully captured bodies.
        if not request_capture.errors and not request_capture.truncated:
            try:
                json.loads(request_capture.buffer)
            except (ValueError, UnicodeError):
                record["capture"]["rejected"] = True
                record["request"]["body"] = None
                record["response"] = {"status": 400, "headers": {}, "body": {"error": "Invalid JSON request body"}}
                return web.json_response(record["response"]["body"], status=400)

        async with request.app[SESSION_KEY].post(
            upstream_url, headers=filter_headers(request.headers), data=raw_request, allow_redirects=False,
        ) as upstream:
            is_sse = upstream.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() == "text/event-stream"
            reassembler = create_reassembler(route.protocol) if is_sse else None
            observation = BodyCapture(upstream.headers.get("Content-Encoding", ""), request.app[CAPTURE_LIMIT_KEY], reassembler)
            record["transport"] = "http-sse" if is_sse else "http"
            record["response"] = {"status": upstream.status, "headers": filter_headers(upstream.headers, redact=True)}
            downstream = web.StreamResponse(status=upstream.status, headers=filter_headers(upstream.headers))
            await downstream.prepare(request)
            async for chunk in upstream.content.iter_chunked(8192):
                await downstream.write(chunk)
                observation.feed(chunk)
            observation.finish()
            await downstream.write_eof()
            return downstream
    except web.HTTPException as exc:
        record["capture"]["rejected"] = True
        record["response"] = {"status": exc.status, "headers": {}, "body": {"error": exc.reason}}
        raise
    except (ClientError, TimeoutError, ConnectionError) as exc:
        record["capture"]["transport_error"] = type(exc).__name__
        if downstream is not None and downstream.prepared:
            record["capture"]["partial"] = True
            # HTTP headers already went out. Close the incomplete entity instead
            # of emitting a second response or a false clean chunked EOF.
            if request.transport is not None:
                request.transport.close()
            return downstream
        status = 504 if isinstance(exc, TimeoutError) else 502
        record["response"] = {"status": status, "headers": {}, "body": {"error": type(exc).__name__}}
        return web.json_response(record["response"]["body"], status=status)
    except asyncio.CancelledError:
        record["capture"]["partial"] = True
        record["capture"]["transport_error"] = "CancelledError"
        raise
    finally:
        if observation is not None:
            record["response"]["body"] = observation.body()
            if observation.truncated:
                record["capture"].update({"response_truncated": True, "partial": True})
            if observation.errors:
                record["capture"]["parse_errors"] = observation.errors
        if reassembler is not None:
            record["response"]["sse_events"] = reassembler.events
            capture_state = reassembler.capture_state()
            parse_errors = record["capture"].get("parse_errors", []) + capture_state.pop("parse_errors", [])
            record["capture"].update(capture_state)
            if parse_errors:
                record["capture"]["parse_errors"] = parse_errors
            if not reassembler.done:
                record["capture"]["partial"] = True
        record["duration_ms"] = int((time.monotonic() - started) * 1000)
        request.app[TRACE_STORE_KEY].append(record)
