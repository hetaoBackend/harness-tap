from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from aiohttp import ClientSession, web

from harness_tap.headers import filter_headers
from harness_tap.proxy import create_app
from harness_tap.store import JsonlTraceStore


def test_filter_headers_redacts_sensitive_values():
    headers = {
        "Authorization": "Bearer secret-token",
        "Cookie": "session=secret",
        "Set-Cookie": "session=secret",
        "X-Api-Key": "sk-secret",
        "Proxy-Authorization": "Basic secret",
        "Connection": "keep-alive",
        "Content-Length": "123",
        "Content-Type": "application/json",
    }

    filtered = filter_headers(headers, redact=True)

    assert filtered["Authorization"] == "***"
    assert filtered["Cookie"] == "***"
    assert filtered["Set-Cookie"] == "***"
    assert filtered["X-Api-Key"] == "***"
    assert filtered["Proxy-Authorization"] == "***"
    assert "Connection" not in filtered
    assert "Content-Length" not in filtered
    assert filtered["Content-Type"] == "application/json"


async def test_proxy_forwards_non_stream_chat_completion_and_records_trace(tmp_path):
    upstream_seen: dict[str, Any] = {}

    async def upstream_handler(request: web.Request) -> web.Response:
        upstream_seen["path"] = request.path
        upstream_seen["host"] = request.headers.get("Host")
        upstream_seen["authorization"] = request.headers.get("Authorization")
        upstream_seen["body"] = await request.json()
        return web.json_response(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "captured"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        )

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", upstream_handler)
    upstream_runner, upstream_url = await _start_app(upstream_app)
    store = JsonlTraceStore(tmp_path)
    proxy_runner, proxy_url = await _start_app(create_app(upstream_base_url=f"{upstream_url}/v1", trace_store=store))

    try:
        async with ClientSession() as session:
            response = await session.post(
                f"{proxy_url}/v1/chat/completions",
                headers={"Authorization": "Bearer real-token", "Content-Type": "application/json"},
                json={
                    "model": "gpt-test",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )
            body = await response.json()

        assert response.status == 200
        assert body["choices"][0]["message"]["content"] == "captured"
        assert upstream_seen["path"] == "/v1/chat/completions"
        assert upstream_seen["host"] == urlparse(upstream_url).netloc
        assert upstream_seen["authorization"] == "Bearer real-token"
        assert upstream_seen["body"]["messages"] == [{"role": "user", "content": "hello"}]

        records = store.load_session(store.session_id)
        assert len(records) == 1
        record = records[0]
        assert record["request"]["path"] == "/v1/chat/completions"
        assert record["request"]["headers"]["Authorization"] == "***"
        assert record["request"]["body"]["messages"][0]["content"] == "hello"
        assert record["response"]["status"] == 200
        assert record["response"]["body"]["usage"]["total_tokens"] == 12
        assert record["upstream_base_url"] == f"{upstream_url}/v1"
        assert isinstance(record["duration_ms"], int)
    finally:
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


async def test_proxy_relays_stream_and_records_reconstructed_response(tmp_path):
    chunks = [
        b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
        b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","model":"gpt-test","choices":[{"index":0,"delta":{"content":"Hello "}}]}\n\n',
        b'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","model":"gpt-test","choices":[{"index":0,"delta":{"content":"stream"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def upstream_handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in chunks:
            await response.write(chunk)
        await response.write_eof()
        return response

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", upstream_handler)
    upstream_runner, upstream_url = await _start_app(upstream_app)
    store = JsonlTraceStore(tmp_path)
    proxy_runner, proxy_url = await _start_app(create_app(upstream_base_url=f"{upstream_url}/v1", trace_store=store))

    try:
        async with ClientSession() as session:
            response = await session.post(
                f"{proxy_url}/v1/chat/completions",
                json={
                    "model": "gpt-test",
                    "messages": [{"role": "user", "content": "stream please"}],
                    "stream": True,
                },
            )
            body = await response.read()

        assert response.status == 200
        assert body == b"".join(chunks)
        assert response.headers["Content-Type"].startswith("text/event-stream")

        records = store.load_session(store.session_id)
        assert len(records) == 1
        record = records[0]
        assert record["transport"] == "http-sse"
        assert record["response"]["sse_events"]
        assert record["response"]["body"]["choices"][0]["message"]["content"] == "Hello stream"
        assert record["response"]["body"]["usage"]["total_tokens"] == 5
    finally:
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


async def test_proxy_rejects_unknown_path_without_forwarding(tmp_path):
    calls = 0

    async def upstream_handler(request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.json_response({"ok": True})

    upstream_app = web.Application()
    upstream_app.router.add_route("*", "/{tail:.*}", upstream_handler)
    upstream_runner, upstream_url = await _start_app(upstream_app)
    store = JsonlTraceStore(tmp_path)
    proxy_runner, proxy_url = await _start_app(create_app(upstream_base_url=f"{upstream_url}/v1", trace_store=store))

    try:
        async with ClientSession() as session:
            response = await session.post(f"{proxy_url}/v1/models", json={})
            text = await response.text()

        assert response.status == 404
        assert text == "Not Found"
        assert calls == 0
        assert store.load_session(store.session_id) == []
    finally:
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


async def test_proxy_rejects_unsupported_method_without_forwarding(tmp_path):
    async def upstream_handler(request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", upstream_handler)
    upstream_runner, upstream_url = await _start_app(upstream_app)
    store = JsonlTraceStore(tmp_path)
    proxy_runner, proxy_url = await _start_app(create_app(upstream_base_url=f"{upstream_url}/v1", trace_store=store))

    try:
        async with ClientSession() as session:
            response = await session.get(f"{proxy_url}/v1/chat/completions")
            text = await response.text()

        assert response.status == 405
        assert text == "Method Not Allowed"
        assert store.load_session(store.session_id) == []
    finally:
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


async def test_proxy_records_invalid_json_rejection(tmp_path):
    upstream_app = web.Application()
    upstream_runner, upstream_url = await _start_app(upstream_app)
    store = JsonlTraceStore(tmp_path)
    proxy_runner, proxy_url = await _start_app(create_app(upstream_base_url=f"{upstream_url}/v1", trace_store=store))

    try:
        async with ClientSession() as session:
            response = await session.post(
                f"{proxy_url}/v1/chat/completions",
                data="{bad json",
                headers={"Content-Type": "application/json"},
            )
            body = await response.json()

        assert response.status == 400
        assert body == {"error": "Invalid JSON request body"}
        records = store.load_session(store.session_id)
        assert len(records) == 1
        assert records[0]["capture"]["rejected"] is True
        assert records[0]["response"]["status"] == 400
    finally:
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


async def test_proxy_relays_and_records_upstream_error(tmp_path):
    async def upstream_handler(request: web.Request) -> web.Response:
        return web.json_response({"error": {"message": "nope"}}, status=429)

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", upstream_handler)
    upstream_runner, upstream_url = await _start_app(upstream_app)
    store = JsonlTraceStore(tmp_path)
    proxy_runner, proxy_url = await _start_app(create_app(upstream_base_url=f"{upstream_url}/v1", trace_store=store))

    try:
        async with ClientSession() as session:
            response = await session.post(f"{proxy_url}/v1/chat/completions", json={"model": "gpt-test"})
            body = await response.json()

        assert response.status == 429
        assert body == {"error": {"message": "nope"}}
        records = store.load_session(store.session_id)
        assert len(records) == 1
        assert records[0]["response"]["status"] == 429
        assert records[0]["response"]["body"] == {"error": {"message": "nope"}}
    finally:
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


async def test_proxy_records_json_error_for_stream_request_when_upstream_is_not_sse(tmp_path):
    async def upstream_handler(request: web.Request) -> web.Response:
        return web.json_response({"error": {"message": "stream quota"}}, status=429)

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", upstream_handler)
    upstream_runner, upstream_url = await _start_app(upstream_app)
    store = JsonlTraceStore(tmp_path)
    proxy_runner, proxy_url = await _start_app(create_app(upstream_base_url=f"{upstream_url}/v1", trace_store=store))

    try:
        async with ClientSession() as session:
            response = await session.post(
                f"{proxy_url}/v1/chat/completions",
                json={"model": "gpt-test", "stream": True},
            )
            body = await response.json()

        assert response.status == 429
        assert body == {"error": {"message": "stream quota"}}
        records = store.load_session(store.session_id)
        assert len(records) == 1
        assert records[0]["transport"] == "http"
        assert records[0]["response"]["body"] == {"error": {"message": "stream quota"}}
    finally:
        await proxy_runner.cleanup()
        await upstream_runner.cleanup()


async def _start_app(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"
