# Harness Tap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a working OpenAI Chat Completions reverse proxy that captures Harness request/response context into local JSONL traces and provides a minimal inspection command.

**Architecture:** The package exposes a small proxy module that owns HTTP routing, upstream forwarding, header redaction, SSE relay/reassembly, and trace writing. Persistence sits behind a `TraceStore` interface with a JSONL adapter, and the CLI composes the proxy and store without leaking storage details into request handling.

**Tech Stack:** Python 3.11+, aiohttp, pytest, pytest-asyncio.

---

## File Structure

- `pyproject.toml`: package metadata, console script, runtime and test dependencies, pytest configuration.
- `harness_tap/__init__.py`: package version.
- `harness_tap/__main__.py`: `python -m harness_tap` entrypoint.
- `harness_tap/cli.py`: `serve` and `inspect` command handling.
- `harness_tap/headers.py`: hop-by-hop filtering and sensitive-header redaction.
- `harness_tap/store.py`: `TraceStore` protocol and `JsonlTraceStore`.
- `harness_tap/sse.py`: OpenAI Chat Completions SSE parser and final response reassembler.
- `harness_tap/proxy.py`: aiohttp reverse proxy and `run_proxy` interface.
- `tests/test_store.py`: JSONL persistence tests.
- `tests/test_sse.py`: streaming reassembly tests.
- `tests/test_proxy.py`: fake-upstream proxy tests.
- `tests/test_cli.py`: CLI inspect tests.

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `harness_tap/__init__.py`
- Create: `harness_tap/__main__.py`
- Create: `harness_tap/cli.py`

- [x] **Step 1: Write a failing CLI smoke test**

Create `tests/test_cli.py`:

```python
from harness_tap.cli import build_parser


def test_parser_accepts_serve_defaults():
    parser = build_parser()

    args = parser.parse_args(["serve", "--upstream", "https://api.openai.com/v1"])

    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 0
    assert args.upstream == "https://api.openai.com/v1"
```

- [x] **Step 2: Run the test and verify it fails**

Run:

```bash
python -m pytest tests/test_cli.py::test_parser_accepts_serve_defaults -q
```

Expected: FAIL because `harness_tap` does not exist.

- [x] **Step 3: Add minimal package scaffold**

Create `pyproject.toml`, `harness_tap/__init__.py`, `harness_tap/__main__.py`, and `harness_tap/cli.py` with a parser that supports `serve --upstream --host --port --trace-dir`.

- [x] **Step 4: Run the smoke test and verify it passes**

Run:

```bash
python -m pytest tests/test_cli.py::test_parser_accepts_serve_defaults -q
```

Expected: PASS.

## Task 2: JSONL Trace Store

**Files:**
- Create: `harness_tap/store.py`
- Test: `tests/test_store.py`

- [x] **Step 1: Write failing JSONL append/load tests**

Create tests for:

- `append(record)` creates `.harness-tap/traces/<date>/<session-id>.jsonl`.
- `list_sessions()` returns newest sessions with counts.
- `load_session(session_id)` returns records in append order.

- [x] **Step 2: Run store tests and verify they fail**

Run:

```bash
python -m pytest tests/test_store.py -q
```

Expected: FAIL because `JsonlTraceStore` does not exist.

- [x] **Step 3: Implement `TraceStore` and `JsonlTraceStore`**

Implement:

```text
JsonlTraceStore(base_dir).append(record)
JsonlTraceStore(base_dir).list_sessions()
JsonlTraceStore(base_dir).load_session(session_id)
```

The adapter assigns a session id when missing, uses the record timestamp for the date directory when available, writes compact JSON lines, and returns session metadata with `id`, `path`, `record_count`, `started_at`, and `updated_at`.

- [x] **Step 4: Run store tests and verify they pass**

Run:

```bash
python -m pytest tests/test_store.py -q
```

Expected: PASS.

## Task 3: Header Redaction

**Files:**
- Create: `harness_tap/headers.py`
- Test: `tests/test_proxy.py`

- [x] **Step 1: Write failing header redaction tests**

In `tests/test_proxy.py`, add a test that passes headers containing `Authorization`, `Cookie`, `Set-Cookie`, `X-Api-Key`, `Proxy-Authorization`, `Connection`, and `Content-Length`, then asserts sensitive values are `***` and hop-by-hop headers are removed.

- [x] **Step 2: Run the header test and verify it fails**

Run:

```bash
python -m pytest tests/test_proxy.py::test_filter_headers_redacts_sensitive_values -q
```

Expected: FAIL because `filter_headers` does not exist.

- [x] **Step 3: Implement `filter_headers`**

Implement case-insensitive filtering and redaction in `harness_tap/headers.py`.

- [x] **Step 4: Run the header test and verify it passes**

Run:

```bash
python -m pytest tests/test_proxy.py::test_filter_headers_redacts_sensitive_values -q
```

Expected: PASS.

## Task 4: SSE Reassembler

**Files:**
- Create: `harness_tap/sse.py`
- Test: `tests/test_sse.py`

- [x] **Step 1: Write failing text streaming test**

Create a test that feeds OpenAI Chat Completions SSE frames containing `choices[0].delta.role`, `choices[0].delta.content`, a finish reason, and `[DONE]`, then asserts `final_response()` returns a chat completion with an assistant message containing the accumulated text.

- [x] **Step 2: Run text streaming test and verify it fails**

Run:

```bash
python -m pytest tests/test_sse.py::test_reassembler_accumulates_streamed_content -q
```

Expected: FAIL because `ChatCompletionSSEReassembler` does not exist.

- [x] **Step 3: Implement text streaming reassembly**

Implement line-buffered SSE parsing, raw event preservation, `[DONE]` handling, role/content accumulation, finish reason preservation, and a provider-shaped final chat completion response.

- [x] **Step 4: Run text streaming test and verify it passes**

Run:

```bash
python -m pytest tests/test_sse.py::test_reassembler_accumulates_streamed_content -q
```

Expected: PASS.

- [x] **Step 5: Write failing tool-call streaming test**

Add a test that feeds incremental `choices[0].delta.tool_calls[0].function.arguments` chunks and asserts final `message.tool_calls[0].function.arguments` is concatenated in order.

- [x] **Step 6: Run tool-call test and verify it fails**

Run:

```bash
python -m pytest tests/test_sse.py::test_reassembler_accumulates_streamed_tool_calls -q
```

Expected: FAIL until tool-call accumulation is implemented.

- [x] **Step 7: Implement tool-call reassembly**

Accumulate tool calls by choice index and tool call index, preserving `id`, `type`, `function.name`, and concatenated `function.arguments`.

- [x] **Step 8: Run all SSE tests and verify they pass**

Run:

```bash
python -m pytest tests/test_sse.py -q
```

Expected: PASS.

## Task 5: Reverse Proxy

**Files:**
- Create: `harness_tap/proxy.py`
- Test: `tests/test_proxy.py`

- [x] **Step 1: Write failing non-stream proxy test**

Use an aiohttp fake upstream that accepts `/chat/completions`, records the forwarded JSON and headers, and returns a normal OpenAI chat completion response. Start the proxy against that upstream, send `POST /v1/chat/completions`, then assert the client receives the upstream body and the store contains a redacted trace record with request messages, response body, usage, duration, and upstream base URL.

- [x] **Step 2: Run non-stream proxy test and verify it fails**

Run:

```bash
python -m pytest tests/test_proxy.py::test_proxy_forwards_non_stream_chat_completion_and_records_trace -q
```

Expected: FAIL because the proxy does not exist.

- [x] **Step 3: Implement non-stream proxy**

Implement `create_app`, `run_proxy`, route allowlisting, upstream URL construction, JSON parsing, non-stream forwarding, error records, and trace writing.

- [x] **Step 4: Run non-stream proxy test and verify it passes**

Run:

```bash
python -m pytest tests/test_proxy.py::test_proxy_forwards_non_stream_chat_completion_and_records_trace -q
```

Expected: PASS.

- [x] **Step 5: Write failing stream proxy test**

Use a fake upstream that returns `text/event-stream` Chat Completions chunks. Assert the proxy relays the exact streamed chunks to the client and writes a trace whose reconstructed response has accumulated assistant content plus `sse_events`.

- [x] **Step 6: Run stream proxy test and verify it fails**

Run:

```bash
python -m pytest tests/test_proxy.py::test_proxy_relays_stream_and_records_reconstructed_response -q
```

Expected: FAIL until streaming is implemented.

- [x] **Step 7: Implement stream relay**

Relay chunks as they arrive, feed chunks into `ChatCompletionSSEReassembler`, write a trace after upstream completion, and mark interrupted streams as partial when an exception occurs.

- [x] **Step 8: Add path/method/error tests**

Add tests for unknown paths returning 404 without forwarding, unsupported methods returning 405, invalid JSON returning 400 with a rejected trace, and upstream non-2xx responses being relayed and recorded.

- [x] **Step 9: Run proxy tests and verify they pass**

Run:

```bash
python -m pytest tests/test_proxy.py -q
```

Expected: PASS.

## Task 6: CLI Serve And Inspect

**Files:**
- Modify: `harness_tap/cli.py`
- Modify: `harness_tap/__main__.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Write failing inspect test**

Add a test that writes a JSONL trace containing request messages, tools, a response message, usage, and duration, then runs `main(["inspect", "--trace-dir", tmp_path, session_id])` and asserts stdout includes the model, message roles, tool names, usage, and duration.

- [x] **Step 2: Run inspect test and verify it fails**

Run:

```bash
python -m pytest tests/test_cli.py::test_inspect_prints_context_summary -q
```

Expected: FAIL until inspect is implemented.

- [x] **Step 3: Implement inspect command**

Implement `inspect` by loading records from `JsonlTraceStore`, printing model, turn, message roles, tool/function names, assistant output, usage, and duration.

- [x] **Step 4: Run CLI tests and verify they pass**

Run:

```bash
python -m pytest tests/test_cli.py -q
```

Expected: PASS.

- [x] **Step 5: Wire serve command**

Implement `serve` so it creates a `JsonlTraceStore`, starts the proxy, prints the local base URL and trace directory, and runs until interrupted.

## Task 7: Documentation And Verification

**Files:**
- Create: `README.md`
- Modify: `docs/superpowers/plans/2026-07-02-harness-tap-implementation.md`

- [x] **Step 1: Add README usage**

Document installation, starting the proxy, configuring Harness base URL as `http://127.0.0.1:<port>/v1`, and inspecting a captured session.

- [x] **Step 2: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS.

- [x] **Step 3: Run lint/compile verification**

Run:

```bash
python -m compileall harness_tap tests
```

Expected: exit 0.

- [x] **Step 4: Review spec coverage**

Re-read `docs/superpowers/specs/2026-07-02-harness-tap-design.md` and verify every goal, security item, error handling item, and core test item has code or tests.

- [x] **Step 5: Commit implementation**

Run:

```bash
git add .
git commit -m "Implement Harness Tap chat completions proxy"
```
