# Harness Tap

Harness Tap is a local OpenAI Chat Completions reverse proxy for inspecting how
configurable harnesses and agent runtimes build conversation context. Point any
OpenAI-compatible client at the local proxy, let the proxy forward requests to
your real upstream, then inspect the JSONL trace on disk.

## Install For Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Run The Proxy

```bash
.venv/bin/harness-tap serve \
  --upstream https://api.openai.com/v1 \
  --port 8080
```

Configure your harness or agent runtime's OpenAI-compatible base URL as:

```text
http://127.0.0.1:8080/v1
```

Open the trace viewer while the proxy is running:

```text
http://127.0.0.1:8080/viewer
```

The client should continue sending `POST /v1/chat/completions`. Harness Tap
forwards that request to:

```text
https://api.openai.com/v1/chat/completions
```

Trace files are written under:

```text
.harness-tap/traces/<date>/<session-id>.jsonl
```

## Inspect A Session

```bash
.venv/bin/harness-tap inspect --trace-dir .harness-tap/traces <session-id>
```

The inspect command prints each turn's model, messages, tools, assistant output,
usage, and duration so you can quickly see what context the client sent.

In the viewer, select a turn to inspect its request. Prompt messages follow the
original `messages` array order, with both a message number and its JSON index.
Expand a message for its complete content and metadata, including tool call IDs.
Tool schemas appear separately after the messages.

The timeline provides previous/next controls, a turn selector, text search and an
errors-only filter. Selecting a turn and expanding its context are independent.
Live refresh preserves the selected turn, expanded messages and scroll position.
Delta compares the shared message prefix and reports context replacement or
truncation, as well as tool schema changes.

## Run Tests

```bash
.venv/bin/python -m pip install -e '.[dev,browser]'
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest -q
```

The optional browser suite checks actual layout, message order, long sessions,
refresh state, filtering and session-switch races. Without the `browser` extra,
those tests are skipped; the API and proxy tests still run.

## What Is Captured

- Request model, messages, tools, tool choice, generation parameters, and stream
  flag.
- Non-stream Chat Completions responses.
- Streaming SSE chunks, plus a reconstructed final assistant message and tool
  calls.
- Usage fields when the upstream returns them.
- Duration, status, and upstream error bodies.

Sensitive headers such as `Authorization`, `Cookie`, `Set-Cookie`, `X-Api-Key`,
and `Proxy-Authorization` are redacted before records are written.
