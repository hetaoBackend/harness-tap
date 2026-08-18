# Harness Tap

Harness Tap is a local reverse proxy for inspecting how configurable harnesses
and agent runtimes build conversation context. Point a Chat Completions,
Responses, or Anthropic Messages client at the local proxy, let the proxy
forward requests to your real upstream, then inspect the JSONL trace on disk or
in the desktop viewer.

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

Configure your harness or agent runtime's base URL as:

```text
http://127.0.0.1:8080/v1
```

Open the trace viewer while the proxy is running:

```text
http://127.0.0.1:8080/viewer
```

The proxy captures these POST endpoints and forwards them to the matching
upstream path:

| Client request | Upstream request |
| --- | --- |
| `/v1/chat/completions` | `<upstream>/chat/completions` |
| `/v1/responses` | `<upstream>/responses` |
| `/v1/messages` | `<upstream>/messages` |

For Anthropic Messages, point `--upstream` at the Anthropic API root:

```bash
.venv/bin/harness-tap serve \
  --upstream https://api.anthropic.com \
  --port 8080
```

Then configure the Anthropic client base URL as `http://127.0.0.1:8080`. The
SDK continues to send `POST /v1/messages`.

Trace files are written under:

```text
.harness-tap/traces/<date>/<session-id>.jsonl
```

## Inspect A Session

```bash
.venv/bin/harness-tap inspect --trace-dir .harness-tap/traces <session-id>
```

The inspect command prints each turn's protocol, model, messages, tools,
assistant output, usage, and duration so you can quickly see what context the
client sent.

## Viewer

The desktop viewer shows a chronological conversation instead of grouping every
system, user, and tool message into separate buckets. Session cards include the
API protocol and model, turns expand in place, and the inspector has structured
request/response views plus raw JSON.

Keyboard shortcuts: `J` and `K` move between turns. Use the Conversation /
Grouped toggle when you want the older role-grouped layout.

## What Is Captured

- Request model, messages or Responses `input`, Anthropic `system` blocks,
  tools, tool choice, generation parameters, and stream flag.
- Non-stream Chat Completions, Responses, and Anthropic Messages responses.
- Streaming SSE chunks, plus a reconstructed final assistant message, tool
  calls, or Responses output items.
- Usage fields when the upstream returns them.
- Duration, status, and upstream error bodies.

Sensitive headers such as `Authorization`, `Cookie`, `Set-Cookie`, `X-Api-Key`,
and `Proxy-Authorization` are redacted before records are written.
