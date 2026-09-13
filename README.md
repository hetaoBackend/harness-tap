# Harness Tap

Harness Tap is a local reverse proxy for inspecting how harnesses and agent
runtimes build conversation context. It forwards native OpenAI Chat Completions,
OpenAI Responses, and Anthropic Messages requests to a compatible upstream,
records JSONL traces, and provides a browser viewer and CLI inspector.

## Install For Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Run The Proxy

Use one upstream API directory for all supported routes:

```bash
.venv/bin/harness-tap serve \
  --upstream https://api.openai.com/v1 \
  --port 8080
```

Or override individual protocols to send them to different upstreams:

```bash
.venv/bin/harness-tap serve \
  --upstream https://api.openai.com/v1 \
  --responses-upstream https://api.openai.com/v1 \
  --messages-upstream https://api.anthropic.com/v1 \
  --port 8080
```

An upstream is an **API directory**, including its version or gateway prefix.
For example, `https://gateway.example/anthropic/v1` receives Messages requests at
`https://gateway.example/anthropic/v1/messages`. Overrides fall back to
`--upstream` when omitted. Each configured upstream must support that route's
native protocol; Harness Tap does not convert between APIs.

| Client endpoint | Upstream suffix | Protocol |
| --- | --- | --- |
| `POST /v1/chat/completions` | `/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | `/responses` | OpenAI Responses |
| `POST /v1/messages` | `/messages` | Anthropic Messages |

All three support JSON and SSE responses. Query parameters, client credentials,
provider headers (including `anthropic-version` and `anthropic-beta`), request
body bytes, response status, and response body bytes are forwarded. Hop-by-hop
headers and content length are handled by the proxy. Redirects are returned to
the client rather than followed by the proxy. The proxy does not retry requests
or maintain an upstream cookie session.

### Configure Clients

For OpenAI SDK clients, use the local API directory:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1")
response = client.responses.create(model="YOUR_MODEL", input="Hello")
# client.chat.completions.create(...) uses the same base URL.
```

For Anthropic SDK clients, use the local origin; the SDK adds `/v1/messages`:

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8080")
message = client.messages.create(
    model="YOUR_MODEL",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello"}],
)
```

Clients continue supplying their own upstream credentials, such as through
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Use a model supported by the selected
upstream. Setting `stream=True` uses the same proxy endpoints.

The initial scope is the three creation endpoints above. Responses WebSocket
mode, retrieval/cancellation, input-items, compaction, Anthropic token counting,
files, and batches are not implemented. Clients using those additional APIs
need corresponding route support; pointing an entire agent at the proxy does
not imply every operation that agent uses is supported.

## View And Inspect Traces

Open [the local trace viewer](http://127.0.0.1:8080/viewer) while the proxy runs.
Trace files are written to:

```text
.harness-tap/traces/<date>/<session-id>.jsonl
```

```bash
.venv/bin/harness-tap inspect --trace-dir .harness-tap/traces <session-id>
```

The viewer and CLI share a read-only protocol projection. Requests display
`instructions`/`input` for Responses, top-level `system` and `messages` for
Anthropic, and `messages` for Chat Completions. The viewer preserves original
order and source paths. Expand items and tool schemas to see their complete
native fields, including tool IDs, arguments, signatures, and unknown blocks.
Responses `output` items and Anthropic `content` blocks remain native in storage.

The timeline provides previous/next controls, a turn selector, text search, and
an errors-only filter. Filtering includes HTTP errors, stream errors, failed or
incomplete Responses, and capture failures. Live refresh preserves the selected
turn, expanded messages, and scroll position. Raw and events tabs expose the
stored record and SSE events without inserting the display projection.

Delta compares local input prefixes, instructions, and tool schemas. It declines
to compare across protocols, incomplete request captures, or Responses requests
referencing server-side history with `previous_response_id` or `conversation`.
Those references are displayed explicitly: the current request alone does not
reveal the complete server-side context.

## Capture Semantics And Limits

New traces use `schema_version: 2` and identify `capture.protocol`. Old Chat
Completions traces continue to load. Both JSON responses and SSE-reconstructed
responses retain the provider's native structure. Events retain their JSON
`data`, `raw_data`, event name, and supplied SSE metadata. Comments and unknown
or malformed events are retained. Chat's `[DONE]` sentinel is recorded as the
terminal event rather than as a JSON event.

- Chat Completions reconstructs messages and tool calls, ending at `[DONE]`.
- Responses reconstructs output items and deltas for partial streams. A
  `response.completed`, `response.failed`, or `response.incomplete` event's
  response object is the authoritative final snapshot.
- Messages reconstructs content blocks, tool input JSON, thinking/signatures,
  stop reason, and usage through `message_stop`. Output usage updates are
  cumulative and are merged rather than added repeatedly.

`capture.terminal_event`, `capture.model_status`, `capture.partial`, parse errors,
and truncation flags distinguish model outcome from capture completeness. A
fully received `response.incomplete` is not marked as a network truncation.
Transport failures after response headers close the stream and record one
partial trace, without attempting to send another HTTP response. Parse failures
degrade capture while forwarding continues.

| Option | Default | Behavior |
| --- | --- | --- |
| `--max-capture-bytes` | 16777216 (16 MiB) | Decoded capture budget per request and response; forwarding continues after the budget is exhausted. |
| `--max-request-bytes` | 33554432 (32 MiB) | Incoming wire-body limit; larger requests receive 413. |
| `--read-timeout` | 300 seconds | Upstream read inactivity timeout; streams have no total duration limit. |

Gzip and deflate response/request bodies are forwarded with their original
encoding and decoded separately for capture. Other encodings pass through with
an explicit capture decoding error. Raw JSON/SSE capture is bounded; budget
truncation is labelled, and a complete reconstruction is not claimed.

Usage preserves all native counters. Anthropic's displayed input counter is
uncached input; derived total tokens include cache creation and cache reads
once, plus output. Nested cache or reasoning detail counters are not added a
second time. See [Anthropic's token breakdown](https://platform.claude.com/docs/en/build-with-claude/prompt-caching#tracking-cache-performance).

Sensitive headers such as `Authorization`, `Cookie`, `Set-Cookie`, `X-Api-Key`,
and `Proxy-Authorization` are redacted before records are written. Request and
response bodies remain inspectable as part of the trace.

## Run Tests

```bash
.venv/bin/python -m pip install -e '.[dev,browser]'
.venv/bin/python -m playwright install chromium
.venv/bin/python -m pytest -q
```

Local integration tests exercise all three protocols against simulated
upstreams, including streaming latency, exact body forwarding, headers, query
parameters, compression, errors, capture budgets, timeouts, and disconnects.
Browser tests check native protocol display, tool details, history references,
error filters, layout, and existing viewer navigation/refresh behavior. Without
the `browser` extra, browser tests are skipped. These tests do not require API
keys and do not substitute for a live provider/client compatibility check.
