# Harness Tap Design

Date: 2026-07-02
Status: approved for implementation planning

## Context

Harness can point its model API base URL at a custom endpoint, and it uses the
OpenAI Chat Completions protocol. The first version of harness-tap should focus on
observing that one protocol well instead of starting with a multi-provider
abstraction.

Reference points:

- OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat/create
- claude-tap: https://github.com/liaohch3/claude-tap

The useful lesson from claude-tap is the product shape: capture local agent
traffic, persist a durable trace, then make context changes visible through
inspection and diffing. Harness Tap should borrow that shape while keeping its
first module narrow: OpenAI-compatible chat completions only.

## Goals

- Let Harness users configure the app with a local OpenAI-compatible base URL.
- Capture each `POST /v1/chat/completions` request without changing its
  semantics.
- Forward the request to the real upstream API and stream or return the response
  back to Harness.
- Persist enough trace evidence to answer how Harness builds conversation context
  across turns.
- Redact credentials before trace data is stored.
- Keep the first implementation small enough to validate with local tests and a
  sample trace.

## Non-goals

- Supporting Anthropic, Gemini, Responses API, or arbitrary provider adapters in
  the first version.
- Building a polished full dashboard in the first version.
- Mutating prompts, truncating context, injecting tools, or otherwise acting as
  a policy layer.
- Capturing app UI state outside model API traffic.

## Recommended Approach

Build a minimal reverse proxy for OpenAI Chat Completions.

Harness is configured with:

```text
base_url = http://127.0.0.1:<port>/v1
```

Harness Tap is configured with:

```text
upstream_base_url = https://api.openai.com/v1
```

The proxy accepts `POST /v1/chat/completions`, records the request, forwards it
to `<upstream_base_url>/chat/completions`, records the response, and returns the
upstream response to Harness.

This is the deepest first module because callers only need to understand one
interface, while the implementation owns routing, header filtering, streaming
reassembly, persistence, and trace summaries.

## Core Module

Module name:

```text
ChatCompletionTraceProxy
```

Primary interface:

```text
run_proxy(local_host, local_port, upstream_base_url, trace_store)
```

Interface guarantees:

- Requests that are not recognized Chat Completions endpoints are rejected
  locally and are not forwarded.
- Request bodies are forwarded unchanged unless a future compatibility fix is
  explicitly added and documented.
- Sensitive request and response headers are redacted before persistence.
- Streaming responses are relayed to Harness as chunks arrive.
- The trace record is written after the response completes or fails.
- Proxy failures are recorded with status and error details where possible.

Internal responsibilities:

- HTTP server and route allowlist.
- Upstream request construction.
- Header filtering and credential redaction.
- Non-stream response capture.
- SSE streaming relay and reassembly.
- Trace record creation.
- Local trace persistence.

## Trace Record Shape

The first version should persist JSON records. SQLite can come later if the
first storage adapter starts as JSONL, but the record shape should be stable.

```json
{
  "timestamp": "2026-07-02T00:00:00Z",
  "request_id": "req_...",
  "turn": 1,
  "duration_ms": 1234,
  "transport": "http-sse",
  "upstream_base_url": "https://api.openai.com/v1",
  "request": {
    "method": "POST",
    "path": "/v1/chat/completions",
    "headers": {},
    "body": {
      "model": "example-model",
      "messages": [],
      "tools": [],
      "tool_choice": "auto",
      "stream": true
    }
  },
  "response": {
    "status": 200,
    "headers": {},
    "body": {
      "id": "chatcmpl_...",
      "object": "chat.completion",
      "model": "example-model",
      "choices": [],
      "usage": {}
    },
    "sse_events": []
  },
  "capture": {
    "client": "harness",
    "protocol": "openai-chat-completions"
  }
}
```

Fields to prioritize in viewer or inspection output:

- `model`
- `messages`
- `tools`
- `tool_choice`
- `temperature`, `top_p`, `max_tokens`
- `stream`
- assistant response message
- `tool_calls`
- `usage`
- `duration_ms`

## Streaming Reassembly

For `stream: true`, the proxy should parse server-sent events while it relays
the original byte stream to Harness.

Expected OpenAI Chat Completions stream shape:

- Frames are usually `data: {...}` messages.
- The terminal sentinel is `data: [DONE]`.
- Text arrives under `choices[].delta.content`.
- Tool calls may arrive as incremental `choices[].delta.tool_calls` fragments.
- Final usage may appear when the upstream includes it.

The reassembler should produce a best-effort final response body:

- Preserve raw SSE events when configured.
- Accumulate assistant content per choice index.
- Accumulate tool call id, type, function name, and function arguments per tool
  call index.
- Preserve finish reasons.
- Preserve usage when present.

If a stream terminates early, the trace should still contain the partial
reconstruction and an error or incomplete marker.

## Data Flow

```text
Harness
  -> POST http://127.0.0.1:<port>/v1/chat/completions
  -> ChatCompletionTraceProxy
  -> POST <upstream_base_url>/chat/completions
  -> upstream model provider
  -> response or SSE stream
  -> ChatCompletionTraceProxy records trace
  -> Harness receives the original provider-shaped response
```

The proxy is observational by default. The only intended behavior difference is
that traffic is stored locally for inspection.

## Storage

Start with a `TraceStore` interface rather than exposing file or database
details to the proxy.

```text
TraceStore.append(record)
TraceStore.list_sessions()
TraceStore.load_session(session_id)
```

First adapter:

```text
JsonlTraceStore
```

Default path:

```text
.harness-tap/traces/<date>/<session-id>.jsonl
```

This keeps the first implementation easy to inspect and easy to test. A SQLite
adapter can replace it later behind the same interface if dashboard filtering or
large-session performance becomes important.

## Security And Privacy

Headers to redact before writing:

- `authorization`
- `cookie`
- `set-cookie`
- `x-api-key`
- `proxy-authorization`

The proxy should not log raw headers to stdout. Trace files are local artifacts
and may include prompts, source snippets, tool arguments, and model outputs, so
the CLI should make the trace output path obvious.

The proxy should bind to `127.0.0.1` by default. Binding to `0.0.0.0` should
require an explicit flag.

## Error Handling

- Unknown path: return `404` without forwarding.
- Unsupported method: return `405` without forwarding.
- Invalid JSON request body: return `400` and record a rejected trace event.
- Upstream connect failure: return `502` and record the upstream error.
- Upstream non-2xx: relay status and body, then record the failed response.
- Stream interruption: close the downstream stream and record a partial trace.

## Testing Strategy

Test through module interfaces rather than internal helpers whenever possible.

Core tests:

- Non-stream request is forwarded and recorded.
- Stream request relays chunks and reconstructs final assistant content.
- Stream tool calls are reconstructed from incremental deltas.
- Sensitive headers are redacted in stored records.
- Unknown paths are rejected and not forwarded.
- Upstream errors are relayed and recorded.

Use an in-process fake upstream server for tests. This creates a real seam:
production uses a real provider, tests use a local adapter.

## Implementation Milestones

1. Project scaffold and CLI entrypoint.
2. `TraceStore` interface and JSONL adapter.
3. Chat Completions reverse proxy for non-stream requests.
4. SSE relay and reassembler.
5. Minimal inspect command to print captured messages and response summaries.
6. Tests with fake upstream.
7. Optional simple HTML viewer or export after the trace format proves useful.

## Open Assumptions

- Harness includes `/v1` in its configured base URL behavior or can be configured
  with `http://127.0.0.1:<port>/v1`.
- Harness sends standard OpenAI Chat Completions request bodies.
- The upstream provider is OpenAI-compatible enough to accept the forwarded
  request unchanged.
- The first useful output is a local trace and text inspection command; a full
  dashboard can follow after the capture path is trusted.
