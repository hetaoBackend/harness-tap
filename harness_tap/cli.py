from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Sequence

from harness_tap.proxy import DEFAULT_CAPTURE_LIMIT, DEFAULT_REQUEST_LIMIT, run_proxy
from harness_tap.projection import project_record, tool_name
from harness_tap.store import JsonlTraceStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness-tap")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the local multi-protocol trace proxy.")
    serve.add_argument("--upstream", required=True, help="Default upstream API directory, including /v1 when needed.")
    serve.add_argument("--responses-upstream", help="Override the Responses API directory.")
    serve.add_argument("--messages-upstream", help="Override the Anthropic Messages API directory.")
    serve.add_argument("--max-capture-bytes", type=_positive_int, default=DEFAULT_CAPTURE_LIMIT,
                       help="Maximum decoded bytes captured per request/response (default: 16 MiB).")
    serve.add_argument("--max-request-bytes", type=_positive_int, default=DEFAULT_REQUEST_LIMIT,
                       help="Maximum incoming request size (default: 32 MiB).")
    serve.add_argument("--read-timeout", type=_positive_int, default=300,
                       help="Upstream read inactivity timeout in seconds; no total stream timeout.")
    serve.add_argument("--host", default="127.0.0.1", help="Local bind host.")
    serve.add_argument("--port", default=0, type=int, help="Local bind port. 0 chooses a free port.")
    serve.add_argument(
        "--trace-dir",
        default=Path(".harness-tap") / "traces",
        type=Path,
        help="Directory where JSONL traces are written.",
    )

    inspect = subparsers.add_parser("inspect", help="Print a readable summary of a captured trace session.")
    inspect.add_argument("--trace-dir", default=Path(".harness-tap") / "traces", type=Path)
    inspect.add_argument("session_id")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.trace_dir, args.session_id)
    if args.command == "serve":
        return asyncio.run(_serve(
            args.host, args.port, args.upstream, args.trace_dir,
            responses_upstream=args.responses_upstream, messages_upstream=args.messages_upstream,
            max_capture_bytes=args.max_capture_bytes, max_request_bytes=args.max_request_bytes,
            read_timeout=args.read_timeout,
        ))
    return 0


def _inspect(trace_dir: Path, session_id: str) -> int:
    store = JsonlTraceStore(trace_dir)
    records = store.load_session(session_id)
    if not records:
        print(f"No records found for session: {session_id}")
        return 1

    print(f"Session: {session_id}")
    for record in records:
        request_body = _dict_at(record, "request", "body")
        print(
            f"Turn {record.get('turn')}: "
            f"model={request_body.get('model', '')} "
            f"duration_ms={record.get('duration_ms', '')}"
        )
        projection = project_record(record)
        print(f"protocol: {projection['protocol']}")
        if projection["context_note"]:
            print(f"context: {projection['context_note']}")
        for item in projection["input_items"]:
            print(f"{item['role']}: {item['text']}")
        for tool in projection["tools"]:
            print(f"tool: {tool_name(tool)}")
        for item in projection["output_items"]:
            print(f"assistant: {item['text']}")
        usage = projection["usage"]
        if usage["raw"]:
            print(f"usage: prompt={usage['input_tokens']} completion={usage['output_tokens']} total={usage['total_tokens']}")
            print(f"usage details: {usage['raw']}")
        if projection["outcome"]["is_error"]:
            print(f"outcome: {projection['outcome']}")
    return 0


async def _serve(host: str, port: int, upstream: str, trace_dir: Path, **options) -> int:
    store = JsonlTraceStore(trace_dir)
    runner, actual_port = await run_proxy(
        local_host=host,
        local_port=port,
        upstream_base_url=upstream,
        trace_store=store,
        **options,
    )
    print(f"Local API base URL: http://{host}:{actual_port}/v1")
    print(f"Trace viewer: http://{host}:{actual_port}/viewer")
    print(f"Trace directory: {trace_dir}")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 0
    finally:
        await runner.cleanup()
    return 0


def _dict_at(record: dict, *keys: str) -> dict:
    current = record
    for key in keys:
        value = current.get(key) if isinstance(current, dict) else None
        if not isinstance(value, dict):
            return {}
        current = value
    return current


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed
