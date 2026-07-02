from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Sequence

from harness_tap.proxy import run_proxy
from harness_tap.store import JsonlTraceStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness-tap")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the local Chat Completions trace proxy.")
    serve.add_argument("--upstream", required=True, help="Upstream OpenAI-compatible base URL.")
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
        return asyncio.run(_serve(args.host, args.port, args.upstream, args.trace_dir))
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
        response_body = _dict_at(record, "response", "body")
        print(
            f"Turn {record.get('turn')}: "
            f"model={request_body.get('model', '')} "
            f"duration_ms={record.get('duration_ms', '')}"
        )
        for message in request_body.get("messages", []):
            if isinstance(message, dict):
                print(f"{message.get('role', 'message')}: {_content_text(message.get('content'))}")
        for tool in request_body.get("tools", []):
            name = _tool_name(tool)
            if name:
                print(f"tool: {name}")
        assistant = _assistant_message(response_body)
        if assistant:
            print(f"assistant: {_content_text(assistant.get('content'))}")
        usage = response_body.get("usage")
        if isinstance(usage, dict):
            print(
                "usage: "
                f"prompt={usage.get('prompt_tokens', usage.get('input_tokens', 0))} "
                f"completion={usage.get('completion_tokens', usage.get('output_tokens', 0))} "
                f"total={usage.get('total_tokens', 0)}"
            )
    return 0


async def _serve(host: str, port: int, upstream: str, trace_dir: Path) -> int:
    store = JsonlTraceStore(trace_dir)
    runner, actual_port = await run_proxy(
        local_host=host,
        local_port=port,
        upstream_base_url=upstream,
        trace_store=store,
    )
    print(f"Local OpenAI-compatible base URL: http://{host}:{actual_port}/v1")
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


def _tool_name(tool: object) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    name = tool.get("name")
    return name if isinstance(name, str) else ""


def _assistant_message(response_body: dict) -> dict:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    return message if isinstance(message, dict) else {}


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""
