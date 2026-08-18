from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Sequence

from harness_tap.context import inspect_turn_lines
from harness_tap.proxy import run_proxy
from harness_tap.store import JsonlTraceStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness-tap")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Run the local multi-protocol trace proxy.")
    serve.add_argument(
        "--upstream",
        required=True,
        help="Upstream API base URL, for example https://api.openai.com/v1 or https://api.anthropic.com.",
    )
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
        for line in inspect_turn_lines(record):
            print(line)
    return 0


async def _serve(host: str, port: int, upstream: str, trace_dir: Path) -> int:
    store = JsonlTraceStore(trace_dir)
    runner, actual_port = await run_proxy(
        local_host=host,
        local_port=port,
        upstream_base_url=upstream,
        trace_store=store,
    )
    print(f"Local proxy base URL: http://{host}:{actual_port}/v1")
    print("Supported endpoints: /v1/chat/completions, /v1/responses, /v1/messages")
    print(f"Trace viewer: http://{host}:{actual_port}/viewer")
    print(f"Trace directory: {trace_dir}")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        return 0
    finally:
        await runner.cleanup()
    return 0
