from harness_tap.cli import build_parser
from harness_tap.cli import main
from harness_tap.store import JsonlTraceStore


def test_parser_accepts_serve_defaults():
    parser = build_parser()

    args = parser.parse_args(["serve", "--upstream", "https://api.openai.com/v1"])

    assert args.command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 0
    assert args.upstream == "https://api.openai.com/v1"
    assert str(args.trace_dir) == ".harness-tap/traces"


def test_inspect_prints_context_summary(tmp_path, capsys):
    store = JsonlTraceStore(tmp_path, session_id="session-1")
    store.append(
        {
            "timestamp": "2026-07-02T00:00:00+00:00",
            "turn": 1,
            "duration_ms": 42,
            "request": {
                "body": {
                    "model": "gpt-test",
                    "messages": [
                        {"role": "system", "content": "You are Harness."},
                        {"role": "user", "content": "Inspect this."},
                    ],
                    "tools": [
                        {"type": "function", "function": {"name": "read_file"}},
                    ],
                }
            },
            "response": {
                "body": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Summary ready.",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                }
            },
        }
    )

    exit_code = main(["inspect", "--trace-dir", str(tmp_path), "session-1"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Session: session-1" in output
    assert "Turn 1: model=gpt-test duration_ms=42" in output
    assert "system: You are Harness." in output
    assert "user: Inspect this." in output
    assert "tool: read_file" in output
    assert "assistant: Summary ready." in output
    assert "usage: prompt=10 completion=3 total=13" in output
