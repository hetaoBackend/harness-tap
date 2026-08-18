from harness_tap.context import inspect_turn_lines, turn_context_summary


def test_chat_completion_turn_keeps_grouped_sections():
    record = {
        "turn": 4,
        "duration_ms": 456,
        "request": {
            "path": "/v1/chat/completions",
            "body": {
                "model": "gpt-test",
                "messages": [
                    {"role": "system", "content": "You are Harness."},
                    {"role": "user", "content": "Build a hello world."},
                    {"role": "assistant", "content": "I will inspect the repo."},
                    {"role": "tool", "content": "README.md\npyproject.toml"},
                ],
                "tools": [
                    {"type": "function", "function": {"name": "Bash", "description": "Run a shell command."}},
                    {"type": "function", "function": {"name": "Edit", "description": "Edit a file."}},
                ],
            },
        },
        "response": {
            "status": 200,
            "body": {"choices": [{"message": {"role": "assistant", "content": "Created hello world."}}]},
        },
        "capture": {"protocol": "openai-chat-completions"},
    }

    summary = turn_context_summary(record, 0)
    sections = {section["kind"]: section for section in summary["sections"]}

    assert summary["protocol"] == "openai-chat-completions"
    assert [item["kind"] for item in summary["conversation"]] == [
        "system",
        "user",
        "assistant",
        "tool_results",
        "response",
    ]
    assert sections["system"]["items"][0]["text"] == "You are Harness."
    assert [item["title"] for item in sections["tool_schemas"]["items"]] == ["Bash", "Edit"]
    assert sections["response"]["items"][0]["text"] == "Created hello world."


def test_responses_turn_extracts_instructions_input_and_output():
    record = {
        "turn": 1,
        "request": {
            "path": "/v1/responses",
            "body": {
                "model": "gpt-test",
                "instructions": "Be terse.",
                "input": [
                    {"role": "user", "content": "Look this up."},
                    {"type": "function_call", "name": "lookup", "arguments": '{"q":"context"}', "call_id": "c1"},
                    {"type": "function_call_output", "call_id": "c1", "output": "found it"},
                ],
                "tools": [{"type": "function", "name": "lookup", "description": "Search context."}],
            },
        },
        "response": {
            "status": 200,
            "body": {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Here it is."}],
                    }
                ],
                "usage": {"input_tokens": 11, "output_tokens": 4},
            },
        },
        "capture": {"protocol": "openai-responses"},
    }

    summary = turn_context_summary(record, 0)

    assert summary["protocol"] == "openai-responses"
    assert summary["conversation"][0]["text"] == "Be terse."
    assert "Look this up." in [item["text"] for item in summary["conversation"]]
    assert "lookup({\"q\":\"context\"})" in [item["text"] for item in summary["conversation"]]
    assert summary["conversation"][-1]["kind"] == "response"
    assert summary["conversation"][-1]["text"] == "Here it is."
    assert summary["tools"][0]["title"] == "lookup"
    assert summary["usage"]["total_tokens"] == 15


def test_anthropic_turn_extracts_system_tool_use_and_tool_result():
    record = {
        "turn": 2,
        "request": {
            "path": "/v1/messages",
            "body": {
                "model": "claude-test",
                "system": "You are Harness.",
                "messages": [
                    {"role": "user", "content": "Inspect this."},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "I will look it up."},
                            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "context"}},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "found it"}],
                    },
                ],
                "tools": [{"name": "lookup", "description": "Search context.", "input_schema": {"type": "object"}}],
            },
        },
        "response": {
            "status": 200,
            "body": {
                "content": [{"type": "text", "text": "Done."}],
                "usage": {"input_tokens": 20, "output_tokens": 2},
            },
        },
        "capture": {"protocol": "anthropic-messages"},
    }

    summary = turn_context_summary(record, 0)

    assert summary["protocol"] == "anthropic-messages"
    assert summary["conversation"][0]["text"] == "You are Harness."
    assert "I will look it up." in [item["text"] for item in summary["conversation"]]
    assert "lookup({\"q\": \"context\"})" in [item["text"] for item in summary["conversation"]]
    assert summary["conversation"][-2]["kind"] == "tool_results"
    assert summary["conversation"][-1]["text"] == "Done."
    assert summary["tools"][0]["title"] == "lookup"


def test_inspect_lines_include_protocol():
    record = {
        "turn": 1,
        "duration_ms": 42,
        "request": {
            "body": {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "Inspect this."}],
            }
        },
        "response": {
            "body": {
                "choices": [{"message": {"role": "assistant", "content": "Summary ready."}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            }
        },
    }

    lines = inspect_turn_lines(record)

    assert lines[0] == "Turn 1: protocol=openai-chat-completions model=gpt-test duration_ms=42"
    assert "user: Inspect this." in lines
    assert "assistant: Summary ready." in lines
    assert "usage: prompt=10 completion=3 total=13" in lines
