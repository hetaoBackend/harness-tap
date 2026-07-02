from __future__ import annotations

import json

from harness_tap.sse import ChatCompletionSSEReassembler


def _frame(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def test_reassembler_accumulates_streamed_content():
    reassembler = ChatCompletionSSEReassembler()

    reassembler.feed_bytes(
        b"".join(
            [
                _frame(
                    {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "model": "gpt-test",
                        "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                    }
                ),
                _frame(
                    {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "model": "gpt-test",
                        "choices": [{"index": 0, "delta": {"content": "Hello "}}],
                    }
                ),
                _frame(
                    {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "model": "gpt-test",
                        "choices": [{"index": 0, "delta": {"content": "Harness"}, "finish_reason": "stop"}],
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )
    )

    response = reassembler.final_response()

    assert response["id"] == "chatcmpl-stream"
    assert response["object"] == "chat.completion"
    assert response["model"] == "gpt-test"
    assert response["choices"][0]["message"] == {"role": "assistant", "content": "Hello Harness"}
    assert response["choices"][0]["finish_reason"] == "stop"
    assert len(reassembler.events) == 3
    assert reassembler.done is True


def test_reassembler_accumulates_streamed_tool_calls():
    reassembler = ChatCompletionSSEReassembler()

    reassembler.feed_bytes(
        b"".join(
            [
                _frame(
                    {
                        "id": "chatcmpl-tools",
                        "model": "gpt-test",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {"name": "lookup", "arguments": '{"q"'},
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ),
                _frame(
                    {
                        "id": "chatcmpl-tools",
                        "model": "gpt-test",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": ':"context"}'},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        )
    )

    message = reassembler.final_response()["choices"][0]["message"]

    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"context"}'},
        }
    ]
