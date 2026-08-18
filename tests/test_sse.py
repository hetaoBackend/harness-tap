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


def _named_frame(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def test_responses_reassembler_uses_completed_response():
    from harness_tap.sse import ResponsesSSEReassembler

    completed = {
        "id": "resp_1",
        "object": "response",
        "model": "gpt-test",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello stream"}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    reassembler = ResponsesSSEReassembler()
    reassembler.feed_bytes(
        b"".join(
            [
                _named_frame("response.created", {"type": "response.created", "response": {"id": "resp_1", "model": "gpt-test"}}),
                _named_frame(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hello "},
                ),
                _named_frame("response.completed", {"type": "response.completed", "response": completed}),
            ]
        )
    )

    assert reassembler.done is True
    assert reassembler.final_response() == completed


def test_responses_reassembler_accumulates_text_without_completed_event():
    from harness_tap.sse import ResponsesSSEReassembler

    reassembler = ResponsesSSEReassembler()
    reassembler.feed_bytes(
        b"".join(
            [
                _named_frame(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {"type": "message", "role": "assistant", "content": []},
                    },
                ),
                _named_frame(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hello "},
                ),
                _named_frame(
                    "response.output_text.delta",
                    {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Harness"},
                ),
            ]
        )
    )

    output = reassembler.final_response()["output"][0]
    assert output["content"][0]["text"] == "Hello Harness"


def test_anthropic_reassembler_accumulates_text_and_tool_use():
    from harness_tap.sse import AnthropicMessagesSSEReassembler

    reassembler = AnthropicMessagesSSEReassembler()
    reassembler.feed_bytes(
        b"".join(
            [
                _named_frame(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "model": "claude-test",
                            "content": [],
                            "usage": {"input_tokens": 8, "output_tokens": 1},
                        },
                    },
                ),
                _named_frame(
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                ),
                _named_frame(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}},
                ),
                _named_frame(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Claude"}},
                ),
                _named_frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
                _named_frame(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
                    },
                ),
                _named_frame(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
                    },
                ),
                _named_frame(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "input_json_delta", "partial_json": '"context"}'},
                    },
                ),
                _named_frame("content_block_stop", {"type": "content_block_stop", "index": 1}),
                _named_frame(
                    "message_delta",
                    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 6}},
                ),
                _named_frame("message_stop", {"type": "message_stop"}),
            ]
        )
    )

    response = reassembler.final_response()
    assert reassembler.done is True
    assert response["content"][0] == {"type": "text", "text": "Hello Claude"}
    assert response["content"][1]["name"] == "lookup"
    assert response["content"][1]["input"] == {"q": "context"}
    assert response["stop_reason"] == "tool_use"
    assert response["usage"]["output_tokens"] == 6
