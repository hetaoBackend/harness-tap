from __future__ import annotations

import copy
import json
from typing import Any

from harness_tap.protocols import (
    PROTOCOL_ANTHROPIC_MESSAGES,
    PROTOCOL_CHAT_COMPLETIONS,
    PROTOCOL_RESPONSES,
)


class SSEReader:
    def __init__(self) -> None:
        self._buffer = b""
        self._event = ""
        self._data_lines: list[str] = []

    def feed_bytes(self, chunk: bytes) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        self._buffer += chunk
        while b"\n" in self._buffer:
            line_bytes, self._buffer = self._buffer.split(b"\n", 1)
            completed = self._feed_line(line_bytes.decode("utf-8", errors="replace").rstrip("\r"))
            if completed is not None:
                events.append(completed)
        return events

    def _feed_line(self, line: str) -> tuple[str, str] | None:
        if line.startswith("event:"):
            self._event = line[len("event:") :].strip()
            return None
        if line.startswith("data:"):
            self._data_lines.append(line[len("data:") :].strip())
            return None
        if line == "":
            if not self._data_lines:
                self._event = ""
                return None
            raw_data = "\n".join(self._data_lines)
            event_name = self._event
            self._data_lines = []
            self._event = ""
            return event_name, raw_data
        return None


class ChatCompletionSSEReassembler:
    def __init__(self, *, store_events: bool = True) -> None:
        self.store_events = store_events
        self.events: list[dict[str, Any]] = []
        self.done = False
        self._reader = SSEReader()
        self._id = ""
        self._model = ""
        self._usage: dict[str, Any] | None = None
        self._choices: dict[int, dict[str, Any]] = {}

    def feed_bytes(self, chunk: bytes) -> None:
        for _, raw_data in self._reader.feed_bytes(chunk):
            if raw_data == "[DONE]":
                self.done = True
                continue
            payload = _parse_json_object(raw_data)
            if payload is None:
                continue
            if self.store_events:
                self.events.append({"data": copy.deepcopy(payload)})
            self._accumulate(payload)

    def final_response(self) -> dict[str, Any]:
        choices: list[dict[str, Any]] = []
        for index in sorted(self._choices):
            state = self._choices[index]
            message: dict[str, Any] = {"role": state.get("role") or "assistant"}
            content = state.get("content")
            if content:
                message["content"] = content
            elif not state.get("tool_calls"):
                message["content"] = ""
            if state.get("tool_calls"):
                message["tool_calls"] = [
                    {
                        "id": tool_call.get("id", ""),
                        "type": tool_call.get("type", "function"),
                        "function": {
                            "name": tool_call.get("function", {}).get("name", ""),
                            "arguments": tool_call.get("function", {}).get("arguments", ""),
                        },
                    }
                    for _, tool_call in sorted(state["tool_calls"].items())
                ]
            choices.append(
                {
                    "index": index,
                    "message": message,
                    "finish_reason": state.get("finish_reason"),
                }
            )

        response: dict[str, Any] = {
            "id": self._id,
            "object": "chat.completion",
            "model": self._model,
            "choices": choices,
        }
        if self._usage is not None:
            response["usage"] = copy.deepcopy(self._usage)
        return response

    def _accumulate(self, payload: dict[str, Any]) -> None:
        if isinstance(payload.get("id"), str):
            self._id = payload["id"]
        if isinstance(payload.get("model"), str):
            self._model = payload["model"]
        if isinstance(payload.get("usage"), dict):
            self._usage = copy.deepcopy(payload["usage"])

        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index")
            if not isinstance(index, int):
                index = 0
            state = self._choices.setdefault(index, {"role": "assistant", "content": "", "tool_calls": {}})
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                state["finish_reason"] = finish_reason
            delta = choice.get("delta")
            if isinstance(delta, dict):
                self._accumulate_delta(state, delta)

    def _accumulate_delta(self, state: dict[str, Any], delta: dict[str, Any]) -> None:
        role = delta.get("role")
        if isinstance(role, str) and role:
            state["role"] = role
        content = delta.get("content")
        if isinstance(content, str):
            state["content"] = state.get("content", "") + content
        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            for fallback_index, tool_delta in enumerate(tool_calls):
                if isinstance(tool_delta, dict):
                    self._accumulate_tool_call(state, fallback_index, tool_delta)

    def _accumulate_tool_call(self, state: dict[str, Any], fallback_index: int, tool_delta: dict[str, Any]) -> None:
        index = tool_delta.get("index")
        if not isinstance(index, int):
            index = fallback_index
        tool_calls = state.setdefault("tool_calls", {})
        tool_call = tool_calls.setdefault(index, {"function": {"arguments": ""}})
        if isinstance(tool_delta.get("id"), str):
            tool_call["id"] = tool_delta["id"]
        if isinstance(tool_delta.get("type"), str):
            tool_call["type"] = tool_delta["type"]
        function = tool_delta.get("function")
        if isinstance(function, dict):
            target_function = tool_call.setdefault("function", {"arguments": ""})
            name = function.get("name")
            if isinstance(name, str):
                target_function["name"] = name
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                target_function["arguments"] = target_function.get("arguments", "") + arguments


class ResponsesSSEReassembler:
    def __init__(self, *, store_events: bool = True) -> None:
        self.store_events = store_events
        self.events: list[dict[str, Any]] = []
        self.done = False
        self._reader = SSEReader()
        self._completed: dict[str, Any] | None = None
        self._id = ""
        self._model = ""
        self._status = "in_progress"
        self._usage: dict[str, Any] | None = None
        self._output: dict[int, dict[str, Any]] = {}

    def feed_bytes(self, chunk: bytes) -> None:
        for event_name, raw_data in self._reader.feed_bytes(chunk):
            if raw_data == "[DONE]":
                self.done = True
                continue
            payload = _parse_json_object(raw_data)
            if payload is None:
                continue
            event_type = payload.get("type") if isinstance(payload.get("type"), str) else event_name
            if self.store_events:
                self.events.append({"event": event_type, "data": copy.deepcopy(payload)})
            self._accumulate(event_type, payload)

    def final_response(self) -> dict[str, Any]:
        if self._completed is not None:
            return copy.deepcopy(self._completed)
        response: dict[str, Any] = {
            "id": self._id,
            "object": "response",
            "status": self._status,
            "model": self._model,
            "output": [self._output[index] for index in sorted(self._output)],
        }
        if self._usage is not None:
            response["usage"] = copy.deepcopy(self._usage)
        return response

    def _accumulate(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type in {"response.completed", "response.failed", "response.incomplete"}:
            self.done = True
            response = payload.get("response")
            if isinstance(response, dict):
                self._completed = copy.deepcopy(response)
                self._merge_meta(response)
            return
        if event_type in {"response.created", "response.in_progress"}:
            response = payload.get("response")
            if isinstance(response, dict):
                self._merge_meta(response)
                output = response.get("output")
                if isinstance(output, list):
                    for index, item in enumerate(output):
                        if isinstance(item, dict):
                            self._output[index] = copy.deepcopy(item)
            return
        if event_type == "response.output_item.added":
            item = payload.get("item")
            index = payload.get("output_index")
            if not isinstance(index, int):
                index = len(self._output)
            if isinstance(item, dict):
                self._output[index] = copy.deepcopy(item)
            return
        if event_type == "response.output_item.done":
            item = payload.get("item")
            index = payload.get("output_index")
            if isinstance(index, int) and isinstance(item, dict):
                self._output[index] = copy.deepcopy(item)
            return
        if event_type in {"response.output_text.delta", "response.text.delta"}:
            delta = payload.get("delta")
            if isinstance(delta, str):
                self._append_output_text(
                    payload.get("output_index"),
                    payload.get("content_index"),
                    delta,
                )
            return
        if event_type == "response.function_call_arguments.delta":
            delta = payload.get("delta")
            index = payload.get("output_index")
            if isinstance(index, int) and isinstance(delta, str):
                item = self._output.setdefault(index, {"type": "function_call", "arguments": ""})
                item["arguments"] = str(item.get("arguments") or "") + delta

    def _merge_meta(self, response: dict[str, Any]) -> None:
        if isinstance(response.get("id"), str):
            self._id = response["id"]
        if isinstance(response.get("model"), str):
            self._model = response["model"]
        if isinstance(response.get("status"), str):
            self._status = response["status"]
        if isinstance(response.get("usage"), dict):
            self._usage = copy.deepcopy(response["usage"])

    def _append_output_text(self, output_index: Any, content_index: Any, delta: str) -> None:
        if not isinstance(output_index, int):
            output_index = 0
        if not isinstance(content_index, int):
            content_index = 0
        item = self._output.setdefault(
            output_index,
            {"type": "message", "role": "assistant", "content": []},
        )
        content = item.setdefault("content", [])
        if not isinstance(content, list):
            content = []
            item["content"] = content
        while len(content) <= content_index:
            content.append({"type": "output_text", "text": ""})
        part = content[content_index]
        if not isinstance(part, dict):
            part = {"type": "output_text", "text": ""}
            content[content_index] = part
        part["text"] = str(part.get("text") or "") + delta


class AnthropicMessagesSSEReassembler:
    def __init__(self, *, store_events: bool = True) -> None:
        self.store_events = store_events
        self.events: list[dict[str, Any]] = []
        self.done = False
        self._reader = SSEReader()
        self._message: dict[str, Any] = {
            "id": "",
            "type": "message",
            "role": "assistant",
            "model": "",
            "content": [],
            "stop_reason": None,
            "usage": {},
        }

    def feed_bytes(self, chunk: bytes) -> None:
        for event_name, raw_data in self._reader.feed_bytes(chunk):
            if raw_data == "[DONE]":
                self.done = True
                continue
            payload = _parse_json_object(raw_data)
            if payload is None:
                continue
            event_type = payload.get("type") if isinstance(payload.get("type"), str) else event_name
            if self.store_events:
                self.events.append({"event": event_type, "data": copy.deepcopy(payload)})
            self._accumulate(event_type, payload)

    def final_response(self) -> dict[str, Any]:
        message = copy.deepcopy(self._message)
        content: list[dict[str, Any]] = []
        raw_content = message.get("content")
        if isinstance(raw_content, list):
            for block in raw_content:
                if not isinstance(block, dict):
                    continue
                cleaned = dict(block)
                pending = cleaned.pop("_json", None)
                if pending is not None:
                    cleaned["input"] = _parse_json_value(pending)
                content.append(cleaned)
        message["content"] = content
        if not message.get("usage"):
            message.pop("usage", None)
        return message

    def _accumulate(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == "message_start":
            message = payload.get("message")
            if isinstance(message, dict):
                self._message.update(copy.deepcopy(message))
                self._message.setdefault("content", [])
            return
        if event_type == "content_block_start":
            index = payload.get("index")
            block = payload.get("content_block")
            if isinstance(index, int) and isinstance(block, dict):
                self._set_block(index, copy.deepcopy(block))
            return
        if event_type == "content_block_delta":
            index = payload.get("index")
            delta = payload.get("delta")
            if isinstance(index, int) and isinstance(delta, dict):
                self._apply_block_delta(index, delta)
            return
        if event_type == "content_block_stop":
            index = payload.get("index")
            if isinstance(index, int):
                block = self._block_at(index)
                pending = block.pop("_json", None)
                if pending is not None:
                    block["input"] = _parse_json_value(pending)
            return
        if event_type == "message_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict):
                if isinstance(delta.get("stop_reason"), str):
                    self._message["stop_reason"] = delta["stop_reason"]
                if "stop_sequence" in delta:
                    self._message["stop_sequence"] = delta["stop_sequence"]
            usage = payload.get("usage")
            if isinstance(usage, dict):
                current = self._message.setdefault("usage", {})
                if isinstance(current, dict):
                    current.update(copy.deepcopy(usage))
            return
        if event_type == "message_stop":
            self.done = True

    def _set_block(self, index: int, block: dict[str, Any]) -> None:
        content = self._message.setdefault("content", [])
        if not isinstance(content, list):
            content = []
            self._message["content"] = content
        while len(content) <= index:
            content.append({"type": "text", "text": ""})
        content[index] = block

    def _block_at(self, index: int) -> dict[str, Any]:
        content = self._message.setdefault("content", [])
        if not isinstance(content, list):
            content = []
            self._message["content"] = content
        while len(content) <= index:
            content.append({"type": "text", "text": ""})
        block = content[index]
        if not isinstance(block, dict):
            block = {"type": "text", "text": ""}
            content[index] = block
        return block

    def _apply_block_delta(self, index: int, delta: dict[str, Any]) -> None:
        block = self._block_at(index)
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text")
            if isinstance(text, str):
                block["text"] = str(block.get("text") or "") + text
            return
        if delta_type == "input_json_delta":
            fragment = delta.get("partial_json")
            if isinstance(fragment, str):
                block["_json"] = str(block.get("_json") or "") + fragment
            return
        if delta_type == "thinking_delta":
            thinking = delta.get("thinking")
            if isinstance(thinking, str):
                block["thinking"] = str(block.get("thinking") or "") + thinking


def reassembler_for_protocol(protocol: str, *, store_events: bool = True) -> Any:
    if protocol == PROTOCOL_RESPONSES:
        return ResponsesSSEReassembler(store_events=store_events)
    if protocol == PROTOCOL_ANTHROPIC_MESSAGES:
        return AnthropicMessagesSSEReassembler(store_events=store_events)
    if protocol == PROTOCOL_CHAT_COMPLETIONS:
        return ChatCompletionSSEReassembler(store_events=store_events)
    return ChatCompletionSSEReassembler(store_events=store_events)


def _parse_json_object(raw_data: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_json_value(raw_data: str) -> Any:
    try:
        return json.loads(raw_data)
    except json.JSONDecodeError:
        return raw_data
