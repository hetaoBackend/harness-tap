from __future__ import annotations

import copy
import json
from typing import Any


class ChatCompletionSSEReassembler:
    def __init__(self, *, store_events: bool = True) -> None:
        self.store_events = store_events
        self.events: list[dict[str, Any]] = []
        self.done = False
        self._buffer = b""
        self._data_lines: list[str] = []
        self._id = ""
        self._model = ""
        self._usage: dict[str, Any] | None = None
        self._choices: dict[int, dict[str, Any]] = {}

    def feed_bytes(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line_bytes, self._buffer = self._buffer.split(b"\n", 1)
            self._feed_line(line_bytes.decode("utf-8", errors="replace").rstrip("\r"))

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

    def _feed_line(self, line: str) -> None:
        if line.startswith("data:"):
            self._data_lines.append(line[len("data:") :].strip())
            return
        if line == "":
            self._emit_data()

    def _emit_data(self) -> None:
        if not self._data_lines:
            return
        raw_data = "\n".join(self._data_lines)
        self._data_lines = []
        if raw_data == "[DONE]":
            self.done = True
            return
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        if self.store_events:
            self.events.append({"data": copy.deepcopy(payload)})
        self._accumulate(payload)

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
