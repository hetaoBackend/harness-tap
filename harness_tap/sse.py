from __future__ import annotations

import codecs
import copy
import json
import re
from collections.abc import Callable
from typing import Any

from harness_tap.protocols import CHAT, MESSAGES, RESPONSES


class SSEDecoder:
    """Incremental SSE framing independent of any provider's JSON schema."""

    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self.emit = emit
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")()
        self._buffer = ""
        self._frame: dict[str, Any] = {}
        self._data: list[str] = []

    def feed_bytes(self, chunk: bytes) -> None:
        self._buffer += self._decoder.decode(chunk)
        self._drain()

    def _drain(self, *, eof: bool = False) -> None:
        while match := re.search(r"\r\n|\r|\n", self._buffer):
            if match.group() == "\r" and match.end() == len(self._buffer) and not eof:
                break
            line, self._buffer = self._buffer[:match.start()], self._buffer[match.end():]
            self._line(line)

    def finish(self) -> bool:
        self._buffer += self._decoder.decode(b"", final=True)
        self._drain(eof=True)
        # SSE dispatch requires a blank line. Keep an unterminated frame as raw
        # evidence, but never let an unterminated terminal event mark success.
        incomplete = bool(self._buffer or self._frame or self._data)
        if incomplete:
            if self._buffer:
                self._line(self._buffer)
            self._frame["unterminated"] = True
            self._emit()
        return incomplete

    def _line(self, line: str) -> None:
        if not line:
            self._emit()
            return
        if line.startswith(":"):
            self._frame.setdefault("comments", []).append(line[1:])
            return
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field in {"event", "id", "retry"}:
            self._frame[field] = value
        else:
            self._frame.setdefault("fields", []).append([field, value])

    def _emit(self) -> None:
        if self._data:
            self._frame["raw_data"] = "\n".join(self._data)
        frame, self._frame, self._data = self._frame, {}, []
        if frame:
            self.emit(frame)


class SSEReassembler:
    def __init__(self, *, store_events: bool = True) -> None:
        self.store_events = store_events
        self.events: list[dict[str, Any]] = []
        self.done = False
        self.terminal_event: str | None = None
        self.model_status: str | None = None
        self.errors: list[str] = []
        self.stream_error: Any = None
        self._decoder = SSEDecoder(self._event)

    def feed_bytes(self, chunk: bytes) -> None:
        self._decoder.feed_bytes(chunk)

    def finish(self) -> None:
        if self._decoder.finish():
            self.errors.append("Unterminated SSE frame at EOF")

    def _event(self, frame: dict[str, Any]) -> None:
        raw = frame.get("raw_data")
        if raw == "[DONE]" and isinstance(self, ChatCompletionSSEReassembler) and not frame.get("unterminated"):
            self.done = True
            self.terminal_event = "[DONE]"
            self.model_status = "failed" if self.stream_error else "completed"
            return
        if raw is not None:
            try:
                frame["data"] = json.loads(raw)
            except json.JSONDecodeError:
                self.errors.append("Invalid JSON in SSE data")
        if self.store_events:
            self.events.append(copy.deepcopy(frame))
        payload = frame.get("data")
        if not isinstance(payload, dict) or frame.get("unterminated"):
            return
        event_type = payload.get("type") or frame.get("event", "")
        if event_type == "error" or payload.get("error"):
            self.stream_error = copy.deepcopy(payload.get("error") or payload)
            self.model_status = "failed"
        try:
            self._accumulate(payload, event_type)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            # Observability must not corrupt the forwarded stream.
            self.errors.append(f"Cannot reconstruct {event_type or 'event'}: {type(exc).__name__}")

    def capture_state(self) -> dict[str, Any]:
        result: dict[str, Any] = {"terminal_event": self.terminal_event, "model_status": self.model_status}
        if self.errors:
            result["parse_errors"] = self.errors
        if self.stream_error is not None:
            result["stream_error"] = self.stream_error
        return result

    def _accumulate(self, payload: dict[str, Any], event_type: str = "") -> None:
        raise NotImplementedError

    def final_response(self) -> dict[str, Any]:
        raise NotImplementedError


class ChatCompletionSSEReassembler(SSEReassembler):
    def __init__(self, *, store_events: bool = True) -> None:
        super().__init__(store_events=store_events)
        self._id = ""
        self._model = ""
        self._usage: dict[str, Any] | None = None
        self._choices: dict[int, dict[str, Any]] = {}

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

    def _accumulate(self, payload: dict[str, Any], event_type: str = "") -> None:
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


class ResponsesSSEReassembler(SSEReassembler):
    def __init__(self, *, store_events: bool = True) -> None:
        super().__init__(store_events=store_events)
        self._response: dict[str, Any] = {"object": "response", "output": []}
        self._items: dict[int, dict[str, Any]] = {}
        self._started = False

    def _accumulate(self, payload: dict[str, Any], event_type: str = "") -> None:
        response = payload.get("response")
        if isinstance(response, dict):
            if not isinstance(response.get("output", []), list):
                raise ValueError("Response output must be a list")
            self._response = copy.deepcopy(response)
            for index, item in enumerate(response.get("output") or []):
                self._items[index] = copy.deepcopy(item)
            if event_type == "response.created":
                self._started = True
            if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                self.done = True
                self.terminal_event = event_type
                self.model_status = event_type.removeprefix("response.")
                if not self._started:
                    self.errors.append("Missing response.created")
            return
        if not event_type.startswith(("response.output_item.", "response.content_part.",
                                      "response.output_text.", "response.refusal.",
                                      "response.reasoning_summary_", "response.function_call_arguments.")):
            return
        index = payload.get("output_index")
        if not isinstance(index, int) or index < 0:
            return
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            if isinstance(payload.get("item"), dict):
                self._items[index] = copy.deepcopy(payload["item"])
            return
        item = self._items.setdefault(index, {"id": payload.get("item_id", "")})
        if event_type in {"response.function_call_arguments.delta", "response.function_call_arguments.done"}:
            item.setdefault("type", "function_call")
            if event_type.endswith(".delta"):
                item["arguments"] = item.get("arguments", "") + payload["delta"]
            else:
                item["arguments"] = payload["arguments"]
            return
        part_index = payload.get("content_index", payload.get("summary_index"))
        if not isinstance(part_index, int) or not 0 <= part_index <= 10000:
            return
        key = "summary" if ".reasoning_summary_" in event_type else "content"
        parts = item.setdefault(key, [])
        while len(parts) <= part_index:
            parts.append({})
        if event_type.endswith("part.added") or event_type.endswith("part.done"):
            parts[part_index] = copy.deepcopy(payload["part"])
        elif event_type in {
            "response.output_text.delta", "response.output_text.done",
            "response.refusal.delta", "response.refusal.done",
            "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
        }:
            field = "refusal" if ".refusal." in event_type else "text"
            part = parts[part_index]
            part.setdefault("type", "refusal" if field == "refusal" else "summary_text" if key == "summary" else "output_text")
            if event_type.endswith(".delta"):
                part[field] = part.get(field, "") + payload["delta"]
            else:
                part[field] = payload[field]

    def final_response(self) -> dict[str, Any]:
        response = copy.deepcopy(self._response)
        if not self.done:
            response["output"] = [copy.deepcopy(item) for _, item in sorted(self._items.items())]
        return response


class MessagesSSEReassembler(SSEReassembler):
    def __init__(self, *, store_events: bool = True) -> None:
        super().__init__(store_events=store_events)
        self._message: dict[str, Any] = {"type": "message", "role": "assistant", "content": []}
        self._blocks: dict[int, dict[str, Any]] = {}
        self._inputs: dict[int, str] = {}
        self._open_blocks: set[int] = set()
        self._started = False

    def _accumulate(self, payload: dict[str, Any], event_type: str = "") -> None:
        if event_type.startswith("content_block_"):
            index = payload.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ValueError("Content block index must be a nonnegative integer")
        if event_type == "message_start":
            if not isinstance(payload.get("message"), dict) or not isinstance(payload["message"].get("content", []), list):
                raise ValueError("Invalid message_start")
            self._message = copy.deepcopy(payload["message"])
            self._started = True
            self._blocks = {i: copy.deepcopy(block) for i, block in enumerate(self._message.get("content", []))}
        elif event_type == "content_block_start":
            index = payload["index"]
            if not isinstance(payload.get("content_block"), dict):
                raise ValueError("Content block must be an object")
            self._blocks[index] = copy.deepcopy(payload["content_block"])
            self._open_blocks.add(index)
        elif event_type == "content_block_delta":
            index, delta = payload["index"], payload["delta"]
            block = self._blocks[index]
            kind = delta.get("type")
            if kind == "input_json_delta":
                self._inputs[index] = self._inputs.get(index, "") + delta["partial_json"]
            elif kind in {"text_delta", "thinking_delta", "signature_delta"}:
                field = kind.removesuffix("_delta")
                block[field] = block.get(field, "") + delta[field]
            elif kind == "citations_delta":
                block.setdefault("citations", []).append(copy.deepcopy(delta["citation"]))
        elif event_type == "content_block_stop":
            index = payload["index"]
            self._open_blocks.discard(index)
            if index in self._inputs:
                raw = self._inputs[index]
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("Tool input must be an object")
                self._blocks[index]["input"] = value
                del self._inputs[index]
        elif event_type == "message_delta":
            self._message.update(copy.deepcopy(payload.get("delta", {})))
            self._message.setdefault("usage", {}).update(copy.deepcopy(payload.get("usage", {})))
        elif event_type == "message_stop":
            self.done = True
            self.terminal_event = "message_stop"
            self.model_status = "failed" if self.stream_error else "completed"
            if not self._started:
                self.errors.append("Missing message_start")
            if self._open_blocks:
                self.errors.append("Unclosed content blocks at message_stop")

    def final_response(self) -> dict[str, Any]:
        message = copy.deepcopy(self._message)
        message["content"] = [copy.deepcopy(block) for _, block in sorted(self._blocks.items())]
        return message

    def capture_state(self) -> dict[str, Any]:
        result = super().capture_state()
        if self._inputs:
            result["partial_tool_inputs"] = {str(index): raw for index, raw in self._inputs.items()}
        return result


def create_reassembler(protocol: str) -> SSEReassembler:
    return {CHAT: ChatCompletionSSEReassembler, RESPONSES: ResponsesSSEReassembler, MESSAGES: MessagesSSEReassembler}[protocol]()
