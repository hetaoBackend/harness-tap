from __future__ import annotations

import json
from typing import Any

from harness_tap.protocols import (
    PROTOCOL_ANTHROPIC_MESSAGES,
    PROTOCOL_CHAT_COMPLETIONS,
    PROTOCOL_RESPONSES,
    protocol_from_path,
)


def protocol_from_record(record: dict[str, Any]) -> str:
    capture = record.get("capture")
    if isinstance(capture, dict):
        protocol = capture.get("protocol")
        if isinstance(protocol, str) and protocol:
            return protocol
    path = ""
    request = record.get("request")
    if isinstance(request, dict) and isinstance(request.get("path"), str):
        path = request["path"]
    return protocol_from_path(path) or PROTOCOL_CHAT_COMPLETIONS


def request_body(record: dict[str, Any]) -> dict[str, Any]:
    request = record.get("request")
    body = request.get("body") if isinstance(request, dict) else {}
    return body if isinstance(body, dict) else {}


def response_body(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response")
    body = response.get("body") if isinstance(response, dict) else {}
    return body if isinstance(body, dict) else {}


def record_status(record: dict[str, Any]) -> int:
    response = record.get("response")
    status = response.get("status") if isinstance(response, dict) else 0
    return status if isinstance(status, int) else 0


def record_model(record: dict[str, Any]) -> str:
    body = request_body(record)
    model = body.get("model")
    if isinstance(model, str) and model:
        return model
    response = response_body(record)
    model = response.get("model")
    return model if isinstance(model, str) else ""


def normalized_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    input_tokens = _int(usage.get("input_tokens"), usage.get("prompt_tokens"))
    output_tokens = _int(usage.get("output_tokens"), usage.get("completion_tokens"))
    total_tokens = _int(usage.get("total_tokens"), input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def record_total_tokens(record: dict[str, Any]) -> int:
    return normalized_usage(response_body(record))["total_tokens"]


def extract_tools(protocol: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    items: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        items.append(
            {
                "title": tool_name(tool),
                "text": tool_description(tool),
                "parameters": _tool_parameters(tool),
            }
        )
    return items


def tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    name = tool.get("name")
    if isinstance(name, str) and name:
        return name
    tool_type = tool.get("type")
    return tool_type if isinstance(tool_type, str) else "tool"


def tool_description(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("description"), str):
        return function["description"]
    description = tool.get("description")
    return description if isinstance(description, str) else ""


def extract_conversation(protocol: str, request: dict[str, Any], response: dict[str, Any]) -> list[dict[str, Any]]:
    if protocol == PROTOCOL_RESPONSES:
        items = _responses_request_items(request)
        response_items = _responses_output_items(response)
    elif protocol == PROTOCOL_ANTHROPIC_MESSAGES:
        items = _anthropic_request_items(request)
        response_items = _anthropic_output_items(response)
    else:
        items = _chat_request_items(request)
        response_items = _chat_output_items(response)
    return items + response_items


def grouped_sections(conversation: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "system": [],
        "user": [],
        "assistant": [],
        "tool_results": [],
        "response": [],
    }
    for item in conversation:
        kind = item.get("kind")
        if kind == "instructions":
            buckets["system"].append(item)
        elif kind in buckets:
            buckets[kind].append(item)
        elif kind in {"tool_call"}:
            buckets["assistant"].append(item)
    sections = [
        _section("system", "System prompts", buckets["system"]),
        _section("tool_schemas", "Tool schemas", tools),
        _section("user", "User prompts", buckets["user"]),
        _section("assistant", "Assistant messages", buckets["assistant"]),
        _section("tool_results", "Tool results", buckets["tool_results"]),
        _section("response", "Upstream response", buckets["response"]),
    ]
    return [section for section in sections if section["count"]]


def turn_context_summary(record: dict[str, Any], index: int) -> dict[str, Any]:
    protocol = protocol_from_record(record)
    req = request_body(record)
    res = response_body(record)
    conversation = extract_conversation(protocol, req, res)
    tools = extract_tools(protocol, req)
    request_messages = [item for item in conversation if item.get("kind") != "response"]
    usage = normalized_usage(res)
    return {
        "index": index,
        "turn": record.get("turn", index + 1),
        "protocol": protocol,
        "model": req.get("model") or record_model(record),
        "status": record_status(record),
        "duration_ms": record.get("duration_ms", 0),
        "message_count": len(request_messages),
        "tool_schema_count": len(tools),
        "usage": usage,
        "conversation": conversation,
        "tools": tools,
        "sections": grouped_sections(conversation, tools),
    }


def session_summary(session_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    timestamps = [record.get("timestamp") for record in records if isinstance(record.get("timestamp"), str)]
    protocols = sorted({protocol_from_record(record) for record in records})
    models = sorted({model for record in records if (model := record_model(record))})
    return {
        "id": session_id,
        "record_count": len(records),
        "started_at": timestamps[0] if timestamps else "",
        "updated_at": timestamps[-1] if timestamps else "",
        "models": models,
        "protocols": protocols,
        "total_duration_ms": sum(
            int(record.get("duration_ms") or 0)
            for record in records
            if isinstance(record.get("duration_ms"), int)
        ),
        "total_tokens": sum(record_total_tokens(record) for record in records),
        "error_count": sum(1 for record in records if record_status(record) >= 400),
    }


def inspect_turn_lines(record: dict[str, Any]) -> list[str]:
    summary = turn_context_summary(record, 0)
    usage = summary["usage"]
    lines = [
        f"Turn {record.get('turn')}: "
        f"protocol={summary['protocol']} "
        f"model={summary.get('model') or ''} "
        f"duration_ms={record.get('duration_ms', '')}"
    ]
    for item in summary["conversation"]:
        if item.get("kind") == "response":
            continue
        role = item.get("role") or item.get("kind") or "message"
        lines.append(f"{role}: {item.get('text') or ''}")
    for tool in summary["tools"]:
        name = tool.get("title")
        if name:
            lines.append(f"tool: {name}")
    for item in summary["conversation"]:
        if item.get("kind") == "response":
            lines.append(f"assistant: {item.get('text') or ''}")
    if usage["total_tokens"] or usage["input_tokens"] or usage["output_tokens"]:
        lines.append(
            "usage: "
            f"prompt={usage['input_tokens']} "
            f"completion={usage['output_tokens']} "
            f"total={usage['total_tokens']}"
        )
    return lines


def _section(kind: str, label: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": kind, "label": label, "count": len(items), "items": items}


def _chat_request_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    items: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") if isinstance(message.get("role"), str) else "message"
        kind = {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool_results"}.get(role, role)
        counts[kind] = counts.get(kind, 0) + 1
        text = content_text(message.get("content")) or tool_call_text(message.get("tool_calls"))
        items.append(_item(kind, role, f"{role} #{counts[kind]}", text))
    return items


def _chat_output_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    error = _error_text(body)
    if error:
        return [_item("response", "error", "error", error)]
    assistant = _chat_assistant_message(body)
    if not assistant:
        return []
    text = content_text(assistant.get("content")) or tool_call_text(assistant.get("tool_calls"))
    if not text:
        return []
    return [_item("response", "assistant", "assistant", text)]


def _chat_assistant_message(body: dict[str, Any]) -> dict[str, Any]:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    return message if isinstance(message, dict) else {}


def _responses_request_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    instruction_text = content_text(instructions)
    if instruction_text:
        items.append(_item("system", "system", "instructions", instruction_text))
    raw_input = body.get("input")
    if isinstance(raw_input, str) and raw_input:
        items.append(_item("user", "user", "user #1", raw_input))
        return items
    if not isinstance(raw_input, list):
        return items
    counts: dict[str, int] = {}
    for entry in raw_input:
        parsed = _responses_input_item(entry)
        if parsed is None:
            continue
        kind = parsed["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        parsed["title"] = parsed.get("title") or f"{parsed['role']} #{counts[kind]}"
        items.append(parsed)
    return items


def _responses_input_item(entry: Any) -> dict[str, Any] | None:
    if isinstance(entry, str) and entry:
        return _item("user", "user", "user", entry)
    if not isinstance(entry, dict):
        return None
    entry_type = entry.get("type")
    if entry_type in {"function_call", "custom_tool_call"}:
        name = entry.get("name") or "tool"
        arguments = entry.get("arguments") or entry.get("input") or ""
        return _item("assistant", "assistant", str(name), f"{name}({_stringify(arguments)})")
    if entry_type in {"function_call_output", "custom_tool_call_output"}:
        output = entry.get("output") or entry.get("result") or ""
        return _item("tool_results", "tool", str(entry.get("call_id") or "tool"), content_text(output))
    role = entry.get("role") if isinstance(entry.get("role"), str) else "user"
    kind = {"system": "system", "developer": "system", "user": "user", "assistant": "assistant"}.get(role, "user")
    text = content_text(entry.get("content")) or content_text(entry)
    if entry_type == "message" or role:
        return _item(kind, role, f"{role}", text)
    if text:
        return _item("user", "user", "input", text)
    return None


def _responses_output_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    error = _error_text(body)
    if error:
        return [_item("response", "error", "error", error)]
    parts: list[str] = []
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text:
        parts.append(output_text)
    output = body.get("output")
    if isinstance(output, list):
        for entry in output:
            if not isinstance(entry, dict):
                continue
            entry_type = entry.get("type")
            if entry_type == "message":
                text = content_text(entry.get("content"))
                if text:
                    parts.append(text)
            elif entry_type in {"function_call", "custom_tool_call"}:
                name = entry.get("name") or "tool"
                arguments = entry.get("arguments") or ""
                parts.append(f"{name}({_stringify(arguments)})")
            else:
                text = content_text(entry.get("content") or entry.get("text"))
                if text:
                    parts.append(text)
    text = "\n".join(part for part in parts if part)
    if not text:
        return []
    return [_item("response", "assistant", "assistant", text)]


def _anthropic_request_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    system = body.get("system")
    system_text = content_text(system)
    if system_text:
        items.append(_item("system", "system", "system", system_text))
    messages = body.get("messages")
    if not isinstance(messages, list):
        return items
    counts: dict[str, int] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role") if isinstance(message.get("role"), str) else "user"
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                parsed = _anthropic_block_item(role, block)
                if parsed is None:
                    continue
                kind = parsed["kind"]
                counts[kind] = counts.get(kind, 0) + 1
                parsed["title"] = parsed.get("title") or f"{parsed['role']} #{counts[kind]}"
                items.append(parsed)
            continue
        kind = {"user": "user", "assistant": "assistant"}.get(role, role)
        counts[kind] = counts.get(kind, 0) + 1
        items.append(_item(kind, role, f"{role} #{counts[kind]}", content_text(content)))
    return items


def _anthropic_block_item(role: str, block: Any) -> dict[str, Any] | None:
    if isinstance(block, str) and block:
        kind = {"user": "user", "assistant": "assistant"}.get(role, "user")
        return _item(kind, role, role, block)
    if not isinstance(block, dict):
        return None
    block_type = block.get("type")
    if block_type == "tool_use":
        name = block.get("name") or "tool"
        return _item("assistant", "assistant", str(name), f"{name}({_stringify(block.get('input'))})")
    if block_type == "tool_result":
        return _item(
            "tool_results",
            "tool",
            str(block.get("tool_use_id") or "tool"),
            content_text(block.get("content")),
        )
    text = content_text(block)
    if not text:
        return None
    kind = {"user": "user", "assistant": "assistant"}.get(role, "user")
    return _item(kind, role, role, text)


def _anthropic_output_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    error = _error_text(body)
    if error:
        return [_item("response", "error", "error", error)]
    content = body.get("content")
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                name = block.get("name") or "tool"
                parts.append(f"{name}({_stringify(block.get('input'))})")
            else:
                text = content_text(block)
                if text:
                    parts.append(text)
    else:
        text = content_text(content)
        if text:
            parts.append(text)
    combined = "\n".join(parts)
    if not combined:
        return []
    return [_item("response", "assistant", "assistant", combined)]


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("output_text") or item.get("thinking")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if item.get("type") == "tool_use":
                    name = item.get("name") or "tool"
                    parts.append(f"{name}({_stringify(item.get('input'))})")
                    continue
                if item.get("type") in {"tool_result", "function_call_output"}:
                    parts.append(content_text(item.get("content") or item.get("output")))
                    continue
                if isinstance(item.get("type"), str):
                    nested = content_text(item.get("content") or item.get("output"))
                    parts.append(nested or f"[{item['type']}]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    if isinstance(content, dict):
        for key in ("text", "output_text", "content", "output"):
            if key in content:
                nested = content_text(content[key])
                if nested:
                    return nested
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def tool_call_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    parts: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name") or tool_call.get("id") or "tool"
            arguments = function.get("arguments") or ""
            parts.append(f"{name}({arguments})")
            continue
        name = tool_call.get("name") or tool_call.get("id") or "tool"
        arguments = tool_call.get("arguments") or tool_call.get("input") or ""
        parts.append(f"{name}({_stringify(arguments)})")
    return "\n".join(parts)


def _tool_parameters(tool: dict[str, Any]) -> Any:
    function = tool.get("function")
    if isinstance(function, dict) and function.get("parameters") is not None:
        return function.get("parameters")
    if tool.get("parameters") is not None:
        return tool.get("parameters")
    return tool.get("input_schema")


def _item(kind: str, role: str, title: str, text: str) -> dict[str, Any]:
    return {"kind": kind, "role": role, "title": title, "text": text}


def _error_text(body: dict[str, Any]) -> str:
    error = body.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
        return json.dumps(error, ensure_ascii=False, sort_keys=True)
    return ""


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _int(*values: Any) -> int:
    for value in values:
        if isinstance(value, int):
            return value
    return 0
