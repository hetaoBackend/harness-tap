"""Read-only presentation of native traces, shared by the viewer and CLI.

Items retain their original value and source path; this is not a wire converter.
"""
from __future__ import annotations

import json
from typing import Any

from harness_tap.protocols import CHAT, MESSAGES, RESPONSES, record_protocol


def as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(content_text(part) for part in value)
    if isinstance(value, dict):
        for key in ("text", "thinking"):
            if isinstance(value.get(key), str):
                return value[key]
        if "content" in value and value.get("type", "message") == "message":
            text = content_text(value["content"])
            if value.get("tool_calls"):
                text += "\n" + json.dumps(value["tool_calls"], ensure_ascii=False)
            return text
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _item(value: Any, path: str, role: str | None = None) -> dict:
    obj = as_dict(value)
    kind = obj.get("type") if isinstance(obj.get("type"), str) else "message"
    inferred = "tool" if kind in {"function_call_output", "tool_result"} else "assistant" if kind in {"function_call", "reasoning", "tool_use"} else "user"
    return {
        "role": role or (obj.get("role") if isinstance(obj.get("role"), str) else None) or inferred,
        "kind": kind, "source_path": path, "text": content_text(value), "value": value,
    }


def tool_name(value: Any) -> str:
    tool = as_dict(value)
    return as_dict(tool.get("function")).get("name") or tool.get("name") or tool.get("type") or "tool"


def usage_summary(usage: dict, protocol: str) -> dict:
    def number(*keys):
        for key in keys:
            if isinstance(usage.get(key), int):
                return usage[key]
        return None
    input_tokens = number("input_tokens", "prompt_tokens")
    output_tokens = number("output_tokens", "completion_tokens")
    total_tokens = number("total_tokens")
    # Anthropic's input_tokens excludes prompt-cache creation and reads. Keep
    # those counters visible and include them once in the derived total.
    cache = sum(number(key) or 0 for key in ("cache_creation_input_tokens", "cache_read_input_tokens")) if protocol == MESSAGES else 0
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens + cache
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": total_tokens, "raw": usage}


def project_record(record: dict) -> dict:
    protocol = record_protocol(record)
    req = as_dict(as_dict(record.get("request")).get("body"))
    response = as_dict(record.get("response"))
    body = as_dict(response.get("body"))
    capture = as_dict(record.get("capture"))
    inputs: list[dict] = []
    outputs: list[dict] = []
    instructions = "instructions" if protocol == RESPONSES else "system" if protocol == MESSAGES else None
    if instructions and req.get(instructions) is not None:
        inputs.append(_item(req[instructions], instructions, "system"))
    field = "input" if protocol == RESPONSES else "messages"
    value = req.get(field)
    if isinstance(value, list):
        inputs.extend(_item(item, f"{field}[{index}]") for index, item in enumerate(value))
    elif value is not None:
        inputs.append(_item(value, field))
    if protocol == RESPONSES:
        outputs = [_item(item, f"output[{index}]", "assistant") for index, item in enumerate(as_list(body.get("output")))]
    elif protocol == MESSAGES:
        outputs = [_item(item, f"content[{index}]", "assistant") for index, item in enumerate(as_list(body.get("content")))]
    else:
        for index, choice in enumerate(as_list(body.get("choices"))):
            if "message" in as_dict(choice):
                outputs.append(_item(choice["message"], f"choices[{index}].message", "assistant"))
    native_field = "output" if protocol == RESPONSES else "content" if protocol == MESSAGES else "choices"
    if not outputs and native_field not in body and response.get("body") is not None and not body.get("error"):
        outputs = [_item(response["body"], "body", "assistant")]
    status = capture.get("model_status") or body.get("status") or body.get("stop_reason")
    status = status if isinstance(status, str) else None
    choices = as_list(body.get("choices"))
    stop_reason = body.get("stop_reason") or (as_dict(choices[0]).get("finish_reason") if choices else None)
    error = body.get("error") or capture.get("stream_error")
    http_status = response.get("status", 0)
    is_error = bool(
        (isinstance(http_status, int) and http_status >= 400) or error
        or status in {"failed", "incomplete", "cancelled"}
        or capture.get("partial") or capture.get("parse_errors") or capture.get("request_parse_errors")
        or capture.get("request_truncated") or capture.get("transport_error")
    )
    context = {key: req[key] for key in ("previous_response_id", "conversation") if req.get(key) is not None} if protocol == RESPONSES else {}
    usage = usage_summary(as_dict(body.get("usage")), protocol)
    return {
        "protocol": protocol, "input_items": inputs, "output_items": outputs,
        "tools": as_list(req.get("tools")), "usage": usage,
        "context_references": context,
        "context_note": "References server-side history; the full context is not present in this request." if context else "",
        "outcome": {"status": status, "stop_reason": stop_reason, "incomplete_details": body.get("incomplete_details"),
                    "error": error, "is_error": is_error, "capture": capture},
    }


def context_sections(projection: dict) -> list[dict]:
    sections = []
    def add(kind: str, label: str, items: list[dict]) -> None:
        if items:
            sections.append({"kind": kind, "label": label, "count": len(items), "items": items})
    for role, label, kind in (("system", "System prompts", "system"), ("developer", "Developer prompts", "developer"),
                              ("user", "User prompts", "user"), ("assistant", "Assistant messages", "assistant"),
                              ("tool", "Tool results", "tool_results")):
        items = [item for item in projection["input_items"] if item["role"] == role]
        add(kind, label, [{"title": f"{role} #{i + 1}", "text": item["text"]} for i, item in enumerate(items)])
    add("tool_schemas", "Tool schemas", [{"title": tool_name(tool), "text": as_dict(as_dict(tool).get("function") or tool).get("description", "")} for tool in projection["tools"]])
    add("response", "Upstream response", [{"title": "assistant", "text": item["text"]} for item in projection["output_items"]])
    return sections
