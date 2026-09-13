"""Endpoint contracts. Upstreams are API directories, including any /v1 prefix."""
from __future__ import annotations

from dataclasses import dataclass

CHAT = "openai-chat-completions"
RESPONSES = "openai-responses"
MESSAGES = "anthropic-messages"


@dataclass(frozen=True)
class RouteSpec:
    protocol: str
    suffix: str


ROUTES = {
    "/v1/chat/completions": RouteSpec(CHAT, "/chat/completions"),
    "/v1/responses": RouteSpec(RESPONSES, "/responses"),
    "/v1/messages": RouteSpec(MESSAGES, "/messages"),
}


def record_protocol(record: dict) -> str:
    capture = record.get("capture")
    protocol = capture.get("protocol") if isinstance(capture, dict) else None
    if isinstance(protocol, str) and protocol:
        return protocol
    request = record.get("request")
    path = request.get("path") if isinstance(request, dict) else None
    route = ROUTES.get(path) if isinstance(path, str) else None
    return route.protocol if route else CHAT
