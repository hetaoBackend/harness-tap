from __future__ import annotations

PROTOCOL_CHAT_COMPLETIONS = "openai-chat-completions"
PROTOCOL_RESPONSES = "openai-responses"
PROTOCOL_ANTHROPIC_MESSAGES = "anthropic-messages"

PROTOCOL_LABELS = {
    PROTOCOL_CHAT_COMPLETIONS: "Chat Completions",
    PROTOCOL_RESPONSES: "Responses",
    PROTOCOL_ANTHROPIC_MESSAGES: "Anthropic Messages",
}

_PATH_PROTOCOLS = {
    "/v1/chat/completions": PROTOCOL_CHAT_COMPLETIONS,
    "/chat/completions": PROTOCOL_CHAT_COMPLETIONS,
    "/v1/responses": PROTOCOL_RESPONSES,
    "/responses": PROTOCOL_RESPONSES,
    "/v1/messages": PROTOCOL_ANTHROPIC_MESSAGES,
    "/messages": PROTOCOL_ANTHROPIC_MESSAGES,
}


def protocol_from_path(path: str) -> str | None:
    return _PATH_PROTOCOLS.get(path)


def protocol_label(protocol: str) -> str:
    return PROTOCOL_LABELS.get(protocol, protocol or "Unknown")


def join_upstream_url(upstream_base_url: str, request_path: str) -> str:
    base = upstream_base_url.rstrip("/")
    if request_path.startswith("/v1/") and base.endswith("/v1"):
        return f"{base}{request_path[3:]}"
    if not request_path.startswith("/v1") and not base.endswith("/v1") and request_path.startswith("/"):
        return f"{base}/v1{request_path}"
    return f"{base}{request_path}"
