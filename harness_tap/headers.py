from __future__ import annotations

from collections.abc import Mapping


HOP_BY_HOP_HEADERS = {
    "connection",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
}

SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "set-cookie",
    "set-cookie2",
    "x-api-key",
    "proxy-authorization",
}


def filter_headers(headers: Mapping[str, str], *, redact: bool = False) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key, value in headers.items():
        normalized = key.lower()
        if redact and normalized in SENSITIVE_HEADERS:
            filtered[key] = "***"
            continue
        if normalized in HOP_BY_HOP_HEADERS:
            continue
        filtered[key] = value
    return filtered
