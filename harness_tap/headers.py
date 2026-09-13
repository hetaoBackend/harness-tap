from __future__ import annotations

from collections.abc import Mapping

from multidict import CIMultiDict


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


def filter_headers(headers: Mapping[str, str], *, redact: bool = False) -> dict[str, str] | CIMultiDict[str]:
    connection_headers = {
        name.strip().lower()
        for key, value in headers.items() if key.lower() == "connection"
        for name in value.split(",")
    }
    filtered: dict[str, str] | CIMultiDict[str] = {} if redact else CIMultiDict()
    for key, value in headers.items():
        normalized = key.lower()
        if redact and normalized in SENSITIVE_HEADERS:
            filtered[key] = "***"
            continue
        if normalized in HOP_BY_HOP_HEADERS or normalized in connection_headers:
            continue
        if isinstance(filtered, CIMultiDict):
            filtered.add(key, value)
        else:
            filtered[key] = value
    return filtered
