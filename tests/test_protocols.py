from harness_tap.protocols import (
    PROTOCOL_ANTHROPIC_MESSAGES,
    PROTOCOL_CHAT_COMPLETIONS,
    PROTOCOL_RESPONSES,
    join_upstream_url,
    protocol_from_path,
)


def test_protocol_from_known_paths():
    assert protocol_from_path("/v1/chat/completions") == PROTOCOL_CHAT_COMPLETIONS
    assert protocol_from_path("/v1/responses") == PROTOCOL_RESPONSES
    assert protocol_from_path("/v1/messages") == PROTOCOL_ANTHROPIC_MESSAGES
    assert protocol_from_path("/messages") == PROTOCOL_ANTHROPIC_MESSAGES
    assert protocol_from_path("/v1/models") is None


def test_join_upstream_url_avoids_duplicate_v1():
    assert (
        join_upstream_url("https://api.openai.com/v1", "/v1/chat/completions")
        == "https://api.openai.com/v1/chat/completions"
    )
    assert join_upstream_url("https://api.openai.com/v1", "/v1/responses") == "https://api.openai.com/v1/responses"
    assert join_upstream_url("https://api.anthropic.com/v1", "/v1/messages") == "https://api.anthropic.com/v1/messages"


def test_join_upstream_url_preserves_anthropic_root():
    assert join_upstream_url("https://api.anthropic.com", "/v1/messages") == "https://api.anthropic.com/v1/messages"
    assert join_upstream_url("https://api.anthropic.com", "/messages") == "https://api.anthropic.com/v1/messages"
