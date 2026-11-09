"""Comprehensive tests for the Bike4Mind OpenAI Shim service."""
import json
import pytest
import respx
import httpx
from shim.config import get_api_key
from shim.main import sanitize_message
from shim.translator import (
    map_stop_reason,
    map_usage,
    prepare_upstream_payload,
    parse_sse_line,
    format_openai_error,
)

SAMPLE_UPSTREAM_SSE = (
    'data: {"type":"meta","requestId":"78c15daa-f852-40a4-8816-32515f946ed3"}\n\n'
    'data: {"type":"content","text":"Hi"}\n\n'
    'data: {"type":"content","text":" there! How"}\n\n'
    'data: {"type":"content","text":" can I help you today?"}\n\n'
    'data: {"type":"content","text":"","usage":{"inputTokens":80,"outputTokens":14,"reasoningTokens":0},"stopReason":"end_turn"}\n\n'
    'data: [DONE]\n\n'
)

# -------------------------------------------------------------------------
# Unit Tests for Translation Logic & Sanitization
# -------------------------------------------------------------------------

def test_map_stop_reason():
    assert map_stop_reason("end_turn") == "stop"
    assert map_stop_reason("max_tokens") == "length"
    assert map_stop_reason("pause_turn") == "stop"
    assert map_stop_reason("tool_use") == "tool_calls"
    assert map_stop_reason("unknown_weird_reason") == "stop"
    assert map_stop_reason(None) == "stop"

def test_map_usage_complete():
    b4m_usage = {
        "inputTokens": 120,
        "outputTokens": 45,
        "reasoningTokens": 15,
        "cacheReadInputTokens": 30,
    }
    mapped = map_usage(b4m_usage)
    assert mapped is not None
    assert mapped["prompt_tokens"] == 120
    assert mapped["completion_tokens"] == 45
    assert mapped["total_tokens"] == 165
    assert mapped["completion_tokens_details"]["reasoning_tokens"] == 15
    assert mapped["prompt_tokens_details"]["cached_tokens"] == 30

def test_map_usage_empty():
    assert map_usage(None) is None
    assert map_usage({}) == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

def test_prepare_upstream_payload():
    openai_req = {
        "model": "anthropic/claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 1000,
        "unsupported_field_xyz": 123,  # Should be ignored safely
    }
    payload, stream_flag = prepare_upstream_payload(openai_req)
    assert stream_flag is True
    assert payload["model"] == "anthropic/claude-3-5-sonnet"
    assert payload["stream"] is True
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 1000
    assert "unsupported_field_xyz" not in payload

def test_prepare_upstream_payload_validation():
    with pytest.raises(ValueError, match="Missing required 'model'"):
        prepare_upstream_payload({"messages": []})
    with pytest.raises(ValueError, match="Missing or invalid 'messages'"):
        prepare_upstream_payload({"model": "test-model"})

def test_parse_sse_line():
    assert parse_sse_line(": ping") is None
    assert parse_sse_line("") is None
    assert parse_sse_line("data: [DONE]") == {"type": "done"}
    parsed = parse_sse_line('data: {"type": "content", "text": "hello"}')
    assert parsed == {"type": "content", "text": "hello"}
    # Malformed json returns None without crashing
    assert parse_sse_line('data: {not valid json}') is None

def test_sanitize_message():
    secret = "b4m_live_abcdef1234567890"
    raw_error = f"Error communicating with key {secret} and Bearer {secret}"
    sanitized = sanitize_message(raw_error, secret)
    assert secret not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized

# -------------------------------------------------------------------------
# Integration Tests with Respx Mocking
# -------------------------------------------------------------------------

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "bike4mind-openai-shim"

@respx.mock
def test_streaming_translation_and_done_terminator(client):
    """Test streaming chat completions, chunk format, requestId mapping, usage, and [DONE] terminator."""
    respx.post("https://app.bike4mind.com/api/ai/v1/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=SAMPLE_UPSTREAM_SSE,
        )
    )

    request_payload = {
        "model": "anthropic/claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }

    response = client.post("/v1/chat/completions", json=request_payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = [line.strip() for line in response.text.strip().split("\n\n") if line.strip()]
    # Verify the stream ends with data: [DONE]
    assert lines[-1] == "data: [DONE]"

    chunks = []
    for line in lines[:-1]:
        assert line.startswith("data: ")
        chunk_json = json.loads(line[6:])
        chunks.append(chunk_json)

    # First chunk should have id with requestId and content 'Hi'
    assert chunks[0]["id"] == "chatcmpl-78c15daa-f852-40a4-8816-32515f946ed3"
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hi"

    # Concatenate text from deltas
    accumulated = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if "content" in c["choices"][0]["delta"]
    )
    assert accumulated == "Hi there! How can I help you today?"

    # The final chunk before [DONE] should carry finish_reason and usage
    final_chunk = chunks[-1]
    assert final_chunk["choices"][0]["finish_reason"] == "stop"
    assert "usage" in final_chunk
    assert final_chunk["usage"]["prompt_tokens"] == 80
    assert final_chunk["usage"]["completion_tokens"] == 14
    assert final_chunk["usage"]["total_tokens"] == 94

@respx.mock
def test_non_streaming_aggregation(client):
    """Test non-streaming request consumes upstream SSE and aggregates full content & usage."""
    respx.post("https://app.bike4mind.com/api/ai/v1/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=SAMPLE_UPSTREAM_SSE,
        )
    )

    request_payload = {
        "model": "anthropic/claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }

    response = client.post("/v1/chat/completions", json=request_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "chat.completion"
    assert data["id"] == "chatcmpl-78c15daa-f852-40a4-8816-32515f946ed3"
    assert data["model"] == "anthropic/claude-3-5-sonnet"
    assert len(data["choices"]) == 1

    choice = data["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hi there! How can I help you today?"

    assert data["usage"]["prompt_tokens"] == 80
    assert data["usage"]["completion_tokens"] == 14
    assert data["usage"]["total_tokens"] == 94
    assert data["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0

@respx.mock
def test_upstream_http_error_handling(client):
    """Test non-200 HTTP response from upstream is mapped to OpenAI error JSON with matching status."""
    respx.post("https://app.bike4mind.com/api/ai/v1/completions").mock(
        return_value=httpx.Response(
            429,
            json={"error": "Rate limit exceeded for organization"},
        )
    )

    request_payload = {
        "model": "anthropic/claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
    }

    response = client.post("/v1/chat/completions", json=request_payload)
    assert response.status_code == 429
    data = response.json()
    assert "error" in data
    assert "Rate limit exceeded" in data["error"]["message"]
    # Check that dummy test key is not in message
    assert "b4m_live_test_dummy_key" not in data["error"]["message"]

@respx.mock
def test_upstream_sse_error_event(client):
    """Test upstream returning an SSE error event."""
    sse_error_body = (
        'data: {"type":"error","code":"insufficient_credits","message":"Your balance is too low."}\n\n'
    )
    respx.post("https://app.bike4mind.com/api/ai/v1/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=sse_error_body,
        )
    )

    request_payload = {
        "model": "anthropic/claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }

    response = client.post("/v1/chat/completions", json=request_payload)
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "insufficient_credits"
    assert "Your balance is too low" in data["error"]["message"]

def test_missing_api_key(client, monkeypatch):
    """Test behavior when B4M_API_KEY is not set."""
    monkeypatch.delenv("B4M_API_KEY", raising=False)
    
    # Fast-fail configuration check raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        get_api_key(required=True)
    assert "B4M_API_KEY environment variable is missing" in str(exc_info.value)

    # API endpoint returns 401
    response = client.post(
        "/v1/chat/completions",
        json={"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 401
    data = response.json()
    assert "error" in data
    assert "B4M_API_KEY" in data["error"]["message"]

@respx.mock
def test_list_models_endpoint(client):
    """Test GET /v1/models returns OpenAI format list of models from bike4mind."""
    upstream_models_payload = {
        "data": [
            {
                "id": "anthropic/claude-3-5-sonnet",
                "name": "Claude 3.5 Sonnet",
                "type": "text",
                "backend": "anthropic",
            },
            {
                "id": "openai/gpt-4o",
                "name": "GPT-4o",
                "type": "text",
                "backend": "openai",
            },
        ]
    }

    respx.get("https://app.bike4mind.com/api/v1/models").mock(
        return_value=httpx.Response(200, json=upstream_models_payload)
    )

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["id"] == "anthropic/claude-3-5-sonnet"
    assert data["data"][0]["object"] == "model"
    assert data["data"][0]["owned_by"] == "anthropic"
    assert data["data"][1]["id"] == "openai/gpt-4o"
    assert data["data"][1]["owned_by"] == "openai"

def test_invalid_request_payloads(client):
    """Test validation errors for invalid payload."""
    res1 = client.post("/v1/chat/completions", content="invalid json text")
    assert res1.status_code == 400

    res2 = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert res2.status_code == 400
    assert "model" in res2.json()["error"]["message"]

    res3 = client.post("/v1/chat/completions", json={"model": "test", "messages": "not a list"})
    assert res3.status_code == 400
    assert "messages" in res3.json()["error"]["message"]
