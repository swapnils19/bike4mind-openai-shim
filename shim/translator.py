"""Translation logic between OpenAI format and Bike4Mind API format."""
import json
import time
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("bike4mind_shim")

# Recognized stopReason mapping to OpenAI finish_reason
STOP_REASON_MAP: Dict[str, str] = {
    "end_turn": "stop",
    "stop": "stop",
    "max_tokens": "length",
    "length": "length",
    "pause_turn": "stop",
    "tool_use": "tool_calls",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
}

def map_stop_reason(stop_reason: Optional[str]) -> str:
    """Map Bike4Mind stopReason to OpenAI finish_reason."""
    if not stop_reason:
        return "stop"
    
    normalized = stop_reason.strip().lower()
    if normalized in STOP_REASON_MAP:
        return STOP_REASON_MAP[normalized]
    
    logger.warning("Unknown stopReason '%s' received from upstream, defaulting to 'stop'", stop_reason)
    return "stop"

def map_usage(b4m_usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Map Bike4Mind usage object to OpenAI usage format."""
    if not isinstance(b4m_usage, dict):
        return None
    
    input_tokens = int(b4m_usage.get("inputTokens") or 0)
    output_tokens = int(b4m_usage.get("outputTokens") or 0)
    reasoning_tokens = int(b4m_usage.get("reasoningTokens") or 0)
    total_tokens = input_tokens + output_tokens

    usage: Dict[str, Any] = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    if "reasoningTokens" in b4m_usage:
        usage["completion_tokens_details"] = {
            "reasoning_tokens": reasoning_tokens
        }

    cache_read = b4m_usage.get("cacheReadInputTokens")
    if cache_read is not None:
        usage["prompt_tokens_details"] = {
            "cached_tokens": int(cache_read)
        }

    return usage

def prepare_upstream_payload(client_request: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Transform incoming OpenAI request payload to Bike4Mind payload.
    Returns (upstream_payload, client_wants_stream).
    Unsupported OpenAI fields are safely ignored.
    """
    model = client_request.get("model")
    messages = client_request.get("messages")
    if not model:
        raise ValueError("Missing required 'model' field in request body")
    if not isinstance(messages, list):
        raise ValueError("Missing or invalid 'messages' array in request body")

    client_wants_stream = bool(client_request.get("stream", False))

    # Upstream bike4mind always responds via Server-Sent Events stream
    payload: Dict[str, Any] = {
        "model": str(model),
        "messages": messages,
        "stream": True,
    }

    # Pass through supported fields documented by Bike4Mind completions API
    if "temperature" in client_request and client_request["temperature"] is not None:
        try:
            payload["temperature"] = float(client_request["temperature"])
        except (ValueError, TypeError):
            pass

    if "max_tokens" in client_request and client_request["max_tokens"] is not None:
        try:
            payload["max_tokens"] = int(client_request["max_tokens"])
        except (ValueError, TypeError):
            pass

    if "response_format" in client_request and isinstance(client_request["response_format"], dict):
        payload["response_format"] = client_request["response_format"]

    if "tools" in client_request and isinstance(client_request["tools"], list):
        payload["tools"] = client_request["tools"]

    # Support optional reasoning effort / thinking if specified
    if "reasoning_effort" in client_request:
        payload["options"] = payload.get("options", {})
        payload["options"]["reasoningEffort"] = client_request["reasoning_effort"]

    return payload, client_wants_stream

def parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single SSE text line. Returns dict if valid data JSON or DONE signal."""
    line = line.strip()
    if not line:
        return None
    if line.startswith(":"):
        # SSE comment / keep-alive
        return None
    if line.startswith("data:"):
        payload_str = line[5:].strip()
        if payload_str == "[DONE]":
            return {"type": "done"}
        try:
            return json.loads(payload_str)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode SSE JSON chunk: %s", exc)
            return None
    return None

def format_chunk(
    chunk_id: str,
    model: str,
    created: int,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> str:
    """Format OpenAI SSE chunk string ending with two newlines."""
    choice: Dict[str, Any] = {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
    }
    chunk_data: Dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    if usage is not None:
        chunk_data["usage"] = usage

    return f"data: {json.dumps(chunk_data)}\n\n"

def format_completion_response(
    completion_id: str,
    model: str,
    created: int,
    content: str,
    finish_reason: str,
    usage: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Format non-streaming OpenAI chat.completion response object."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

def format_openai_error(message: str, code: str = "upstream_error", status_code: int = 500) -> Dict[str, Any]:
    """Format an OpenAI-compliant error payload."""
    err_type = "invalid_request_error" if status_code < 500 else "api_error"
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": None,
            "code": code,
        }
    }
