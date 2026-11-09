# Bike4Mind OpenAI-Compatible Shim Service

A lightweight, high-performance OpenAI-compatible proxy shim for the **Bike4Mind Completions API**.

This shim allows tools and interfaces that communicate in OpenAI API format—such as **LiteLLM**, **Open WebUI**, **OmniRoute**, and custom agent frameworks—to seamlessly connect to Bike4Mind.

---

## Features

- **OpenAI Compatible Endpoints**:
  - `POST /v1/chat/completions`: Supports both Server-Sent Events (`stream: true`) and full aggregation (`stream: false`).
  - `GET /v1/models`: Returns available models in OpenAI `model` list format.
  - `GET /health`: Fast service health check.
- **Accurate SSE & Chunk Translation**:
  - Extracts `requestId` from upstream `meta` events and maps it to `chatcmpl-{requestId}` chunk IDs.
  - Streams `delta.content` chunks in real-time.
  - Emits final chunk with `finish_reason` and token usage.
  - Concludes every streaming response with `data: [DONE]`.
- **Comprehensive Usage & Token Details**:
  - `inputTokens` ➔ `prompt_tokens`
  - `outputTokens` ➔ `completion_tokens`
  - `reasoningTokens` ➔ `completion_tokens_details.reasoning_tokens`
  - `cacheReadInputTokens` ➔ `prompt_tokens_details.cached_tokens`
  - `total_tokens` ➔ sum of prompt and completion tokens.
- **Stop Reason Mapping**:
  - `end_turn` ➔ `stop`
  - `max_tokens` ➔ `length`
  - `pause_turn` ➔ `stop`
  - `tool_use` / `tool_calls` ➔ `tool_calls`
  - `content_filter` ➔ `content_filter`
  - Unknown values log a warning and safely default to `stop`.
- **Security & Privacy**:
  - No secret leaks: API keys are redacted from all logs and client-facing error messages.
  - Request logging includes timestamp, HTTP method, path, status, and duration, without dumping request bodies or authentication headers.
  - Upstream errors (HTTP 4xx/5xx or SSE `error` events) are mapped to standard OpenAI error JSON payloads.
  - Fails fast on startup if `B4M_API_KEY` is missing.

---

## Project Structure

```text
.
├── pyproject.toml           # Poetry project configuration and dependencies
├── poetry.lock              # Poetry dependency lock file
├── pytest.ini               # Pytest configuration
├── run_shim.py              # Application runner script with startup validation
├── README.md                # Documentation and usage guide
├── shim/
│   ├── __init__.py          # Shim package definition
│   ├── config.py            # Environment configuration & validation
│   ├── main.py              # FastAPI endpoints, CORS, logging, and error handling
│   └── translator.py        # Request translation, SSE parser, and usage mapper
└── tests/
    ├── __init__.py
    ├── conftest.py          # Pytest fixtures and mock environment
    └── test_shim.py         # 15 unit and integration tests (mocked with respx)
```

---

## Requirements & Prerequisites

- Python 3.10+ (Python 3.11+ recommended)
- [Poetry](https://python-poetry.org/) dependency manager
- A Bike4Mind API key (starts with `b4m_live_...`), generated from your Bike4Mind account profile settings.

---

## Installation

Install dependencies using Poetry (this automatically manages the virtual environment):

```bash
poetry install
```

---

## Configuration

The shim is configured via environment variables:

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `B4M_API_KEY` | **Yes** | _None_ | Your Bike4Mind API key (`b4m_live_...`). Fails fast if missing. |
| `B4M_BASE_URL` | No | `https://app.bike4mind.com` | Upstream Bike4Mind base URL. |
| `PORT` | No | `8000` | Local port to bind. |
| `HOST` | No | `127.0.0.1` | Local IP to bind (defaults to localhost). |

---

## Running the Service

### Option 1: Using the runner script with Poetry
```bash
export B4M_API_KEY="b4m_live_your_actual_key_here"
poetry run python run_shim.py
```

### Option 2: Using Poetry Uvicorn directly
```bash
export B4M_API_KEY="b4m_live_your_actual_key_here"
poetry run uvicorn shim.main:app --host 127.0.0.1 --port 8000
```

---

## Running Tests

Execute the automated test suite with pytest via Poetry:

```bash
poetry run pytest -v
```

The test suite covers:
- Streaming translation with `chat.completion.chunk` formatting
- The `[DONE]` stream terminator
- Non-streaming aggregation into single `chat.completion` objects
- Complete token usage and reasoning token mapping
- Upstream HTTP errors (401, 429, 500) mapped to OpenAI error JSON
- SSE error event parsing (`type: "error"`)
- Fast-fail missing API key enforcement
- `/v1/models` translation
- `/health` check

---

## Curl Examples

### 1. Health Check
```bash
curl -X GET http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "bike4mind-openai-shim",
  "upstream_url": "https://app.bike4mind.com"
}
```

### 2. List Models
```bash
curl -X GET http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer any-key"
```

### 3. Non-Streaming Chat Completion (`stream: false`)
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key" \
  -d '{
    "model": "anthropic/claude-3-5-sonnet",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Explain shims in one sentence."}
    ],
    "temperature": 0.7,
    "stream": false
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-78c15daa-f852-40a4-8816-32515f946ed3",
  "object": "chat.completion",
  "created": 1731099600,
  "model": "anthropic/claude-3-5-sonnet",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A shim is a software library that transparently intercepts API calls and translates arguments or operations to provide compatibility between different systems."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 26,
    "total_tokens": 54,
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  }
}
```

### 4. Streaming Chat Completion (`stream: true`)
```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer any-key" \
  -d '{
    "model": "anthropic/claude-3-5-sonnet",
    "messages": [
      {"role": "user", "content": "Count from 1 to 3."}
    ],
    "stream": true
  }'
```

**Response:**
```text
data: {"id": "chatcmpl-78c15daa-...", "object": "chat.completion.chunk", "created": 1731099600, "model": "anthropic/claude-3-5-sonnet", "choices": [{"index": 0, "delta": {"content": "1,"}, "finish_reason": null}]}

data: {"id": "chatcmpl-78c15daa-...", "object": "chat.completion.chunk", "created": 1731099600, "model": "anthropic/claude-3-5-sonnet", "choices": [{"index": 0, "delta": {"content": " 2,"}, "finish_reason": null}]}

data: {"id": "chatcmpl-78c15daa-...", "object": "chat.completion.chunk", "created": 1731099600, "model": "anthropic/claude-3-5-sonnet", "choices": [{"index": 0, "delta": {"content": " 3."}, "finish_reason": null}]}

data: {"id": "chatcmpl-78c15daa-...", "object": "chat.completion.chunk", "created": 1731099600, "model": "anthropic/claude-3-5-sonnet", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23, "completion_tokens_details": {"reasoning_tokens": 0}}}

data: [DONE]
```

---

## Tool Integration Guides

### 1. LiteLLM
In your `config.yaml`:
```yaml
model_list:
  - model_name: bike4mind/claude-3-5-sonnet
    litellm_params:
      model: openai/anthropic/claude-3-5-sonnet
      api_base: http://127.0.0.1:8000/v1
      api_key: dummy
```

### 2. Open WebUI
1. Go to **Settings** ➔ **Connections** ➔ **OpenAI API**.
2. Set API URL to: `http://127.0.0.1:8000/v1`
3. Set API Key to any dummy string (the shim uses `B4M_API_KEY` from its environment).
4. Save and select your Bike4Mind models from the model dropdown.

### 3. OmniRoute
Set your provider base URL:
```json
{
  "provider": "openai",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key": "not-needed"
}
```

---

## Bike4Mind API Reference & Investigation Findings

Inspecting `https://app.bike4mind.com/api/v1/openapi.json` confirmed the following specifications:

1. **Completions Endpoint**: `POST /api/ai/v1/completions`
   - Content-Type: `text/event-stream` for both streaming and programmatic calls.
   - Request fields:
     - `model`: Model ID string (e.g. `anthropic/claude-3-5-sonnet`, `claude-opus-4-8`, `openai/gpt-4o`).
     - `messages`: List of objects with `role` (`user`, `assistant`, `system`) and `content`.
     - `stream`: Boolean.
     - `temperature`: Float (0.0 to 2.0).
     - `max_tokens`: Integer output ceiling.
     - `response_format`: Supports text and json_schema definitions.
     - `tools`: Function/tool definitions.
     - `options`: Nested dictionary containing `reasoningEffort` (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`), `thinking`, and `cacheTtl`.
2. **SSE Events**:
   - `meta`: `{"type":"meta", "requestId":"..."}`
   - `content`: `{"type":"content", "text":"...", "usage":{...}, "stopReason":"..."}`
   - `error`: `{"type":"error", "message":"...", "code":"insufficient_credits"|"spend_cap_exceeded"|"invalid_request"|"provider_error"|"provider_rejected"|"content_blocked"|"output_budget_exhausted"}`
3. **Models Endpoint**: `GET /api/v1/models`
   - Returns `{ "data": [ { "id": "...", "name": "...", "type": "text", "backend": "..." } ] }`.
