"""FastAPI application for Bike4Mind OpenAI Shim."""
import time
import uuid
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from shim.config import get_api_key, get_base_url, get_port, get_host
from shim.translator import (
    prepare_upstream_payload,
    parse_sse_line,
    format_chunk,
    format_completion_response,
    format_openai_error,
    map_stop_reason,
    map_usage,
)

# Configure logging without request bodies or sensitive keys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bike4mind_shim")

def sanitize_message(msg: str, secret_key: Optional[str] = None) -> str:
    """Ensure no API keys leak into logs or client-facing responses."""
    if not msg:
        return ""
    if secret_key and secret_key in msg:
        msg = msg.replace(secret_key, "[REDACTED_API_KEY]")
    # Redact common b4m key patterns if any
    msg = re.sub(r"b4m_live_[a-zA-Z0-9_-]+", "[REDACTED_API_KEY]", msg)
    msg = re.sub(r"Bearer\s+[a-zA-Z0-9_\-\.]+", "Bearer [REDACTED]", msg)
    return msg

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context to validate configuration on server startup."""
    logger.info("Initializing Bike4Mind OpenAI Shim...")
    base_url = get_base_url()
    logger.info("Upstream Base URL: %s", base_url)
    try:
        # Validate that the API key is present
        _ = get_api_key(required=True)
        logger.info("B4M_API_KEY loaded successfully.")
    except Exception as e:
        logger.critical("Startup failed: %s", e)
        raise
    yield
    logger.info("Shutting down Bike4Mind OpenAI Shim.")

app = FastAPI(
    title="Bike4Mind OpenAI Shim",
    description="OpenAI-compatible proxy service translating to Bike4Mind completions API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for tools like Open WebUI, LiteLLM, OmniRoute
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log basic request details without inspecting bodies or headers."""
    start_time = time.time()
    method = request.method
    path = request.url.path
    response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info("%s %s -> status=%d elapsed=%sms", method, path, response.status_code, duration_ms)
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "bike4mind-openai-shim",
        "upstream_url": get_base_url(),
    }

@app.get("/v1/models")
async def list_models():
    """Proxy and translate models endpoint to OpenAI format."""
    try:
        api_key = get_api_key(required=True)
    except Exception as e:
        return JSONResponse(
            status_code=401,
            content=format_openai_error("B4M_API_KEY is not configured", code="invalid_api_key", status_code=401),
        )

    base_url = get_base_url()
    target_url = f"{base_url}/api/v1/models"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                target_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
            )
            if resp.status_code != 200:
                raw_err = sanitize_message(resp.text, api_key)
                logger.error("Upstream /api/v1/models returned HTTP %d: %s", resp.status_code, raw_err)
                return JSONResponse(
                    status_code=resp.status_code,
                    content=format_openai_error(
                        f"Upstream returned status {resp.status_code}: {raw_err[:200]}",
                        code="upstream_models_error",
                        status_code=resp.status_code,
                    ),
                )
            
            data = resp.json()
            models_list = data.get("data", [])
            now_ts = int(time.time())
            
            openai_models = []
            for item in models_list:
                model_id = item.get("id")
                if not model_id:
                    continue
                openai_models.append({
                    "id": model_id,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": item.get("backend", "bike4mind"),
                })

            return {
                "object": "list",
                "data": openai_models,
            }
    except Exception as exc:
        err_str = sanitize_message(str(exc), api_key)
        logger.exception("Error querying /v1/models: %s", err_str)
        return JSONResponse(
            status_code=502,
            content=format_openai_error(f"Failed to fetch models from upstream: {err_str}", code="upstream_connect_error", status_code=502),
        )

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle OpenAI chat completions, proxying to Bike4Mind."""
    try:
        api_key = get_api_key(required=True)
    except Exception:
        return JSONResponse(
            status_code=401,
            content=format_openai_error("B4M_API_KEY environment variable is missing", code="invalid_api_key", status_code=401),
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=format_openai_error("Invalid JSON body", code="invalid_request_error", status_code=400),
        )

    try:
        upstream_payload, client_wants_stream = prepare_upstream_payload(body)
    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content=format_openai_error(str(ve), code="invalid_request_error", status_code=400),
        )

    requested_model = upstream_payload["model"]
    base_url = get_base_url()
    completions_url = f"{base_url}/api/ai/v1/completions"

    # Streaming httpx client with no read timeout for long generations
    timeout_config = httpx.Timeout(connect=15.0, read=None, write=15.0, pool=15.0)

    if client_wants_stream:
        return await handle_streaming_request(completions_url, upstream_payload, api_key, requested_model, timeout_config)
    else:
        return await handle_non_streaming_request(completions_url, upstream_payload, api_key, requested_model, timeout_config)

async def handle_streaming_request(
    url: str,
    payload: Dict[str, Any],
    api_key: str,
    model: str,
    timeout_config: httpx.Timeout,
) -> Response:
    """Stream SSE chunks translated from Bike4Mind SSE."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    client = httpx.AsyncClient(timeout=timeout_config)
    
    try:
        req = client.build_request("POST", url, json=payload, headers=headers)
        resp = await client.send(req, stream=True)
    except Exception as exc:
        await client.aclose()
        err_msg = sanitize_message(str(exc), api_key)
        logger.error("Failed to connect to Bike4Mind upstream: %s", err_msg)
        return JSONResponse(
            status_code=502,
            content=format_openai_error(f"Failed to connect to upstream service: {err_msg}", code="upstream_connect_error", status_code=502),
        )

    if resp.status_code != 200:
        raw_error = await resp.aread()
        await resp.aclose()
        await client.aclose()
        err_text = sanitize_message(raw_error.decode("utf-8", errors="replace"), api_key)
        logger.error("Upstream completions returned HTTP %d: %s", resp.status_code, err_text)
        return JSONResponse(
            status_code=resp.status_code,
            content=format_openai_error(
                f"Upstream error ({resp.status_code}): {err_text[:200]}",
                code="upstream_error",
                status_code=resp.status_code,
            ),
        )

    async def event_generator() -> AsyncIterator[str]:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())
        finish_emitted = False

        try:
            async for raw_line in resp.aiter_lines():
                parsed = parse_sse_line(raw_line)
                if not parsed:
                    continue

                event_type = parsed.get("type")
                
                # Meta event may provide requestId
                if event_type == "meta":
                    req_id = parsed.get("requestId")
                    if req_id:
                        chunk_id = f"chatcmpl-{req_id}"
                    continue

                # Content event
                if event_type == "content":
                    text = parsed.get("text", "")
                    if text:
                        yield format_chunk(chunk_id, model, created_ts, {"content": text})

                    stop_reason = parsed.get("stopReason")
                    if stop_reason is not None:
                        finish_reason = map_stop_reason(stop_reason)
                        usage = map_usage(parsed.get("usage"))
                        yield format_chunk(
                            chunk_id,
                            model,
                            created_ts,
                            {},
                            finish_reason=finish_reason,
                            usage=usage,
                        )
                        finish_emitted = True

                elif event_type == "error":
                    err_msg = sanitize_message(parsed.get("message", "Upstream error event"), api_key)
                    err_code = parsed.get("code", "upstream_error")
                    logger.error("Received error event from upstream: code=%s message=%s", err_code, err_msg)
                    # Emit error chunk or terminate
                    if not finish_emitted:
                        yield format_chunk(chunk_id, model, created_ts, {}, finish_reason="stop")
                        finish_emitted = True
                    break

                elif event_type == "done":
                    break

            # If stream finished without an explicit stopReason chunk, emit clean finish
            if not finish_emitted:
                yield format_chunk(chunk_id, model, created_ts, {}, finish_reason="stop")

            # Always end streaming response with [DONE]
            yield "data: [DONE]\n\n"

        except Exception as stream_err:
            sanitized = sanitize_message(str(stream_err), api_key)
            logger.exception("Error while streaming response: %s", sanitized)
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

async def handle_non_streaming_request(
    url: str,
    payload: Dict[str, Any],
    api_key: str,
    model: str,
    timeout_config: httpx.Timeout,
) -> Response:
    """Consume upstream SSE stream fully and aggregate into a single chat.completion response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            req = client.build_request("POST", url, json=payload, headers=headers)
            resp = await client.send(req, stream=True)

            if resp.status_code != 200:
                raw_error = await resp.aread()
                await resp.aclose()
                err_text = sanitize_message(raw_error.decode("utf-8", errors="replace"), api_key)
                logger.error("Upstream completions returned HTTP %d: %s", resp.status_code, err_text)
                return JSONResponse(
                    status_code=resp.status_code,
                    content=format_openai_error(
                        f"Upstream error ({resp.status_code}): {err_text[:200]}",
                        code="upstream_error",
                        status_code=resp.status_code,
                    ),
                )

            full_content: list[str] = []
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created_ts = int(time.time())
            finish_reason = "stop"
            final_usage: Optional[Dict[str, Any]] = None

            async for raw_line in resp.aiter_lines():
                parsed = parse_sse_line(raw_line)
                if not parsed:
                    continue

                event_type = parsed.get("type")

                if event_type == "meta":
                    req_id = parsed.get("requestId")
                    if req_id:
                        completion_id = f"chatcmpl-{req_id}"

                elif event_type == "content":
                    text = parsed.get("text", "")
                    if text:
                        full_content.append(text)

                    if "stopReason" in parsed and parsed.get("stopReason") is not None:
                        finish_reason = map_stop_reason(parsed.get("stopReason"))

                    if "usage" in parsed and parsed.get("usage") is not None:
                        final_usage = map_usage(parsed.get("usage"))

                elif event_type == "error":
                    err_msg = sanitize_message(parsed.get("message", "Upstream error event"), api_key)
                    err_code = parsed.get("code", "upstream_error")
                    await resp.aclose()
                    logger.error("Received error event from upstream: code=%s message=%s", err_code, err_msg)
                    return JSONResponse(
                        status_code=400,
                        content=format_openai_error(err_msg, code=err_code, status_code=400),
                    )

                elif event_type == "done":
                    break

            await resp.aclose()

            concatenated_text = "".join(full_content)
            response_data = format_completion_response(
                completion_id=completion_id,
                model=model,
                created=created_ts,
                content=concatenated_text,
                finish_reason=finish_reason,
                usage=final_usage,
            )
            return JSONResponse(content=response_data)

    except Exception as exc:
        err_msg = sanitize_message(str(exc), api_key)
        logger.exception("Error handling non-streaming chat completion: %s", err_msg)
        return JSONResponse(
            status_code=502,
            content=format_openai_error(f"Failed to communicate with upstream: {err_msg}", code="upstream_connect_error", status_code=502),
        )
