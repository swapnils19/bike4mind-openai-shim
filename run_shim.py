#!/usr/bin/env python3
"""Entrypoint script to run the Bike4Mind OpenAI Shim service."""
import sys
import logging
import uvicorn
from shim.config import get_api_key, get_host, get_port, get_base_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bike4mind_shim")

def main():
    try:
        # Fail-fast check on startup if B4M_API_KEY is missing
        get_api_key(required=True)
    except Exception as exc:
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        sys.exit(1)

    host = get_host()
    port = get_port()
    base_url = get_base_url()

    logger.info("==================================================")
    logger.info(" Starting Bike4Mind OpenAI Shim Service")
    logger.info(" Listening on http://%s:%d", host, port)
    logger.info(" Upstream Bike4Mind Base URL: %s", base_url)
    logger.info(" Endpoints:")
    logger.info("   - GET  /health")
    logger.info("   - GET  /v1/models")
    logger.info("   - POST /v1/chat/completions")
    logger.info("==================================================")

    uvicorn.run("shim.main:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
