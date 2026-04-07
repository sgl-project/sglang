# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Bootstrap server for dynamic encoder URL registration and discovery in EPD mode."""

import logging
import threading
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)


class EncoderBootstrapServer:
    """Lightweight HTTP server for dynamic encoder URL registration.

    Encoder-only servers register their URLs via HTTP POST on startup.
    Language-only (prefill) servers query via HTTP GET to discover live encoders
    instead of requiring static --encoder-urls at startup.

    Runs in a daemon thread. Lifecycle is managed by the caller (e.g., the
    prefill server process).

    Endpoints:
        GET  /health                - Health check; returns 200 OK.
        POST /register_encoder_url  - Register an encoder URL.
        DELETE /unregister_encoder_url - Remove an encoder URL.
        GET  /list_encoder_urls     - List all registered encoder URLs.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self._encoder_urls: List[str] = []
        self._lock = threading.Lock()

        app = FastAPI()

        @app.get("/health")
        def health():
            return PlainTextResponse("OK")

        @app.post("/register_encoder_url")
        def register_encoder_url(data: dict):
            try:
                url = data.get("url")
                if not url:
                    raise ValueError("Missing or empty 'url' field in request body")

                with self._lock:
                    if url not in self._encoder_urls:
                        self._encoder_urls.append(url)
                        logger.info(f"Registered encoder URL: {url}")
                    else:
                        logger.debug(f"Encoder URL already registered: {url}")

                return PlainTextResponse("OK")
            except Exception as e:
                logger.error(f"Failed to register encoder URL: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @app.delete("/unregister_encoder_url")
        def unregister_encoder_url(data: dict):
            try:
                url = data.get("url")
                if not url:
                    raise ValueError("Missing or empty 'url' field in request body")

                with self._lock:
                    if url in self._encoder_urls:
                        self._encoder_urls.remove(url)
                        logger.info(f"Unregistered encoder URL: {url}")

                return PlainTextResponse("OK")
            except Exception as e:
                logger.error(f"Failed to unregister encoder URL: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @app.get("/list_encoder_urls")
        def list_encoder_urls():
            with self._lock:
                urls = list(self._encoder_urls)
            return {"encoder_urls": urls}

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            daemon=True,
        )
        self._thread.start()
        logger.info(f"EncoderBootstrapServer started on {host}:{port}")

    def close(self):
        """Stop the bootstrap server."""
        self._server.should_exit = True
        self._thread.join(timeout=5)

    def get_encoder_urls(self) -> List[str]:
        """Direct in-process access to the registered encoder URLs."""
        with self._lock:
            return list(self._encoder_urls)
