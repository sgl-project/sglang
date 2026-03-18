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

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

# Port offset from server_args.port for the bootstrap server
ENGINE_INFO_BOOTSTRAP_PORT_OFFSET = 100


def get_engine_info_bootstrap_port(server_args) -> int:
    """Compute the bootstrap server port consistently across all nodes.

    In multi-node setups, dist_init_addr is the same on every node, so we
    derive the port from it.  In single-node setups we fall back to
    server_args.port.
    """
    if server_args.dist_init_addr:
        return (
            NetworkAddress.parse(server_args.dist_init_addr).port
            + ENGINE_INFO_BOOTSTRAP_PORT_OFFSET
        )
    return server_args.port + ENGINE_INFO_BOOTSTRAP_PORT_OFFSET


class EngineInfoBootstrapServer:
    """Lightweight HTTP server for per-rank model info registration.

    Runs in a daemon thread on node_rank==0. Each ModelRunner registers its
    info via HTTP PUT after model initialization. The Engine
    accesses the collected info directly in-process; external consumers can
    query via HTTP GET.

    Uses a route-table pattern: adding a new endpoint is just one
    ``self._routes[("METHOD", "/path")] = handler_fn`` line in
    ``_register_routes``.

    Currently supports transfer engine memory registration info.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        # Storage: {tp_rank: (session_id, weights_info_dict)}
        self.transfer_engine_info: Dict[int, Tuple] = {}
        self.lock = threading.Lock()

        # Route table: {("METHOD", "/path"): handler_fn}
        self._routes: Dict[Tuple[str, str], callable] = {}
        self._register_routes()

        self._httpd = HTTPServer((host, port), self._make_handler())
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def _register_routes(self):
        self._routes[("PUT", "/register_engine_info")] = self._put_engine_info
        self._routes[("GET", "/get_transfer_engine_info")] = self._get_engine_info
        self._routes[("GET", "/health")] = self._get_health

    # ------------------------------------------------------------------
    # Route handlers — each receives (handler, parsed_url)
    # ------------------------------------------------------------------

    def _get_health(self, h, _parsed):
        h._reply(200, "OK")

    def _put_engine_info(self, h, _parsed):
        """Handle PUT /register_engine_info from ModelRunner.

        Payload: {
            "tp_rank": int,
            "transfer_engine_info": {
                "session_id": str,
                "weights_info_dict": dict
            }
        }
        """
        try:
            content_length = int(h.headers.get("Content-Length", 0))
            body = h.rfile.read(content_length)
            data = json.loads(body)
            tp_rank = data["tp_rank"]
            info = data["transfer_engine_info"]
            session_id = info["session_id"]
            weights_info_dict = info["weights_info_dict"]

            with self.lock:
                self.transfer_engine_info[tp_rank] = (
                    session_id,
                    weights_info_dict,
                )

            logger.info(
                f"Registered transfer engine info for tp_rank={tp_rank}, "
                f"session_id={session_id}"
            )
            h._reply(200, "OK")
        except Exception as e:
            logger.error(f"Failed to register engine info: {e}")
            h._reply(400, str(e))

    def _get_engine_info(self, h, parsed):
        """Handle GET /get_transfer_engine_info?rank=N.

        Response:
        {
            "rank": N,
            "remote_instance_transfer_engine_info": [session_id, weights_info_dict]
        }
        """
        params = parse_qs(parsed.query)
        rank_values = params.get("rank")
        if not rank_values:
            h._reply(400, "Missing rank parameter")
            return

        try:
            rank = int(rank_values[0])
        except ValueError:
            h._reply(400, "Invalid rank parameter")
            return

        if rank < 0:
            h._reply(400, "Invalid rank parameter")
            return

        with self.lock:
            info = self.transfer_engine_info.get(rank)

        if info is None:
            h._reply(404, f"No transfer engine info for rank {rank}")
            return

        h._reply(
            200,
            {"rank": rank, "remote_instance_transfer_engine_info": list(info)},
        )

    # ------------------------------------------------------------------
    # Handler factory
    # ------------------------------------------------------------------

    def _make_handler(self):
        routes = self._routes

        class Handler(BaseHTTPRequestHandler):
            def _dispatch(h):
                parsed = urlparse(h.path)
                fn = routes.get((h.command, parsed.path))
                if fn:
                    fn(h, parsed)
                else:
                    h._reply(404, "Not Found")

            do_GET = do_PUT = _dispatch

            def _reply(h, status, body):
                if isinstance(body, dict):
                    payload = json.dumps(body).encode()
                    content_type = "application/json"
                else:
                    payload = str(body).encode()
                    content_type = "text/plain"
                h.send_response(status)
                h.send_header("Content-Type", content_type)
                h.send_header("Content-Length", str(len(payload)))
                h.end_headers()
                h.wfile.write(payload)

            def log_message(h, fmt, *args):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(fmt, *args)

        return Handler

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _run_server(self):
        """Run the HTTP server in a dedicated thread."""
        try:
            logger.info(f"EngineInfoBootstrapServer started on {self.host}:{self.port}")
            self._httpd.serve_forever()
        except Exception as e:
            logger.error(f"EngineInfoBootstrapServer error: {e}")

    def get_transfer_engine_info(self, rank: int) -> Optional[Tuple]:
        """Direct in-process access for co-located HTTP server (no HTTP round-trip)."""
        return self.transfer_engine_info.get(rank)
