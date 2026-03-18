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

import asyncio
import logging
import threading
from typing import Dict, Optional, Tuple

from aiohttp import web

logger = logging.getLogger(__name__)

# Port offset from server_args.port for the bootstrap server
ENGINE_INFO_BOOTSTRAP_PORT_OFFSET = 100


class EngineInfoBootstrapServer:
    """Lightweight aiohttp server for per-rank model info registration.

    Runs in a daemon thread on node_rank==0. Each ModelRunner registers its
    info via HTTP PUT after model initialization. The Engine
    accesses the collected info directly in-process; external consumers can
    query via HTTP GET.

    Currently supports transfer engine memory registration info.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.app = web.Application()
        self._setup_routes()

        # Storage: {tp_rank: (session_id, weights_info_dict)}
        self.transfer_engine_info: Dict[int, Tuple] = {}
        self.lock = asyncio.Lock()

        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()

    def _setup_routes(self):
        self.app.router.add_put(
            "/register_engine_info", self._handle_register_engine_info
        )
        self.app.router.add_get(
            "/get_transfer_engine_info", self._handle_get_transfer_engine_info
        )
        self.app.router.add_get("/health", self._handle_health_check)

    async def _handle_health_check(self, request: web.Request):
        return web.Response(text="OK", status=200)

    async def _handle_register_engine_info(self, request: web.Request):
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
            data = await request.json()
            tp_rank = data["tp_rank"]
            info = data["transfer_engine_info"]
            session_id = info["session_id"]
            weights_info_dict = info["weights_info_dict"]

            async with self.lock:
                self.transfer_engine_info[tp_rank] = (session_id, weights_info_dict)

            logger.info(
                f"Registered transfer engine info for tp_rank={tp_rank}, "
                f"session_id={session_id}"
            )
            return web.Response(text="OK", status=200)
        except Exception as e:
            logger.error(f"Failed to register engine info: {e}")
            return web.Response(text=str(e), status=400)

    async def _handle_get_transfer_engine_info(self, request: web.Request):
        """Handle GET /get_transfer_engine_info?rank=N.

        Response:
        {
            "rank": N,
            "remote_instance_transfer_engine_info": [session_id, weights_info_dict]
        }
        """
        rank_str = request.query.get("rank")
        if rank_str is None:
            return web.Response(text="Missing rank parameter", status=400)

        try:
            rank = int(rank_str)
        except ValueError:
            return web.Response(text="Invalid rank parameter", status=400)

        if rank < 0:
            return web.Response(text="Invalid rank parameter", status=400)

        async with self.lock:
            info = self.transfer_engine_info.get(rank)

        if info is None:
            return web.Response(
                text=f"No transfer engine info for rank {rank}", status=404
            )

        result = {
            "rank": rank,
            "remote_instance_transfer_engine_info": list(info),
        }
        return web.json_response(result, status=200)

    def _run_server(self):
        """Run the aiohttp server in a dedicated thread."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            access_log = None
            if logging.getLogger(__name__).getEffectiveLevel() <= logging.DEBUG:
                access_log = self.app.logger

            self._runner = web.AppRunner(self.app, access_log=access_log)
            self._loop.run_until_complete(self._runner.setup())

            site = web.TCPSite(self._runner, host=self.host, port=self.port)
            self._loop.run_until_complete(site.start())

            logger.info(f"EngineInfoBootstrapServer started on {self.host}:{self.port}")

            self._loop.run_forever()
        except Exception as e:
            logger.error(f"EngineInfoBootstrapServer error: {e}")
        finally:
            if hasattr(self, "_loop") and self._loop is not None:
                self._loop.run_until_complete(self._runner.cleanup())
                self._loop.close()

    def get_transfer_engine_info(self, rank: int) -> Optional[Tuple]:
        """Direct in-process access for co-located HTTP server (no HTTP round-trip)."""
        return self.transfer_engine_info.get(rank)
