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

import logging
import threading
from typing import Dict, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)


class EngineInfoBootstrapServer:
    """Lightweight HTTP server for per-rank model info registration.

    Runs in a daemon thread on node_rank==0. Each ModelRunner registers its
    info via HTTP PUT after model initialization. The Engine
    accesses the collected info directly in-process; external consumers can
    query via HTTP GET.

    Currently supports transfer engine memory registration info.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        # Storage: {tp_rank: (session_id, weights_info_dict)}
        self.transfer_engine_info: Dict[int, Tuple] = {}
        self.lock = threading.Lock()

        app = FastAPI()

        @app.get("/health")
        def health():
            return PlainTextResponse("OK")

        @app.put("/register_transfer_engine_info")
        def register_transfer_engine_info(data: dict):
            try:
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
                return PlainTextResponse("OK")
            except Exception as e:
                logger.error(f"Failed to register engine info: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @app.get("/get_transfer_engine_info")
        def get_transfer_engine_info(rank: int):
            if rank < 0:
                raise HTTPException(status_code=400, detail="Invalid rank parameter")

            with self.lock:
                info = self.transfer_engine_info.get(rank)

            if info is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No transfer engine info for rank {rank}",
                )

            return {"rank": rank, "remote_instance_transfer_engine_info": list(info)}

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            daemon=True,
        )
        self._thread.start()
        logger.info(f"EngineInfoBootstrapServer started on {host}:{port}")

    def close(self):
        self._server.should_exit = True
        self._thread.join(timeout=5)

    def get_transfer_engine_info(self, rank: int) -> Optional[Tuple]:
        """Direct in-process access for co-located HTTP server (no HTTP round-trip)."""
        return self.transfer_engine_info.get(rank)
