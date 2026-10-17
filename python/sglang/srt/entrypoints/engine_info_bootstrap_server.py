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
from typing import Dict, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from sglang.srt.observability.trace import fastapi_telemetry_disabled_kwargs

logger = logging.getLogger(__name__)


class EngineInfoBootstrapServer:
    """Lightweight HTTP server for per-rank model info registration.

    Runs in a daemon thread on node_rank==0. Each ModelRunner registers one rank-info record via HTTP PUT after
    model initialization: its parallelism config and, when its weights are registered with a transfer engine, their
    locations and the session to reach them. External consumers query via HTTP GET. Each model runner of a rank
    registers under its weight-update role (``target``, or a draft role such as ``draft``), so a speculative draft
    never replaces its target's entry.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        # {(role, tp_rank): {"parallelism_config", "weight_locations", "transfer_session_id"}}
        self.rank_info: Dict[Tuple[str, int], dict] = {}
        self.lock = threading.Lock()

        app = FastAPI(**fastapi_telemetry_disabled_kwargs())

        @app.get("/health")
        def health():
            return PlainTextResponse("OK")

        @app.put("/register_rank_info")
        def register_rank_info(data: dict):
            try:
                role = data["role"]
                tp_rank = data["tp_rank"]
                rank_info = data["rank_info"]
                with self.lock:
                    self.rank_info[(role, tp_rank)] = rank_info
                logger.info(
                    f"Registered rank info for role={role} tp_rank={tp_rank}, "
                    f"transfer_session_id={rank_info['transfer_session_id']}"
                )
                return PlainTextResponse("OK")
            except Exception as e:
                logger.error(f"Failed to register rank info: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        def get_or_404(rank: int, role: str) -> dict:
            if rank < 0:
                raise HTTPException(status_code=400, detail="Invalid rank parameter")
            with self.lock:
                rank_info = self.rank_info.get((role, rank))
            if rank_info is None:
                raise HTTPException(
                    status_code=404, detail=f"No rank info for role {role} rank {rank}"
                )
            return rank_info

        @app.get("/get_rank_info")
        def get_rank_info(rank: int, role: str = "target"):
            return get_or_404(rank, role)

        @app.get("/get_transfer_engine_info")
        def get_transfer_engine_info(rank: int, role: str = "target"):
            rank_info = get_or_404(rank, role)
            if rank_info["transfer_session_id"] is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No transfer engine info for role {role} rank {rank}",
                )
            return {
                "rank": rank,
                "remote_instance_transfer_engine_info": [
                    rank_info["transfer_session_id"],
                    rank_info["weight_locations"],
                ],
            }

        @app.get("/get_parallelism_config")
        def get_parallelism_config(rank: int, role: str = "target"):
            rank_info = get_or_404(rank, role)
            if rank_info["parallelism_config"] is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No parallelism config for role {role} rank {rank}",
                )
            return rank_info["parallelism_config"]

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
