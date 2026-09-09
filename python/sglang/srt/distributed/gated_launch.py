import logging
import threading
import time
from typing import Optional

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from sglang.srt.distributed import get_world_group

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 1.0
LOG_INTERVAL_SECONDS = 10.0

_instance: Optional["_GatedLaunchServer"] = None


def maybe_wait_for_gated_launch(*, host: str, port: Optional[int]) -> None:
    global _instance

    if port is None or _instance is not None:
        return

    world_group = get_world_group()

    _instance = _GatedLaunchServer()
    if world_group.rank_in_group == 0:
        _instance.serve(host=host, port=port)

    logger.info(f"Gated launch waiting for activation. rank={world_group.rank}")
    tic = time.perf_counter()
    _wait_until_activated(world_group=world_group, server=_instance)
    logger.info(f"Gated launch activated. elapsed={time.perf_counter() - tic:.2f} s")


def _wait_until_activated(*, world_group, server: "_GatedLaunchServer") -> None:
    activated = torch.zeros(1, dtype=torch.int32)
    started_at = time.perf_counter()
    next_log_at = started_at + LOG_INTERVAL_SECONDS

    while True:
        activated[0] = int(server.activated)

        if world_group.world_size > 1:
            dist.broadcast(
                activated,
                src=world_group.ranks[0],
                group=world_group.cpu_group,
            )

        if bool(activated[0]):
            return

        if (now := time.perf_counter()) >= next_log_at:
            logger.info(
                f"Gated launch still waiting for activation. "
                f"rank={world_group.rank} elapsed={now - started_at:.0f} s"
            )
            next_log_at = now + LOG_INTERVAL_SECONDS

        time.sleep(POLL_INTERVAL_SECONDS)


class _GatedLaunchServer:
    def __init__(self):
        self.activated = False
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def serve(self, *, host: str, port: int) -> None:
        config = uvicorn.Config(
            _build_app(self), host=host, port=port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        logger.info(f"Gated launch control server started on {host}:{port}")


def _build_app(server: _GatedLaunchServer) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return PlainTextResponse("OK")

    @app.post("/gate/activate")
    def activate():
        server.activated = True
        return PlainTextResponse("OK")

    return app
