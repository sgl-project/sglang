"""WeightBridge receiver for one TP rank of an SGLang engine.

WeightBridge (meta-pytorch/monarch, examples/weightbridge) owns the weight-update control
plane: a coordinator process per engine relays the trainer's one-time connect, and every
later update is signalled by the trainer's own RDMA data flags. The scheduler therefore
never receives an HTTP or tokenizer request for an update; it calls :meth:`poll` once per
loop iteration on every TP rank, and rank 0's hub broadcast keeps the ranks in lockstep.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Callable
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class WeightBridgeReceiver:
    """Builds the coordinator (rank 0) and this rank's ``ReceiverAdapter`` at startup."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        hf_checkpoint: str,
        engine_port: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        from wbridge.backend import coordinator
        from wbridge.backend.control_channel import (
            coordinator_ipc,
            coordinator_tcp_port,
        )
        from wbridge.frontend.adapters import AdapterContext, ReceiverAdapter
        from wbridge.utils.specgen import hf_weights_from_checkpoint

        self._tp_rank = tp_rank
        ipc = coordinator_ipc(engine_port)
        self._coordinator: Optional[subprocess.Popen] = (
            coordinator.spawn(ipc, coordinator_tcp_port(engine_port))
            if tp_rank == 0
            else None
        )

        hf_weights, hf_shapes = hf_weights_from_checkpoint(hf_checkpoint)
        ctx = AdapterContext(
            hf_weights=hf_weights,
            hf_shapes=hf_shapes,
            wksd_factory=lambda: dict(model.named_parameters()),
            load_weights=lambda fetcher: model.load_weights(
                (name, factory()) for name, factory in fetcher.items()
            ),
            rank=tp_rank,
        )
        # Probing load_weights overwrites every parameter, then restores it from the checkpoint,
        # which is what a freshly started engine holds.
        with torch.no_grad():
            self._adapter = ReceiverAdapter(ctx, ipc, num_workers=tp_size)
        logger.info(
            f"[wbridge] receiver ready: {tp_rank=} {tp_size=} {engine_port=} coordinator={ipc}"
        )

    def poll(self, *, quiesce: Callable[[], None]) -> Optional[int]:
        """Service one control round; return the update's epoch once its weights are loaded.

        ``quiesce`` runs on every rank after WeightBridge's GO and before any weight is written.
        """
        epoch: list[Optional[int]] = [None]

        def before_receive(update_epoch: int) -> None:
            epoch[0] = update_epoch
            quiesce()

        with torch.no_grad():
            loaded = self._adapter.poll_requests(before_receive=before_receive)
        if not loaded:
            return None
        # EWTT endpoint: wall clock right after the update is loaded, before generation resumes.
        logger.info(
            f"[wbridge] consume_end epoch={epoch[0]} tp_rank={self._tp_rank} "
            f"t_ns={time.time_ns()}"
        )
        return epoch[0]

    def flush_profile_outputs(self) -> None:
        """Release WeightBridge profiling output once the scheduler has resumed (rollout block_end)."""
        self._adapter.flush_profile_outputs()
