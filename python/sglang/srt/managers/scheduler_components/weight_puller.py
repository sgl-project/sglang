import logging
import threading
import traceback
from typing import Any, Optional, Tuple

import torch

from sglang.srt.managers.io_struct import PullWeightsReqInput, PullWeightsReqOutput
from sglang.srt.managers.scheduler_components.ipc_channels import (
    SchedulerIpcChannels,
)
from sglang.srt.model_loader.weight_utils import download_weights_from_hf
from sglang.srt.runtime_context import get_exec, get_model, get_parallel
from sglang.srt.weight_sync import local_checkpoint

logger = logging.getLogger(__name__)


class SchedulerWeightPuller:
    """Runs /pull_weights off the event loop so generation keeps going during the disk copy."""

    def __init__(self, *, tp_cpu_group: Any, ipc_channels: SchedulerIpcChannels):
        self._tp_cpu_group = tp_cpu_group
        self._ipc_channels = ipc_channels
        self._pending: Optional[Tuple[PullWeightsReqInput, threading.Thread]] = None
        self._error: Optional[str] = None

    def handle(self, recv_req: PullWeightsReqInput) -> Optional[PullWeightsReqOutput]:
        if self._pending is not None:
            return PullWeightsReqOutput(
                success=False, message="Another pull_weights is already in progress."
            )
        # check_pending syncs over tp_cpu_group, which needs every TP rank to see the request
        # in the same iteration; local control broadcast delivers it per DP group instead
        if (
            get_parallel().enable_dp_attention_local_control_broadcast
            or get_exec().moe.is_ep_scale_joiner
        ):
            return PullWeightsReqOutput(
                success=False,
                message="pull_weights does not support DP-attention local control broadcast.",
            )
        thread = threading.Thread(target=self._pull, args=(recv_req,), daemon=True)
        thread.start()
        self._pending = (recv_req, thread)
        return None

    def _pull(self, recv_req: PullWeightsReqInput) -> None:
        model = get_model()
        try:
            local_checkpoint.pull(
                local_checkpoint_dir=recv_req.local_checkpoint_dir,
                # the served weights are already on disk, so an HF repo id resolves to its cache
                base_dir=download_weights_from_hf(
                    model.model_path,
                    cache_dir=model.download_dir,
                    allow_patterns=["*.safetensors"],
                    revision=model.revision,
                ),
                source_dir=recv_req.source_dir,
                target_version=recv_req.target_version,
                pre_read_hook=model.custom_pull_weights_pre_read_hook,
            )
        except Exception:
            self._error = traceback.format_exc()
            logger.error(self._error)

    def check_pending(self) -> None:
        if self._pending is None:
            return
        recv_req, thread = self._pending
        # ranks finish in different iterations; agree on one before the gather
        done = torch.tensor([int(not thread.is_alive())])
        torch.distributed.all_reduce(
            done, op=torch.distributed.ReduceOp.MIN, group=self._tp_cpu_group
        )
        if not done.item():
            return
        thread.join()
        errors = [None] * torch.distributed.get_world_size(group=self._tp_cpu_group)
        torch.distributed.all_gather_object(
            errors, self._error, group=self._tp_cpu_group
        )
        self._pending, self._error = None, None
        failed = [error for error in errors if error is not None]
        self._ipc_channels.send_to_tokenizer.send_output(
            PullWeightsReqOutput(
                success=not failed, message="; ".join(failed) or "Success."
            ),
            recv_req,
        )
