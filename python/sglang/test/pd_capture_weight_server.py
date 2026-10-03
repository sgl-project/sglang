"""Observe real P/D weight loading while the existing chunk gate holds a request."""

import hashlib
import os
import sys
from pathlib import Path

from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.training_capture.protocol import tensor_bytes
from sglang.test.pd_capture_control_server import write_observation

_update = SchedulerWeightUpdaterManager.update_weights_from_disk


def update_weights_from_disk(self, request):
    projection = self.tp_worker.model_runner.model.model.layers[0].self_attn.qkv_proj
    size = projection.num_kv_heads * projection.v_head_size
    before = projection.weight[-size:].detach().cpu().clone()
    result = _update(self, request)
    after = projection.weight[-size:].detach().cpu().clone()
    root = Path(self.tp_worker.training_capture.config.journal_directory).parent
    digest = lambda tensor: hashlib.sha256(
        tensor_bytes(tensor.contiguous())
    ).hexdigest()
    write_observation(
        root / "weight-mutation.json",
        {
            "success": result.success,
            "before_sha256": digest(before),
            "expected_after_sha256": digest(-before),
            "after_sha256": digest(after),
        },
    )
    return result


SchedulerWeightUpdaterManager.update_weights_from_disk = update_weights_from_disk


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
