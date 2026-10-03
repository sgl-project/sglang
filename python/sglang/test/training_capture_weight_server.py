"""Pause the selected test request at a deterministic committed-token boundary."""

import hashlib
import json
import os
import sys
from pathlib import Path

from sglang.srt.managers.io_struct import PauseGenerationReqInput
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.training_capture.protocol import tensor_bytes
from sglang.test.training_capture_pressure_server import Scheduler

_process_batch_result = Scheduler.process_batch_result
_update_weights_from_disk = SchedulerWeightUpdaterManager.update_weights_from_disk
_paused_rids = set()


def process_batch_result(self, batch, result):
    value = _process_batch_result(self, batch, result)
    for req in batch.reqs:
        if (
            req.rid.endswith("-interrupted")
            and req.rid not in _paused_rids
            and req.output_ids
            and not req.finished()
            and req.training_capture_context is not None
        ):
            _paused_rids.add(req.rid)
            self.pause_generation(PauseGenerationReqInput(mode="in_place"))
            root = Path(os.environ["TRAINING_CAPTURE_TEST_OUTPUT"])
            (root / "pause-boundary.json").write_text(
                json.dumps(
                    {
                        "rid": req.rid,
                        "capture_id": req.training_capture_context.lease.capture_id,
                        "output_ids": list(req.output_ids),
                        "kv_committed_len": req.kv_committed_len,
                    }
                )
            )
    return value


Scheduler.process_batch_result = process_batch_result


def update_weights_from_disk(self, request):
    projection = self.tp_worker.model_runner.model.model.layers[0].self_attn.qkv_proj
    value_size = projection.num_kv_heads * projection.v_head_size
    before = projection.weight[-value_size:].detach().cpu().clone()
    result = _update_weights_from_disk(self, request)
    after = projection.weight[-value_size:].detach().cpu().clone()
    root = Path(os.environ["TRAINING_CAPTURE_TEST_OUTPUT"])
    digest = lambda tensor: hashlib.sha256(
        tensor_bytes(tensor.contiguous())
    ).hexdigest()
    (root / "weight-mutation.json").write_text(
        json.dumps(
            {
                "success": result.success,
                "before_sha256": digest(before),
                "expected_after_sha256": digest(-before),
                "after_sha256": digest(after),
            }
        )
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
