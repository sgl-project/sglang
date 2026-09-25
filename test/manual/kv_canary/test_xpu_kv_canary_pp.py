"""KV-canary end-to-end on Intel XPU with pipeline parallelism.

``--pp 2`` routes the run through ``Qwen3ForCausalLM.set_embed_and_head`` (mha mode
is Qwen/Qwen3-0.6B), the embedding/head handoff that syncs and releases the device
cache. Needs two XPU cards, so it is manual until a 2-card lane is confirmed.
"""

from __future__ import annotations

import unittest

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.kv_canary.e2e_base import CanaryE2EBase


class TestXPUCanaryPipelineParallel(CanaryE2EBase):
    """Clean canary run across a pipeline-parallel XPU pair."""

    model_mode = "mha"
    kv_canary_mode = CanaryMode.LOG
    # --disable-cuda-graph is mandatory, not tuning: install_canary refuses a captured decode
    # on a device that routes to the torch reference (host work and D2H, so replay checks nothing).
    extra_server_args = ("--device", "xpu", "--disable-cuda-graph", "--pp", "2")
    # The torch reference folds the chain slot-by-slot on the host, so the workload is much
    # smaller than the CUDA-tuned defaults on the shared base.
    default_parallel_n = 2
    default_max_new_tokens = 32
    default_request_timeout = 120.0

    def test_no_violation(self) -> None:
        self.send_parallel_requests()
        self.assert_no_violation(wait_seconds=2.0)


if __name__ == "__main__":
    unittest.main()
