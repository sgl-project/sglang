"""End-to-end tests of --cuda-graph-persistence on two GPUs: the parallel
layouts Foundry persists, same native / SAVE / LOAD sequence and assertions
as test_cuda_graph_persistence.py. Qwen3-30B-A3B-FP8 with dummy weights
(--load-format dummy, no download): the graphs, their restore and the
outputs' agreement between the engines are exercised, model quality is not.

- TP2: attention and experts tensor-parallel; the in-graph all-reduce runs on
  torch symmetric memory (custom all-reduce off), which Foundry pins.
- EP2: DP attention + expert parallel over DeepEP low-latency; only the
  sequential outputs are compared (DP attention composes batches per rank).
- TP2 + EP2: tensor-parallel attention with expert parallel on the same GPUs.

Needs the optional dependency sglang[foundry]. The EP rows need DeepEP and a
usable NVSHMEM transport: without deep_ep they are skipped, and a native
engine that does not start on the runner skips the row instead of failing it.

Run:  python3 test/registered/e2e/cuda_graph/persistence/test_cuda_graph_persistence_multigpu.py
"""

import importlib.util
import unittest

# Module import (not the class): unittest would collect an imported class and
# run the per-commit tests here as well.
import test_cuda_graph_persistence as e2e

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, stage="base-b", runner_config="2-gpu-large")

_MODEL = "Qwen/Qwen3-30B-A3B-FP8"
# Single-node DeepEP: no InfiniBand discovery, no per-team NCCL communicators
# (Foundry pins NVSHMEM_DISABLE_NCCL on SAVE / LOAD; set for native too).
_EP_ENV = {
    "NVSHMEM_DISABLE_NCCL": "1",
    "NCCL_IB_DISABLE": "1",
    "NVSHMEM_REMOTE_TRANSPORT": "none",
}
_COLLECTIVES = ["--disable-custom-all-reduce", "--enable-torch-symm-mem"]
_DEEPEP = [
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--moe-runner-backend",
    "deep_gemm",
    "--disable-radix-cache",
    "--chunked-prefill-size",
    "256",
]


class TestFoundryGraphPersistenceTP2(e2e.TestFoundryGraphPersistence):
    model = _MODEL
    decode_bs = [1, 2, 4, 8]
    extra_server_args = ["--tp-size", "2", "--load-format", "dummy"] + _COLLECTIVES


_HAS_DEEPEP = importlib.util.find_spec("deep_ep") is not None


@unittest.skipUnless(_HAS_DEEPEP, "DeepEP (deep_ep) is not installed")
class TestFoundryGraphPersistenceEP2(e2e.TestFoundryGraphPersistence):
    model = _MODEL
    decode_bs = [1, 2, 4, 8]
    compare_batched = False
    native_failure_skips = True
    extra_env = _EP_ENV
    extra_server_args = (
        [
            "--tp-size",
            "2",
            "--attn-dp-size",
            "2",
            "--ep-size",
            "2",
            "--load-format",
            "dummy",
        ]
        + _DEEPEP
        + _COLLECTIVES
    )


@unittest.skipUnless(_HAS_DEEPEP, "DeepEP (deep_ep) is not installed")
class TestFoundryGraphPersistenceTP2EP2(e2e.TestFoundryGraphPersistence):
    model = _MODEL
    decode_bs = [1, 2, 4, 8]
    native_failure_skips = True
    extra_env = _EP_ENV
    extra_server_args = (
        ["--tp-size", "2", "--ep-size", "2", "--load-format", "dummy"]
        + _DEEPEP
        + _COLLECTIVES
    )


if __name__ == "__main__":
    unittest.main(verbosity=3)
