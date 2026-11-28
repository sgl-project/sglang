"""End-to-end tests of --cuda-graph-persistence on two GPUs (TP2, EP2, TP2 + EP2),
with the sequence and assertions of test_cuda_graph_persistence.py."""

import importlib.util
import unittest

# Module import, not the class;
# unittest would collect an imported class and run the per-commit tests here too.
import test_cuda_graph_persistence as e2e

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=679, stage="base-b", runner_config="2-gpu-large")

# Dummy weights, no download;
# the graphs, their restore and output agreement are exercised, model quality is not.
_MODEL = "Qwen/Qwen3-30B-A3B-FP8"
# Single-node DeepEP: no InfiniBand discovery, no per-team NCCL communicators;
# Foundry pins NVSHMEM_DISABLE_NCCL on SAVE / LOAD, set here for native too.
_EP_ENV = {
    "NVSHMEM_DISABLE_NCCL": "1",
    "NCCL_IB_DISABLE": "1",
    "NVSHMEM_REMOTE_TRANSPORT": "none",
}
# In-graph all-reduce on torch symmetric memory, custom all-reduce off; Foundry pins it.
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
    extra_server_args = ["--tp-size", "2", "--load-format", "dummy"] + _COLLECTIVES


_HAS_DEEPEP = importlib.util.find_spec("deep_ep") is not None


@unittest.skipUnless(_HAS_DEEPEP, "DeepEP (deep_ep) is not installed")
class TestFoundryGraphPersistenceEP2(e2e.TestFoundryGraphPersistence):
    model = _MODEL
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
    native_failure_skips = True
    extra_env = _EP_ENV
    extra_server_args = (
        ["--tp-size", "2", "--ep-size", "2", "--load-format", "dummy"]
        + _DEEPEP
        + _COLLECTIVES
    )


if __name__ == "__main__":
    unittest.main(verbosity=3)
