"""PD disaggregation for Inkling with MXFP8 KV and a hierarchical prefill cache.

Inkling ships three heterogeneous state components -- full-attention KV,
sliding-window KV and ShortConv state -- and MXFP8 adds a block-scale component
per KV sub-pool, each addressed like the KV it describes. A transfer that drops
or misaligns any of them collapses generation rather than shaving accuracy.

MXFP8 KV needs SM100+, so this is Blackwell-only.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-b200")

MODEL = os.environ.get(
    "INKLING_SMALL_TEST_MODEL_PATH", "thinkingmachines/Inkling-Small-NVFP4"
)

# The unified radix tree is what merges the three components into one tree, so
# it is a precondition rather than a tuning knob here.
ENV = {"SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1"}

# Shared with the single-server Inkling recipe; TP is per role.
COMMON_ARGS = [
    "--tp",
    "2",
    "--quantization",
    "modelopt_fp4",
    "--attention-backend",
    "fa4",
    "--page-size",
    "128",
    "--fp4-gemm-backend",
    "flashinfer_trtllm",
    "--moe-runner-backend",
    "flashinfer_trtllm_routed",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--mem-fraction-static",
    "0.6",
    "--swa-full-tokens-ratio",
    "0.1",
    "--mamba-full-memory-ratio",
    "0.1",
    "--kv-cache-dtype",
    "mxfp8",
]


class TestDisaggregationInklingMXFP8(PDDisaggregationServerBase, GSM8KMixin):
    # Shot count and floor match the single-server Inkling case so the two numbers
    # are comparable. This pair measured 0.845 to 0.865 across runs, closer to the
    # floor than that case is; a dropped or misaligned state component collapses
    # generation to near zero, which is what the floor is sized to catch.
    gsm8k_num_shots = 10
    gsm8k_score_threshold = 0.80

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(MODEL)
        cls.launch_all()

    @classmethod
    def start_prefill(cls):
        # HiCache rides the prefill role only: the decode role forces chunk cache,
        # and its radix opt-in is refused for sliding-window models.
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            *COMMON_ARGS,
            "--enable-hierarchical-cache",
            # Absolute host size, not the default ratio: the ratio scales with the
            # device pool, and two roles on one node then reserve more host memory
            # than a runner has. This exercises the tiering, not its capacity.
            "--hicache-size",
            "8",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env={**os.environ, **ENV},
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            *COMMON_ARGS,
            "--base-gpu-id",
            "2",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env={**os.environ, **ENV},
        )


if __name__ == "__main__":
    unittest.main()
