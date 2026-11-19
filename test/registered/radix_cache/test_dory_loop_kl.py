"""Prefill-vs-decode KL consistency for Dory's per-loop KV cache.

Each recurrent attention layer owns one KV slot per loop, so the radix cache holds
loop-indexed entries keyed by a virtual layer id. A restore that keys the wrong slot, or a
sliding-window eviction that drops tokens a later loop still reads, shows up here as a
nonzero KL while a plain generation test would pass.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

MODEL = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
LOOP_OVERRIDES = '{"n_recurrent_loops": 2, "recurrent_kv_cache_mode": "per_loop"}'

register_cuda_ci(est_time=300, stage="nightly", runner_config="1-gpu-large")


class TestDoryPerLoopKL(KLDivergenceMixin, DefaultServerBase):
    model = MODEL
    # TODO(hermes): calibrate on the first GPU run; this value is inherited from the SWA KL
    # test and has not been measured on Dory.
    kl_div_thres = 0.02
    # Long enough to walk past the 1024-token sliding window during decode.
    kl_div_decode_max_new_tokens = 2048
    other_args = [
        "--mem-fraction-static",
        "0.70",
        "--hf-overrides",
        LOOP_OVERRIDES,
        "--cuda-graph-backend-prefill=disabled",
    ]


if __name__ == "__main__":
    unittest.main()
