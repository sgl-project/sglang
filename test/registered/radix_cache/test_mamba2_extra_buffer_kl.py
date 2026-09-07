"""Regression test: Mamba2 (NemotronH) ``extra_buffer`` state tracking stays
consistent under CUDA-graph decode.

The decode-cache-hit KL check guards the graph-replay track-save path: a decode
that crosses ``--mamba-track-interval`` donates its tracked state to the mamba
radix tree, the second turn hits that decode-seeded prefix, and its logprobs
must match a cold recompute. Before the ``[:bs]`` track-buffer fix in
``MambaAttnBackendBase._replay_metadata``, the captured track-save kernel bound
the stale tail of the static track buffer and scattered the state to the wrong
slot, so the tree cached an unwritten slot and this check fails with KL ~1.5
(vs ~1e-3 healthy).

The existing ``extra_buffer`` KL coverage (test_unified_radix_cache_kl_mamba,
the int8 checkpoint e2e) runs GDN models (Qwen3-Next), whose decode track-save
indexes the track buffer from the front and never hit the bug; NemotronH
(Mamba2) slices it from the tail (``[-num_decodes:]``), so it needs its own
registered coverage.

Usage:
    python3 -m unittest test_mamba2_extra_buffer_kl
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=154, stage="extra-a", runner_config="1-gpu-large")


class TestMamba2ExtraBufferKL(KLDivergenceMixin, DefaultServerBase):
    """NemotronH (Mamba2) + extra_buffer: cache-hit logprobs match cold recompute."""

    model = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"

    # Decode-seeded reuse is the regression trigger (the graphed decode
    # track-save); the broken path fails at KL ~1.5, so 0.005 discriminates
    # cleanly while absorbing bf16 reuse noise. Prefill reuse (chunk-aligned
    # intermediate h states) is inherently looser; 0.012 retains a wide margin
    # below the broken path while covering the observed bf16 calibration noise.
    kl_div_thres = 0.005
    kl_div_thres_prefill = 0.012
    kl_div_max_samples = 16

    other_args = [
        "--max-mamba-cache-size",
        "256",
        "--mem-fraction-static",
        "0.8",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        # The 512-token decode turns must cross a track boundary for the tree
        # to hold decode-seeded states; halve the default interval (must stay
        # >= the model's mamba chunk size, 128) so nearly every sample donates
        # one even when it stops early.
        "--mamba-track-interval",
        "128",
    ]


if __name__ == "__main__":
    unittest.main()
