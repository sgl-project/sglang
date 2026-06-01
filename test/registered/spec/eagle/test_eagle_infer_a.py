"""EAGLE3 spec-decoding: the overlap (spec v2) x no-overlap (spec v1) matrix.

Both cases run the same full kit set on the same standard config (EAGLE3,
topk=1, page_size=1); the only difference is ``disable_overlap``. Config-specific
variants (other models / page sizes / backends / timeouts) live in
test_eagle_infer_b.py.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.matched_stop_kit import MatchedStopMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
    SpecFeatureKit,
    SpecLogprobKit,
    SpecParityKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import SpecEagleServerBase

register_cuda_ci(est_time=900, stage="extra-a", runner_config="1-gpu-large")


class _Eagle3Standard(SpecEagleServerBase):
    """Standard EAGLE3 config shared by the overlap / no-overlap cases."""

    spec_algo = "EAGLE3"
    spec_steps = 5
    spec_topk = 1
    spec_tokens = 6
    attention_backend = "flashinfer"
    max_running_requests = 8
    chunked_prefill_size = 1024
    mem_fraction_static = 0.6  # leave room for the parity reference server

    # Kit thresholds (EAGLE3 topk=1 accepts modestly; smoke measured
    # acc_length~2.05, batch avg accept~1.50, so leave margin below those).
    acc_length_thres = 1.6
    batch_accept_len_thres = 1.3
    gsm8k_score_thres = 0.7
    gsm8k_accept_len_thres = 1.3

    # Busy-time pool accounting check (topk=1 only).
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


_KITS = (
    SpecCorrectnessKit,
    SpecAccuracyKit,
    SpecLogprobKit,
    SpecPenaltyKit,
    SpecFeatureKit,
    MatchedStopMixin,
)


class TestEagle3SpecOverlap(_Eagle3Standard, *_KITS, SpecParityKit):
    """Spec v2 (overlap scheduler on). Also runs the lossless output-parity
    check (the one place the heavy 2nd-server parity is exercised)."""

    disable_overlap = False


class TestEagle3SpecNoOverlap(_Eagle3Standard, *_KITS):
    """Spec v1 (overlap scheduler off)."""

    disable_overlap = True


class TestEagle3Topk16NoOverlap(SpecEagleServerBase, SpecCorrectnessKit, SpecParityKit):
    """EAGLE3 topk>1 tree drafting on spec v1 (preserves the old TestEAGLE3Engine
    correctness coverage: acc-length + EOS + output==reference parity)."""

    spec_algo = "EAGLE3"
    spec_steps = 5
    spec_topk = 16
    spec_tokens = 64
    attention_backend = "flashinfer"
    disable_overlap = True  # topk>1 runs on spec v1
    cuda_graph_max_bs = 5
    mem_fraction_static = 0.6  # room for the parity reference server
    acc_length_thres = 3.1
    batch_accept_len_thres = 1.75


if __name__ == "__main__":
    unittest.main()
