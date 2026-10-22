"""EAGLE-4 output parity and API regression matrix on CUDA.

Install the eagle4 extra, then run:
  python -m pytest test/manual/spec/eagle/test_eagle4.py -v

Override EAGLE4_TARGET_MODEL and EAGLE4_DRAFT_MODEL with local checkpoint paths.
Each class starts a reference server, stops it, then starts a speculative server.
"""

import os
import unittest

from sglang.test.kits.spec_server_kits import (
    SpecLogprobKit,
    SpecParityKit,
    SpecPenaltyKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import SpecEagleServerBase


class _Eagle4Base(SpecEagleServerBase):
    model = os.environ.get("EAGLE4_TARGET_MODEL", "Qwen/Qwen3.8-27B")
    draft_model = os.environ.get("EAGLE4_DRAFT_MODEL", "EAGLE-Corp/Qwen3.8-27B-EAGLE4")
    spec_algo = "EAGLE_HARNESS"
    spec_steps = 12
    spec_topk = 4
    spec_tokens = 16
    max_running_requests = 8
    cuda_graph_max_bs_decode = 8
    chunked_prefill_size = 4096
    enable_deterministic_inference = True
    mem_fraction_static = 0.75
    extra_args = (
        "--speculative-draft-model-quantization",
        "unquant",
        "--speculative-algorithm-config",
        '{"mode":"static"}',
        "--disable-radix-cache",
    )


class TestEagle4Static(SpecParityKit, SpecLogprobKit, SpecPenaltyKit, _Eagle4Base):
    disable_overlap = True


class TestEagle4Overlap(SpecParityKit, _Eagle4Base):
    disable_overlap = False


class TestEagle4Dynamic(SpecParityKit, _Eagle4Base):
    disable_overlap = True
    extra_args = (
        "--speculative-draft-model-quantization",
        "unquant",
        "--speculative-algorithm-config",
        '{"mode":"serve"}',
        "--disable-radix-cache",
    )


class TestEagle4Eager(SpecParityKit, _Eagle4Base):
    disable_overlap = True
    extra_args = (*_Eagle4Base.extra_args, "--disable-cuda-graph")


if __name__ == "__main__":
    unittest.main()
