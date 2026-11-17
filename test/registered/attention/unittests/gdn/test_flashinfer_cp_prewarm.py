"""The FlashInfer GDN prefill prewarm under tensor parallelism on SM100, where
the context-parallel kernel's final-state fixup specializes on sequences times
heads, so one sequence and two take different fixups."""

import unittest

import torch

from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_prewarm import (
    assert_prewarm_leaves_nothing_to_compile,
    prewarm_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="extra-b", runner_config="4-gpu-b200")

_cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
_sm_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0


@unittest.skipUnless(
    is_flashinfer_available() and _sm_major == 10 and _cuda_major >= 13,
    "the fixup specializations are SM100's (CUDA 13+)",
)
class TestFlashInferContextParallelPrewarm(CustomTestCase):
    def test_prewarm_covers_each_context_parallel_sequence_count(self):
        # Qwen3.5-4B's heads per rank under tensor parallelism 4: FlashInfer
        # keeps up to a few sequences on its context-parallel kernel.
        assert_prewarm_leaves_nothing_to_compile(
            self,
            [
                prewarm_case("one_sequence", 4, 8, (0,), (192,), [129]),
                prewarm_case("two_sequences", 4, 8, (0, 64), (129, 65), [129, 129]),
                prewarm_case(
                    "four_sequences",
                    4,
                    8,
                    (0, 64, 0, 128),
                    (64, 65, 129, 129),
                    [64, 129, 129, 257],
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
