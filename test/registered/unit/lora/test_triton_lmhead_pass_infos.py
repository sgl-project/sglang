"""Regression: TritonLoRABackend._prepare_lm_head_batch_info must collect the
per-pass lm_head batch infos into the LOCAL list it returns.

Bug: it created `lm_head_pass_batch_infos = []` but appended to
`self.lm_head_pass_batch_infos` (which init_lm_head_config sets to None), then
returned the empty local list. So the first batch that reaches the pass-segments
branch (lm_head LoRA + chunked logprobs) crashed with
`AttributeError: 'NoneType' object has no attribute 'append'`, and the consumer
would otherwise index an empty list. The sibling chunked_backend appends to the
local list; this matches it.

Drives the real method with the segment helpers stubbed to reach the branch.
"""

import unittest
import unittest.mock as mock

import sglang.srt.lora.backend.triton_backend as tb
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTritonLmHeadPassInfos(unittest.TestCase):
    def test_pass_infos_collected_not_dropped(self):
        be = TritonLoRABackend.__new__(TritonLoRABackend)
        # init_lm_head_config sets this to None; the buggy code appended to it.
        be.lm_head_pass_batch_infos = None
        be._build_lm_head_batch_info = lambda *a, **k: "INFO"
        be._get_lm_head_pass_segments = lambda wi, pl: [([0], [2, 2]), ([1], [2, 2])]

        with mock.patch.object(
            tb, "get_lm_head_pruned_lens", lambda fb: [2, 2]
        ), mock.patch.object(tb, "merge_and_chunk_segments", lambda *a, **k: "SEG"):
            _, pass_infos = be._prepare_lm_head_batch_info(
                forward_batch=None, weight_indices=[0, 0], batch_info=None
            )

        # One LoRABatchInfo per pass; previously this raised or returned [].
        self.assertEqual(pass_infos, ["INFO", "INFO"])


if __name__ == "__main__":
    unittest.main()
