"""The Gluon sparse-prefill gather scratch cap follows SGLANG_MINIMAX_GLUON_PREFILL_SCRATCH_MB
and the pool dtype (a batch of long-prefix extends must not fall back to Triton at 2M tokens)."""

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.minimax_sparse_ops import gluon_prefill as gp
from sglang.test.test_utils import CustomTestCase


class TestGluonPrefillScratchCap(CustomTestCase):
    def test_default_cap_covers_the_agentx_span(self):
        # 51,866 pages (3.3M tokens across one prefill batch) was observed in the
        # AgentX replay and fell back under the former 32,768-page cap.
        self.assertGreater(gp._max_scratch_pages(torch.bfloat16), 51_866)
        self.assertGreater(gp._max_scratch_pages(torch.float8_e4m3fnuz), 51_866)

    def test_cap_is_dtype_aware_and_env_configurable(self):
        with envs.SGLANG_MINIMAX_GLUON_PREFILL_SCRATCH_MB.override(512):
            self.assertEqual(gp._max_scratch_pages(torch.bfloat16), 32_768)
            self.assertEqual(gp._max_scratch_pages(torch.float8_e4m3fnuz), 65_536)
        with envs.SGLANG_MINIMAX_GLUON_PREFILL_SCRATCH_MB.override(2048):
            self.assertEqual(gp._max_scratch_pages(torch.bfloat16), 131_072)


if __name__ == "__main__":
    unittest.main()
