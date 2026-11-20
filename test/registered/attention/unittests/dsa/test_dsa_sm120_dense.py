import unittest

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    DSA_PAGE_SIZE,
    DSAAttentionCase,
    run_dsa_attention_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "DSA dense one-shot SM120 test requires an SM120 GPU.",
)
class TestDSASM120DenseOneShot(CustomTestCase):
    def test_ragged_prefix_matches_reference(self):
        # This exercises DSA's runtime selection and ragged metadata end to end.
        # The SM120 FA4 kernel suite separately covers the production-specific
        # asymmetric QK192/V128 tensor contract against a PyTorch reference.
        case = DSAAttentionCase(
            name="dsa_sm120_mha_one_shot_prefix_ragged",
            backend="dsa",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(3, 8),
            extend_lens=(2, 3),
        )
        run_dsa_attention_case(
            self,
            case,
            head_dim=128,
            require_mha=True,
        )


if __name__ == "__main__":
    unittest.main()
