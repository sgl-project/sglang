"""AMD (gfx950) coverage for the DSA MHA_ONE_SHOT dense-prefill fallback.

The dense-MHA prefill path (`_forward_standard_mha`) is enabled on gfx950
(MI350/MI355) for DSA models so that short-context prefills (<= the dense-attn
KV-len threshold) skip the lightning-indexer logits GEMM + top-k. This file
runs the same correctness cases as the CUDA suite's
``test_mha_one_shot_dense_fallback_cases``
(test/registered/attention/unittests/dsa/test_dsa.py), but registered on the
AMD CI runner.

Only the dense-fallback cases are included here: the sparse DSA cases in the
CUDA suite rely on CUDA-only kernels (flashmla_kv / flashmla_sparse / fa3) that
have no gfx950 implementation, so they are intentionally not registered for AMD.
"""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    make_dsa_dense_fallback_cases,
    run_dsa_attention_case,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-large-amd")


@unittest.skipIf(not (torch.cuda.is_available() and is_hip()), "AMD/ROCm GPU required")
class TestDSADenseMHAFallbackAMD(CustomTestCase):
    """gfx950 dense MHA_ONE_SHOT fallback (aiter flash_attn_varlen_func)."""

    CASES = make_dsa_dense_fallback_cases("dsa")

    def test_mha_one_shot_dense_fallback_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_attention_case(self, case, head_dim=128)


if __name__ == "__main__":
    unittest.main()
