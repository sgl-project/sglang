"""gfx950 coverage for the DSA MHA_ONE_SHOT dense-prefill fallback.

The dense-MHA prefill path (`DeepseekSparseAttnBackend._forward_standard_mha`)
is enabled on gfx95x by the `_IS_GFX95` clause of the `use_mha` gate in
`set_dsa_prefill_impl`, so short-context prefills (<= the dense-attn KV-len
threshold, which defaults to the model's `index_topk`) skip the lightning
indexer entirely and run aiter's `flash_attn_varlen_func` instead of sparse
MLA. That gate has no AMD CI coverage: the CUDA suite
(test/registered/attention/unittests/dsa/test_dsa.py) exercises the same cases
on SM90/SM100 kernels only.

This file runs the CUDA suite's dense-fallback cases on an MI35x runner so the
gfx950 kernel path is checked too. Only the dense-fallback cases are included:
the sparse DSA cases rely on flashmla_kv / flashmla_sparse / fa3, which have no
gfx950 implementation.
"""

import unittest

import torch

from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    make_dsa_dense_fallback_cases,
    run_dsa_attention_case,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

# The `use_mha` gate keys off `_IS_GFX95`, so MI300/MI325 runners would take the
# sparse path and never reach the kernel under test.
_IS_GFX95_GPU = torch.cuda.is_available() and is_hip() and is_gfx95_supported()

# The shared cases are built at `DSA_PAGE_SIZE` (64), but the HIP DSA pool only
# accepts a paged layout on the aiter preshuffle path — the legacy path asserts
# `page_size == 1`. Preshuffle needs triton >= 3.5.0 or
# AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1, so it is absent on some ROCm images.
_RUNNABLE = _IS_GFX95_GPU and aiter_can_use_preshuffle_paged_mqa()


@unittest.skipUnless(
    _RUNNABLE,
    "requires an AMD gfx95x (MI350/MI355) GPU with the aiter preshuffle paged-MQA path",
)
class TestDSADenseMHAFallbackGfx950(CustomTestCase):
    CASES = make_dsa_dense_fallback_cases("dsa")

    def test_mha_one_shot_dense_fallback_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                # Match the CUDA suite: head_dim=128 rather than the generic
                # DEFAULT_HEAD_DIM=16.
                run_dsa_attention_case(self, case, head_dim=128)


if __name__ == "__main__":
    unittest.main()
