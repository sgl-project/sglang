import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.dsa_attention import (
    DSA_PAGE_SIZE,
    DSAAttentionCase,
    make_dsa_dense_fallback_cases,
    make_dsa_sparse_cases,
    run_dsa_attention_case,
    run_dsa_sparse_attention_case,
)
from common.runner_modes.cuda_graph_decode_runner import (
    run_dsa_sparse_cuda_graph_decode_case,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSAAttentionBackendCorrectness(CustomTestCase):
    CASES = make_dsa_dense_fallback_cases("dsa")
    SPARSE_CASES = make_dsa_sparse_cases("dsa")
    # PCG/BCG split-op extend coverage is *not* added here — DSA's
    # MHA_ONE_SHOT dense fallback passes K as concatenated prefix+extend
    # (length = sum(seq_lens)) to `module.attn`, but
    # `unified_attention_with_output` (`radix_attention.py:170-208`) slices
    # K to `forward_batch.num_token_non_padded_cpu` (= live extend-token
    # count), under the per-token K convention used by Triton/FlashInfer/
    # FA. The K-slice removes the prefix portion, so DSA's dense fallback
    # output diverges by ~50% mismatch under piecewise CG. See
    # dsa/README.md "Production-Unsupported" for the path forward.

    def test_mha_one_shot_dense_fallback_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_attention_case(self, case)

    def test_sparse_topk_cases(self):
        for case in self.SPARSE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_sparse_attention_case(self, case)

    # CG decode replay via the sparse `flashmla_kv` path (cached MLA latent
    # KV, written by `_populate_dsa_sparse_prefix_kv` at fixture build).
    # Unlike the MHA_ONE_SHOT dense fallback (where K is passed inline as
    # prefix+extend and `unified_attention_with_output` slicing breaks
    # piecewise CG), sparse decode reads cached K and is CG-compatible.
    CUDA_GRAPH_DECODE_CASES = (
        DSAAttentionCase(
            name="runner_cuda_graph_dsa_sparse_decode_flashmla_kv",
            backend="dsa",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(127, 128),
        ),
    )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_DECODE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_sparse_cuda_graph_decode_case(self, case)


if __name__ == "__main__":
    unittest.main()
