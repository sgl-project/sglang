import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.attention_methods.dsa_attention import (
    DSA_DECODE_IMPL_VARIANTS,
    DSA_PAGE_SIZE,
    DSA_PREFILL_IMPL_VARIANTS,
    DSAAttentionCase,
    make_dsa_dense_fallback_cases,
    make_dsa_sparse_cases,
    run_dsa_attention_case,
    run_dsa_sparse_attention_case,
    run_dsa_sparse_decode_impl_variant_case,
    run_dsa_sparse_prefill_impl_variant_case,
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

    # DSA implementation-variant matrix. DSA exposes multiple kernel
    # implementations (`flashmla_sparse`, `flashmla_kv`, `fa3`, `tilelang`,
    # `trtllm`, `aiter`) selected by `--dsa-prefill-backend` /
    # `--dsa-decode-backend`. Each variant maps to a distinct kernel path
    # in `dsa_backend.py`; `dsa_impl_capability` gates per hardware/SDK so
    # impls unavailable on the test box (e.g., `trtllm` requires SM100+,
    # `aiter` requires HIP) emit a clean `skipTest` with a reason rather
    # than spuriously failing.
    PREFILL_IMPL_CASE = DSAAttentionCase(
        name="dsa_sparse_prefill_impl_variant",
        backend="dsa",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        # Long prefix keeps the backend on the MLA path (above the
        # MHA_ONE_SHOT short-sequence threshold) so the impl override
        # actually routes through `dsa_prefill_impl`.
        prefix_lens=(2048,),
        extend_lens=(1,),
    )
    DECODE_IMPL_CASE = DSAAttentionCase(
        name="dsa_sparse_decode_impl_variant",
        backend="dsa",
        forward_mode=ForwardMode.DECODE,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(128,),
    )

    def test_sparse_prefill_impl_variants(self):
        for impl in DSA_PREFILL_IMPL_VARIANTS:
            with self.subTest(impl=impl):
                run_dsa_sparse_prefill_impl_variant_case(
                    self, self.PREFILL_IMPL_CASE, impl
                )

    def test_sparse_decode_impl_variants(self):
        for impl in DSA_DECODE_IMPL_VARIANTS:
            with self.subTest(impl=impl):
                run_dsa_sparse_decode_impl_variant_case(
                    self, self.DECODE_IMPL_CASE, impl
                )


if __name__ == "__main__":
    unittest.main()
