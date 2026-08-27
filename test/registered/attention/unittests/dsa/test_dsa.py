import sys
import unittest
from pathlib import Path

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    DSA_DECODE_IMPL_VARIANTS,
    DSA_PAGE_SIZE,
    DSA_PREFILL_IMPL_VARIANTS,
    DSAAttentionCase,
    make_dsa_dense_fallback_cases,
    make_dsa_sparse_cases,
    run_dsa_attention_case,
    run_dsa_sparse_attention_case,
    run_dsa_sparse_cuda_graph_decode_impl_variant_case,
    run_dsa_sparse_decode_impl_variant_case,
    run_dsa_sparse_fp8_decode_case,
    run_dsa_sparse_fp8_prefill_case,
    run_dsa_sparse_prefill_impl_variant_case,
    run_dsa_sparse_speculative_forward_mode_case,
    run_dsa_sparse_tilelang_decode_case,
    run_dsa_sparse_tilelang_prefill_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dsa_sparse_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_runner import (
    run_dsa_eagle_draft_cuda_graph_runner_case,
)

register_cuda_ci(est_time=25, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")


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
                # GB300 (SM10.x) kernel requires 128-dim query/value;
                # use head_dim=128 rather than the generic DEFAULT_HEAD_DIM=16.
                run_dsa_attention_case(self, case, head_dim=128)

    def test_sparse_topk_cases(self):
        for case in self.SPARSE_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_sparse_attention_case(self, case)

    # Non-trailing index layouts. The reference gathers Q/K via
    # `fixture.topk_rows`, so any valid permutation of keys in
    # `[0, key_count)` produces a matching reference. These layouts
    # exercise the kernel's non-contiguous gather path (production
    # top-k by attention score is not naturally trailing for long
    # prefixes). Use long-prefix decode where `key_count > index_topk`
    # so the pattern actually subsamples (with key_count <= topk,
    # strided/head_tail collapse back to the trailing case).
    NON_TRAILING_INDEX_CASES = (
        (
            DSAAttentionCase(
                name="dsa_sparse_decode_strided_index_long_prefix",
                backend="dsa",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=1,
                page_size=DSA_PAGE_SIZE,
                prefix_lens=(2048,),
            ),
            "strided",
        ),
        (
            DSAAttentionCase(
                name="dsa_sparse_decode_head_tail_index_long_prefix",
                backend="dsa",
                forward_mode=ForwardMode.DECODE,
                num_heads=4,
                num_kv_heads=1,
                page_size=DSA_PAGE_SIZE,
                prefix_lens=(2048,),
            ),
            "head_tail",
        ),
    )

    def test_sparse_non_trailing_index_cases(self):
        for case, pattern in self.NON_TRAILING_INDEX_CASES:
            with self.subTest(case=case.name, backend=case.backend, pattern=pattern):
                run_dsa_sparse_attention_case(self, case, index_pattern=pattern)

    # Layout-robustness. See dense/test_triton.py for the rationale.
    # shuffled_pages is the default for all DSA tests via
    # build_dsa_attention_fixture / build_dsa_sparse_attention_fixture;
    # this method opts into the more aggressive interleaved_pages +
    # non_monotonic_extend layouts on representative dense fallback and
    # sparse top-k cases.
    LAYOUT_DENSE_CASES = (
        DSAAttentionCase(
            name="layout_dsa_dense_fallback_two_request",
            backend="dsa",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(0, 32),
            extend_lens=(32, 16),
        ),
    )
    LAYOUT_SPARSE_CASES = (
        DSAAttentionCase(
            name="layout_dsa_sparse_decode_long_prefix",
            backend="dsa",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(2048,),
        ),
    )

    def test_layout_robustness_dense_cases(self):
        for case in self.LAYOUT_DENSE_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                with self.subTest(case=case.name, layout=layout):
                    run_dsa_attention_case(self, case, head_dim=128, loc_layout=layout)

    def test_layout_robustness_sparse_cases(self):
        for case in self.LAYOUT_SPARSE_CASES:
            for layout in ("interleaved_pages",):
                with self.subTest(case=case.name, layout=layout):
                    run_dsa_sparse_attention_case(self, case, loc_layout=layout)

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

    # Speculative forward-mode coverage. TARGET_VERIFY and
    # DRAFT_EXTEND_V2 both route through the `dsa_decode_impl`
    # dispatcher (the same kernel selection as plain DECODE) but
    # produce different `seqlens_expanded` and `cu_seqlens_q` from
    # `dsa_backend.py:469-529`. `DSAMockModelRunner.__init__` derives
    # `speculative_num_draft_tokens` from `case.extend_lens` so deep_gemm
    # JIT-compiles with a non-zero aligned batch size.
    SPECULATIVE_FORWARD_MODE_CASES = (
        DSAAttentionCase(
            name="dsa_sparse_target_verify",
            backend="dsa",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_heads=4,
            num_kv_heads=1,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(128,),
            extend_lens=(3,),
        ),
        DSAAttentionCase(
            name="dsa_sparse_draft_extend_v2",
            backend="dsa",
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            num_heads=4,
            num_kv_heads=1,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(128,),
            extend_lens=(3,),
        ),
    )

    def test_sparse_speculative_forward_mode_cases(self):
        for case in self.SPECULATIVE_FORWARD_MODE_CASES:
            with self.subTest(case=case.name, mode=case.forward_mode.name):
                run_dsa_sparse_speculative_forward_mode_case(self, case)

    # FP8 KV cache (`dsa_kv_cache_store_fp8=True`) — the production
    # deployment dtype. Switches `DSATokenToKVPool` to packed
    # FP8-nope/BF16-rope storage at 656 bytes/token; `set_mla_kv_buffer`
    # routes through `quantize_k_cache_separate` and the kernel reads
    # FP8 directly. The reference stays on BF16 K (independent of the
    # cache bytes), and `DSA_SPARSE_FP8_ATOL=0.2` absorbs FP8 quant
    # noise — same separation principle as the DSV4 SWA fixture so a
    # silent pack/write bug cannot corrupt both paths identically.
    #
    # FP8 + `flashmla_sparse` prefill + EXTEND + non-empty prefix is the
    # only combo that hits `TopkTransformMethod.RAGGED`
    # (`get_topk_transform_method`), which exercises
    # `dequantize_k_cache_paged` and the `topk_indices_offset` shift —
    # paths that the BF16 default suite never reaches.
    FP8_PREFILL_RAGGED_CASE = DSAAttentionCase(
        name="dsa_sparse_fp8_prefill_ragged_topk",
        backend="dsa",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        # Long prefix → above MHA threshold, RAGGED topk transform
        prefix_lens=(2048,),
        extend_lens=(1,),
    )
    FP8_PREFILL_PAGED_CASE = DSAAttentionCase(
        name="dsa_sparse_fp8_prefill_paged_topk",
        backend="dsa",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(2048,),
        extend_lens=(1,),
    )
    FP8_DECODE_CASE = DSAAttentionCase(
        name="dsa_sparse_fp8_decode",
        backend="dsa",
        forward_mode=ForwardMode.DECODE,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(128,),
    )

    def test_sparse_fp8_prefill_cases(self):
        for impl in DSA_PREFILL_IMPL_VARIANTS:
            with self.subTest(impl=impl):
                # Each impl that isn't in `DSA_FP8_COMPATIBLE_PREFILL_IMPLS`
                # emits skipTest from the helper with the reason. The
                # `flashmla_sparse` impl hits the RAGGED-topk path; the
                # others stay on PAGED.
                case = (
                    self.FP8_PREFILL_RAGGED_CASE
                    if impl == "flashmla_sparse"
                    else self.FP8_PREFILL_PAGED_CASE
                )
                run_dsa_sparse_fp8_prefill_case(self, case, dsa_prefill_backend=impl)

    def test_sparse_fp8_decode_cases(self):
        for impl in DSA_DECODE_IMPL_VARIANTS:
            with self.subTest(impl=impl):
                run_dsa_sparse_fp8_decode_case(
                    self, self.FP8_DECODE_CASE, dsa_decode_backend=impl
                )

    # Tilelang sparse cases — dedicated topk=2048 fixture.
    # `tilelang_sparse_fwd` asserts `topk == 2048` at
    # `dsa/tilelang_kernel.py:1345`, so this fixture variant carries a
    # 2048-wide trailing-topk row builder. Prefix length must be >= 2048
    # to produce a real (non-padded) topk row.
    TILELANG_PREFILL_CASE = DSAAttentionCase(
        name="dsa_sparse_tilelang_prefill",
        backend="dsa",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(4096,),
        extend_lens=(1,),
    )
    TILELANG_DECODE_CASE = DSAAttentionCase(
        name="dsa_sparse_tilelang_decode",
        backend="dsa",
        forward_mode=ForwardMode.DECODE,
        num_heads=4,
        num_kv_heads=1,
        page_size=DSA_PAGE_SIZE,
        prefix_lens=(4096,),
    )

    def test_sparse_tilelang_prefill_case(self):
        run_dsa_sparse_tilelang_prefill_case(self, self.TILELANG_PREFILL_CASE)

    def test_sparse_tilelang_decode_case(self):
        run_dsa_sparse_tilelang_decode_case(self, self.TILELANG_DECODE_CASE)

    # EAGLE production draft CUDA-graph runner integration. Wires DSA
    # through `speculative_draft_runner.py`'s shared
    # `EagleDraftCudaGraphRunnerAdapter` (same lifecycle as DSV4 /
    # dense / MLA). DSA's chain-only constraint comes from the
    # synthesized topk_indices path — tree draft needs parent-indices
    # plumbing through that synthesis; deferred.
    EAGLE_DRAFT_CASES = (
        DSAAttentionCase(
            name="runner_eagle_draft_decode_cuda_graph_dsa_chain",
            backend="dsa",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=1,
            page_size=DSA_PAGE_SIZE,
            prefix_lens=(128, 192),
        ),
    )

    def test_runner_mode_eagle_draft_cuda_graph_runner_cases(self):
        for case in self.EAGLE_DRAFT_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dsa_eagle_draft_cuda_graph_runner_case(self, case)

    # CG decode replay with FP8 KV cache. Captures and replays through
    # `flashmla_kv` (the only FP8-compatible decode kernel). The
    # `_clone_dsa_sparse_cache` hook is reused as-is — it snapshots the
    # raw uint8 K buffer bytes, which round-trip correctly across
    # capture/replay regardless of bf16 vs FP8 packing.
    def test_sparse_fp8_cuda_graph_decode_case(self):
        from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
            run_dsa_sparse_cuda_graph_decode_case,
        )

        run_dsa_sparse_cuda_graph_decode_case(
            self,
            self.FP8_DECODE_CASE,
            dsa_decode_backend="flashmla_kv",
            fp8_kv_cache=True,
        )

    # CG decode replay parametrized over `dsa_decode_backend` impl. The
    # `flashmla_kv` baseline is already covered by
    # `test_runner_mode_cuda_graph_decode_cases`; this method extends the
    # CG matrix to every supported decode impl (`flashmla_sparse` /
    # `flashmla_kv` / `fa3` on H200, with `tilelang` / `trtllm` / `aiter`
    # skip-gated). Each impl re-builds the fixture with the impl forced
    # so the captured graph uses that specific kernel.
    def test_sparse_cuda_graph_decode_impl_variants(self):
        for impl in DSA_DECODE_IMPL_VARIANTS:
            with self.subTest(impl=impl):
                run_dsa_sparse_cuda_graph_decode_impl_variant_case(
                    self, self.CUDA_GRAPH_DECODE_CASES[0], impl
                )


if __name__ == "__main__":
    unittest.main()
