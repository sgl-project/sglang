import unittest
from contextlib import contextmanager
from dataclasses import replace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
    make_swa_no_prefix_input_config_cases,
    make_swa_prefix_input_config_cases,
    run_dense_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_dense_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_dense_spec_verify_case,
    run_dense_spec_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_dense_split_op_extend_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=13, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferSWAAttentionBackendCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_DIM = 64
    HIDDEN_SIZE = 256

    CASES = make_swa_no_prefix_input_config_cases(
        "flashinfer"
    ) + make_swa_prefix_input_config_cases("flashinfer")
    # Paged-only prefill has no ragged pass / custom prefix mask, so the kernel
    # must enforce the window; the long case puts tokens past the window.
    PAGED_MODE_CASES = CASES + (
        DenseAttentionCase(
            name="swa_extend_no_prefix_above_window_long",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(6, 8, 12),
            sliding_window_size=4,
        ),
    )
    # Above-window decode case requires the `extend_window` reference rule
    # (window+1 keys), not the `min_seq_len_window` rule — FlashInfer's
    # decode metadata uses `clamp(seq_lens, max=window+1)` per
    # `flashinfer_backend.py:1031`. See `_SWA_DECODE_EXTEND_WINDOW` in
    # `common/attention_methods/dense_attention.py`.
    CUDA_GRAPH_CASES = (
        DenseAttentionCase(
            name="runner_cuda_graph_swa_decode_within_window",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(1, 2, 3),
            sliding_window_size=4,
        ),
        DenseAttentionCase(
            name="runner_cuda_graph_swa_decode_above_window",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(7, 8, 9),
            sliding_window_size=4,
        ),
    )
    # NOTE: a `runner_split_op_swa_extend_prefix_within_window` clone of the
    # triton SWA test fails on flashinfer (~0.21 max diff). FlashInfer's
    # prefill-split path does not handle SWA prefix the same way as triton;
    # the projected EXTEND covers the prefix path through the unsplit kernel
    # which does match the reference. Investigate before adding split_op
    # prefix to flashinfer SWA.
    SPLIT_OP_CASES = (
        (
            DenseAttentionCase(
                name="runner_split_op_swa_extend_no_prefix_window_edges",
                backend="flashinfer",
                forward_mode=ForwardMode.EXTEND,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(0, 0, 0),
                extend_lens=(3, 4, 5),
                sliding_window_size=4,
            ),
            16,
        ),
    )
    SPEC_VERIFY_CASES = (
        (
            DenseAttentionCase(
                name="runner_dflash_verify_swa_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "dflash",
        ),
        (
            DenseAttentionCase(
                name="runner_dflash_verify_swa_window_edges",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                # Straddle the window: one request below, one at, one above.
                prefix_lens=(1, 4, 9),
                extend_lens=(3, 3, 3),
                sliding_window_size=4,
            ),
            1,
            "dflash",
        ),
    )
    SPEC_VERIFY_CUDA_GRAPH_CASES = (
        (
            DenseAttentionCase(
                name="runner_cuda_graph_dflash_verify_swa_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_heads=4,
                num_kv_heads=4,
                page_size=16,
                prefix_lens=(3, 5),
                extend_lens=(3, 3),
                sliding_window_size=4,
            ),
            1,
            "dflash",
        ),
    )

    def test_projected_swa_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_attention_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_projected_swa_attention_cases_paged_mode(self):
        for case in self.PAGED_MODE_CASES:
            with self.subTest(case=case.name, backend=case.backend, mode="paged"):
                with envs.SGLANG_FLASHINFER_USE_PAGED.override(True):
                    run_dense_attention_case(
                        self,
                        case,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                    )

    @contextmanager
    def _swa_fixture(self, case, *, head_dim, max_context_len=64, use_paged=False):
        with envs.SGLANG_FLASHINFER_USE_PAGED.override(use_paged):
            fixture = build_dense_attention_fixture(
                self,
                case,
                head_dim=head_dim,
                hidden_size=self.HIDDEN_SIZE,
                max_context_len=max_context_len,
                dtype=torch.bfloat16,
            )
            try:
                with (
                    torch.no_grad(),
                    forward_context(ForwardContext(attn_backend=fixture.backend)),
                ):
                    yield fixture
            finally:
                fixture.runner._server_args_override.restore()

    def test_window_511_prefill_and_cached_decode_boundary_keys(self):
        # Orthogonal key coordinates give the selected sentinel a logit of 32.
        # Its value dominates when visible; masking it out yields an exactly
        # representable uniform mean. A missing/extra boundary key therefore
        # changes outputs by about 1 or more, independently of GEMM rounding.
        num_tokens = 514
        q = torch.zeros(num_tokens, 4, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(num_tokens, 2, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros_like(k)
        for head in range(3):
            q[:, head, head] = 16
            k[head, :, head] = 16
            v[head] = 2**head
        for offset, position in enumerate((511, 512, 513)):
            q[position, 3, 3 + offset] = 16
            k[position, :, 3 + offset] = 16
            v[position] = 8
        expected = (
            torch.tensor(
                [[1, 2, 4, 8], [22 / 512, 2, 4, 8], [28 / 512, 28 / 512, 4, 8]],
                dtype=torch.bfloat16,
                device="cuda",
            )
            .unsqueeze(-1)
            .expand(-1, -1, 64)
        )

        case = DenseAttentionCase(
            name="swa_511_dominant_boundary_keys",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(num_tokens,),
            sliding_window_size=511,
        )
        for use_paged in (False, True):
            with (
                self.subTest(use_paged=use_paged),
                self._swa_fixture(
                    case, head_dim=64, max_context_len=528, use_paged=use_paged
                ) as fixture,
            ):
                batch = fixture.forward_batch

                def attend(batch, start, end):
                    fixture.backend.init_forward_metadata(batch)
                    return fixture.actual_module.attn(
                        q[start:end], k[start:end], v[start:end], batch
                    ).view(-1, 4, 64)

                prefix = replace(
                    batch,
                    input_ids=batch.input_ids[:511],
                    positions=batch.positions[:511],
                    out_cache_loc=batch.out_cache_loc[:511],
                    seq_lens=batch.seq_lens.new_tensor([511]),
                    seq_lens_cpu=torch.tensor([511], dtype=torch.int32),
                    seq_lens_sum=511,
                    extend_seq_lens=batch.extend_seq_lens.new_tensor([511]),
                    extend_seq_lens_cpu=[511],
                    extend_num_tokens=511,
                )
                attend(prefix, 0, 511)
                for offset, position in enumerate((511, 512, 513)):
                    decode = ForwardBatch(
                        forward_mode=ForwardMode.DECODE,
                        batch_size=1,
                        input_ids=batch.input_ids[position : position + 1],
                        req_pool_indices=batch.req_pool_indices,
                        seq_lens=batch.seq_lens.new_tensor([position + 1]),
                        seq_lens_cpu=torch.tensor([position + 1], dtype=torch.int32),
                        seq_lens_sum=position + 1,
                        positions=batch.positions[position : position + 1],
                        out_cache_loc=batch.out_cache_loc[position : position + 1],
                    )
                    actual = attend(decode, position, position + 1)
                    with self.subTest(phase="decode", position=position):
                        torch.testing.assert_close(
                            actual[0], expected[offset], atol=1e-3, rtol=0
                        )

                actual = attend(batch, 0, num_tokens)
                for offset, position in enumerate((511, 512, 513)):
                    with self.subTest(phase="prefill", position=position):
                        torch.testing.assert_close(
                            actual[position], expected[offset], atol=1e-3, rtol=0
                        )

    def test_window_511_mixed_prefill_boundary_keys(self):
        case = DenseAttentionCase(
            name="swa_511_mixed_prefill_boundary_keys",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(512, 513),
            sliding_window_size=511,
        )
        with self._swa_fixture(case, head_dim=64, max_context_len=528) as fixture:
            q = torch.zeros(1025, 4, 64, dtype=torch.bfloat16, device="cuda")
            k = torch.zeros(1025, 2, 64, dtype=torch.bfloat16, device="cuda")
            v = torch.zeros_like(k)
            q[:, :, 0] = 16
            k[[0, 512], :, 0] = 16
            v[[0, 512]] = 8
            # Each first key has logit 32: it dominates the shorter
            # sequence's last row and is masked out of the longer one.
            batch = fixture.forward_batch
            fixture.backend.init_forward_metadata(batch)
            actual = fixture.actual_module.attn(q, k, v, batch).view(-1, 4, 64)
            torch.testing.assert_close(
                actual[[511, 1024]],
                actual.new_tensor([8, 0]).view(2, 1, 1).expand(2, 4, 64),
                atol=1e-3,
                rtol=0,
            )

    def test_long_prefix_and_extend_masks(self):
        # Both portions exceed the window: cached keys must leave the window
        # while new keys first enter it, then leave it within the same extend.
        case = DenseAttentionCase(
            name="swa_long_prefix_and_extend_masks",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(7,),
            extend_lens=(7,),
            sliding_window_size=4,
        )
        for head_dim in (128, 256):
            with (
                self.subTest(head_dim=head_dim),
                self._swa_fixture(
                    case, head_dim=head_dim, max_context_len=32
                ) as fixture,
            ):
                q = torch.zeros(7, 4, head_dim, device="cuda", dtype=torch.bfloat16)
                k = torch.zeros(14, 4, head_dim, device="cuda", dtype=torch.bfloat16)
                v = torch.zeros_like(k)
                q[:, :, 0] = 16
                sentinel_positions = torch.tensor([0, 3, 7, 12], device="cuda")
                heads = torch.arange(4, device="cuda")
                k[sentinel_positions, heads, 0] = 16
                v[sentinel_positions, heads] = 8
                layer = fixture.actual_module.attn
                prefix_locs = fixture.runner.req_to_token_pool.req_to_token[
                    0, :7
                ].long()
                fixture.runner.token_to_kv_pool.set_kv_buffer(
                    layer, prefix_locs, k[:7], v[:7]
                )

                # Each head has one nonzero-value key. When visible,
                # its softmax denominator contains it and four zeros.
                score = q.new_tensor(256 * layer.scaling, dtype=torch.float32)
                visible_value = 8 * score.exp() / (score.exp() + 4)
                positions = torch.arange(7, 14, device="cuda").unsqueeze(1)
                visible = (sentinel_positions >= positions - 4) & (
                    sentinel_positions <= positions
                )
                expected = (visible * visible_value).to(q.dtype)
                batch = fixture.forward_batch
                fixture.backend.init_forward_metadata(batch)
                actual = layer(q, k[7:], v[7:], batch).view(7, 4, head_dim)
                torch.testing.assert_close(
                    actual,
                    expected.unsqueeze(-1).expand_as(actual),
                    atol=0.04,
                    rtol=0,
                )

    def test_sliding_and_full_layers_reuse_same_prefill_metadata(self):
        case = DenseAttentionCase(
            name="swa_full_swa_shared_prefill_metadata",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(7,),
            sliding_window_size=4,
        )
        with self._swa_fixture(case, head_dim=128) as fixture:
            sliding = fixture.actual_module.attn
            full = RadixAttention(
                num_heads=4,
                head_dim=128,
                scaling=sliding.scaling,
                num_kv_heads=2,
                layer_id=0,
                sliding_window_size=-1,
            )
            q = torch.zeros(7, 4, 128, device="cuda", dtype=torch.bfloat16)
            k = torch.zeros(7, 2, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.zeros_like(k)
            q[:, :, 0] = 32
            k[0, :, 0] = 32
            v[0] = 8
            batch = fixture.forward_batch
            fixture.backend.init_forward_metadata(batch)
            # Do not re-plan between layers. Both wrappers must retain
            # their own window specialization across repeated use.
            for layer in (sliding, full, sliding):
                actual = layer(q, k, v, batch, save_kv_cache=False).view(7, 4, 128)
                expected = actual.new_full((7, 4, 128), 8)
                if layer is sliding:
                    expected[5:] = 0
                torch.testing.assert_close(actual, expected, atol=1e-3, rtol=0)

    def test_full_layer_ragged_prefill_reads_cached_prefix_only(self):
        """Full attention must ignore stale K/V in unwritten extend slots."""
        case = DenseAttentionCase(
            name="swa_full_layer_ragged_prefix_only",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(7,),
            extend_lens=(7,),
            sliding_window_size=4,
        )
        with envs.SGLANG_FLASHINFER_USE_PAGED.override(False):
            fixture = build_dense_attention_fixture(
                self, case, head_dim=128, max_context_len=32, dtype=torch.bfloat16
            )
        self.addCleanup(fixture.runner._server_args_override.restore)
        full = RadixAttention(
            num_heads=4,
            head_dim=128,
            scaling=fixture.actual_module.attn.scaling,
            num_kv_heads=4,
            layer_id=0,
            sliding_window_size=-1,
        )
        q = torch.zeros(7, 4, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.zeros(14, 4, 128, dtype=torch.bfloat16, device="cuda")
        v = torch.zeros_like(k)
        q[:, :, 0] = 16
        k[0, :, 0] = 16
        v[0] = 8
        pool = fixture.runner.token_to_kv_pool
        prefix_locs = fixture.runner.req_to_token_pool.req_to_token[0, :7].long()
        pool.set_kv_buffer(full, prefix_locs, k[:7], v[:7])
        batch = fixture.forward_batch
        poison_k = torch.zeros_like(k[7:])
        poison_k[:, :, 0] = 16
        pool.set_kv_buffer(
            full, batch.out_cache_loc, poison_k, torch.full_like(v[7:], -8)
        )
        with (
            torch.no_grad(),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            fixture.backend.init_forward_metadata(batch)
            actual = full(q, k[7:], v[7:], batch, save_kv_cache=False).view(7, 4, 128)
        torch.testing.assert_close(
            actual, actual.new_full((7, 4, 128), 8), atol=0.04, rtol=0
        )

    # Layout-robustness. See dense/test_triton.py for the full rationale.
    # The default `shuffled_pages` is already exercised by
    # test_projected_swa_attention_cases on the existing case list.
    # This method opts into the more aggressive interleaved_pages +
    # non_monotonic_extend on within-window extend + decode.
    LAYOUT_ROBUSTNESS_CASES = (
        DenseAttentionCase(
            name="layout_swa_extend_below_window",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(10,),
            sliding_window_size=12,
        ),
        DenseAttentionCase(
            name="layout_swa_decode_within_window",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_heads=8,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(8, 10),
            sliding_window_size=12,
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_dense_attention_case(
                        self,
                        case,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                        loc_layout=layout,
                    )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_dense_cuda_graph_decode_case(
                    self,
                    case,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_split_op_extend_cases(self):
        for case, static_num_tokens in self.SPLIT_OP_CASES:
            for breakable in (False, True):
                runner = "bcg" if breakable else "pcg"
                with self.subTest(
                    case=case.name,
                    backend=case.backend,
                    runner=runner,
                ):
                    run_dense_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                        head_dim=self.HEAD_DIM,
                        hidden_size=self.HIDDEN_SIZE,
                    )

    def test_runner_mode_spec_verify_cases(self):
        for case, topk, spec_kind in self.SPEC_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_dense_spec_verify_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )

    def test_runner_mode_spec_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.SPEC_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_dense_spec_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_dim=self.HEAD_DIM,
                    hidden_size=self.HIDDEN_SIZE,
                )


if __name__ == "__main__":
    unittest.main()
