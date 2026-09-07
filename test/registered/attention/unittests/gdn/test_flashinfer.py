import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    _cache_indices,
    _clone_gdn_cache,
    _restore_gdn_cache,
    build_gdn_attention_fixture,
    make_gdn_cases,
    run_gdn_attention_case,
    run_gdn_fixture_eager,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_gdn_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    _make_spec_verify_input,
    _prepare_target_verify_batch,
    run_gdn_eagle_verify_case,
    run_gdn_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_gdn_split_op_extend_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=14, stage="base-b", runner_config="1-gpu-large")

_cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
_sm_major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
_supports_flashinfer_linear_gdn = _sm_major == 9 or (
    _sm_major == 10 and _cuda_major >= 13
)


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferGDNBackendCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_K_DIM = 64
    HEAD_V_DIM = 64

    CASES = make_gdn_cases("flashinfer")
    CUDA_GRAPH_CASES = (
        GDNAttentionCase(
            name="runner_cuda_graph_gdn_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    SPLIT_OP_CASES = (
        (
            GDNAttentionCase(
                name="runner_split_op_gdn_extend_ragged_page_boundary",
                backend="flashinfer",
                forward_mode=ForwardMode.EXTEND,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
        ),
    )
    EAGLE_VERIFY_CASES = (
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_frozen_kv_mtp_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_dflash_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_ngram_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_tree",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_frozen_kv_mtp_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_dflash_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_ngram_verify_gdn_chain",
                backend="flashinfer",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )

    def test_projected_gdn_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_attention_case(
                    self,
                    case,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )

    # Layout-robustness. See dense/test_triton.py for the rationale.
    LAYOUT_ROBUSTNESS_CASES = (
        GDNAttentionCase(
            name="layout_gdn_extend_two_request",
            backend="flashinfer",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
        ),
        GDNAttentionCase(
            name="layout_gdn_decode_page_boundary",
            backend="flashinfer",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_gdn_attention_case(
                        self,
                        case,
                        head_k_dim=self.HEAD_K_DIM,
                        head_v_dim=self.HEAD_V_DIM,
                        loc_layout=layout,
                    )

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_cuda_graph_decode_case(
                    self,
                    case,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
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
                    run_gdn_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                        head_k_dim=self.HEAD_K_DIM,
                        head_v_dim=self.HEAD_V_DIM,
                    )

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_cuda_graph_case(
                    self,
                    case,
                    topk=topk,
                    spec_kind=spec_kind,
                    head_k_dim=self.HEAD_K_DIM,
                    head_v_dim=self.HEAD_V_DIM,
                )


@unittest.skipUnless(
    torch.cuda.is_available()
    and is_flashinfer_available()
    and _supports_flashinfer_linear_gdn,
    "FlashInfer linear GDN requires SM90 or SM100/SM103 with CUDA 13+",
)
class TestFlashInferLinearGDNBackendCorrectness(CustomTestCase):
    # FlashInfer's DSL prefill kernels require head size 128 on SM90 and SM100.
    HEAD_DIM = 128
    CHECKPOINT_CASE = GDNAttentionCase(
        name="flashinfer_gdn_prefill_state_checkpoints",
        backend="triton",
        linear_attn_prefill_backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=2,
        num_v_heads=4,
        page_size=16,
        prefix_lens=(0, 64, 128),
        extend_lens=(64, 65, 129),
    )

    def test_prefill_tracked_state_checkpoints(self):
        fixture = build_gdn_attention_fixture(
            self,
            self.CHECKPOINT_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=320,
            runner_batch_size=6,
        )
        batch = fixture.forward_batch
        # Simulate the tracking metadata produced by the extra-buffer scheduler.
        # This test covers checkpoint mapping and state copies, not scheduler setup.
        batch.mamba_track_mask = torch.ones(3, dtype=torch.bool, device="cuda")
        batch.mamba_track_indices = torch.tensor(
            [4, 5, 6], dtype=torch.int64, device="cuda"
        )
        batch.mamba_track_seqlens = torch.tensor(
            # The final entry selects the second checkpoint at absolute S256.
            [64, 129, 257],
            dtype=torch.int64,
            device="cuda",
        )

        cache = fixture.runner.req_to_token_pool.mamba2_layer_cache(0)
        initial_conv = cache.conv[0].clone()
        initial_ssm = cache.temporal.clone()
        flashinfer_output = run_gdn_fixture_eager(fixture)
        flashinfer_tracked = cache.temporal[batch.mamba_track_indices].clone()

        cache.conv[0].copy_(initial_conv)
        cache.temporal.copy_(initial_ssm)
        fixture.backend.linear_attn_backend.kernel_dispatcher.extend_kernel = (
            TritonGDNKernel()
        )
        triton_output = run_gdn_fixture_eager(fixture)
        triton_tracked = cache.temporal[batch.mamba_track_indices]

        torch.testing.assert_close(
            flashinfer_output, triton_output, atol=3e-2, rtol=3e-2
        )
        torch.testing.assert_close(
            flashinfer_tracked, triton_tracked, atol=3e-2, rtol=3e-2
        )

    def _install_flashinfer_verify(self, fixture):
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            FlashInferGDNKernel,
        )

        verify_kernel = FlashInferGDNKernel()
        self.assertTrue(verify_kernel.supports_target_verify)
        dispatcher = fixture.backend.linear_attn_backend.kernel_dispatcher
        dispatcher.verify_kernel = verify_kernel
        dispatcher.verify_kernel_is_flashinfer = True

    def _run_padded_verify_pair(
        self,
        *,
        real_prefix_lens: tuple[int, ...],
        draft_token_num: int,
        padded_prefix_lens: tuple[int, ...],
    ) -> None:
        real_batch_size = len(real_prefix_lens)
        padded_batch_size = len(padded_prefix_lens)
        real_case = GDNAttentionCase(
            name=(
                f"flashinfer_gdn_verify_reference_b{real_batch_size}_t{draft_token_num}"
            ),
            backend="triton",
            linear_attn_prefill_backend="flashinfer",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_k_heads=2,
            num_v_heads=4,
            page_size=16,
            prefix_lens=real_prefix_lens,
            extend_lens=(draft_token_num,) * real_batch_size,
        )
        padded_case = GDNAttentionCase(
            name=(
                f"flashinfer_gdn_verify_padded_b{real_batch_size}_"
                f"p{padded_batch_size}_t{draft_token_num}"
            ),
            backend=real_case.backend,
            linear_attn_prefill_backend=real_case.linear_attn_prefill_backend,
            forward_mode=real_case.forward_mode,
            num_k_heads=real_case.num_k_heads,
            num_v_heads=real_case.num_v_heads,
            page_size=real_case.page_size,
            prefix_lens=padded_prefix_lens,
            extend_lens=(draft_token_num,) * padded_batch_size,
        )
        max_context_len = max(padded_prefix_lens) + draft_token_num + 1
        reference = build_gdn_attention_fixture(
            self,
            real_case,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=max_context_len,
        )
        padded = build_gdn_attention_fixture(
            self,
            padded_case,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=max_context_len,
            runner_batch_size=padded_batch_size,
        )
        self._install_flashinfer_verify(reference)
        self._install_flashinfer_verify(padded)

        real_num_tokens = real_batch_size * draft_token_num
        with torch.no_grad():
            padded.actual_module.A_log.copy_(reference.actual_module.A_log)
            padded.actual_module.dt_bias.copy_(reference.actual_module.dt_bias)
            padded.mixed_qkv[:real_num_tokens].copy_(reference.mixed_qkv)
            padded.a[:real_num_tokens].copy_(reference.a)
            padded.b[:real_num_tokens].copy_(reference.b)

            reference_cache = _clone_gdn_cache(reference)
            padded_cache = _clone_gdn_cache(padded)
            reference_indices = _cache_indices(reference)
            padded_indices = _cache_indices(padded)
            padded_cache[0][padded_indices[:real_batch_size]] = reference_cache[0][
                reference_indices
            ]
            padded_cache[1][padded_indices[:real_batch_size]] = reference_cache[1][
                reference_indices
            ]
            _restore_gdn_cache(reference, reference_cache)
            _restore_gdn_cache(padded, padded_cache)

        for fixture, is_padded in ((reference, False), (padded, True)):
            _prepare_target_verify_batch(fixture.forward_batch, fixture.case, "cuda")
            fixture.forward_batch.spec_info = _make_spec_verify_input(
                fixture.case,
                fixture.forward_batch,
                topk=1,
                device="cuda",
                spec_kind="eagle",
            )
            if is_padded:
                # Mirror MLP-sync metadata: physical rows remain in the batch,
                # while only the first real requests own target-verify tokens.
                fixture.forward_batch._original_batch_size = real_batch_size
                fixture.forward_batch.global_num_token_non_padded = torch.tensor(
                    real_num_tokens, dtype=torch.int32, device="cuda"
                )
                fixture.forward_batch.global_num_token_non_padded_cpu = real_num_tokens

        reference_output = run_gdn_fixture_eager(reference)
        padded_output = run_gdn_fixture_eager(padded)
        expected_query_start_loc = torch.tensor(
            [
                *range(0, real_num_tokens + 1, draft_token_num),
                *([real_num_tokens] * (padded_batch_size - real_batch_size)),
            ],
            dtype=torch.int32,
            device="cuda",
        )
        torch.testing.assert_close(
            padded.backend.linear_attn_backend.forward_metadata.query_start_loc,
            expected_query_start_loc,
        )
        torch.testing.assert_close(
            padded.backend.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :real_batch_size
            ],
            padded_indices[:real_batch_size],
        )
        self.assertTrue(
            torch.all(
                padded.backend.linear_attn_backend.forward_metadata.mamba_cache_indices[
                    real_batch_size:
                ]
                == -1
            ).item()
        )
        reference_intermediate = (
            reference.runner.req_to_token_pool.mamba2_layer_cache(0)
            .intermediate_ssm[:real_batch_size]
            .clone()
        )
        padded_intermediate = (
            padded.runner.req_to_token_pool.mamba2_layer_cache(0)
            .intermediate_ssm[:real_batch_size]
            .clone()
        )

        torch.testing.assert_close(
            padded_output[:, :real_num_tokens],
            reference_output,
            atol=3e-2,
            rtol=3e-2,
        )
        torch.testing.assert_close(
            padded_output[:, real_num_tokens:],
            torch.zeros_like(padded_output[:, real_num_tokens:]),
        )
        torch.testing.assert_close(
            padded_intermediate,
            reference_intermediate,
            atol=3e-2,
            rtol=3e-2,
        )

    def test_target_verify_padded_requests_match_flashinfer_reference(self):
        # 18 real tokens in a 24-token physical bucket: old FI verify inferred
        # B=4 and T=4, instead of the real B=3, T=6.
        self._run_padded_verify_pair(
            real_prefix_lens=(4, 7, 5),
            draft_token_num=6,
            padded_prefix_lens=(4, 7, 5, 0),
        )

    def test_target_verify_divisible_padding_matches_flashinfer_reference(self):
        # A divisible padded shape still catches the old inference: 4 real
        # tokens in an 8-token bucket must stay B=1, T=4 rather than B=2, T=2.
        self._run_padded_verify_pair(
            real_prefix_lens=(4,),
            draft_token_num=4,
            padded_prefix_lens=(4, 0),
        )

    def test_target_verify_padded_idle_rank_preserves_state(self):
        case = GDNAttentionCase(
            name="flashinfer_gdn_verify_idle_rank",
            backend="triton",
            linear_attn_prefill_backend="flashinfer",
            forward_mode=ForwardMode.TARGET_VERIFY,
            num_k_heads=2,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(4, 7, 5, 0),
            extend_lens=(6, 6, 6, 6),
        )
        fixture = build_gdn_attention_fixture(
            self, case, head_k_dim=self.HEAD_DIM, head_v_dim=self.HEAD_DIM
        )
        self._install_flashinfer_verify(fixture)
        _prepare_target_verify_batch(fixture.forward_batch, case, "cuda")
        fixture.forward_batch.spec_info = _make_spec_verify_input(
            case, fixture.forward_batch, topk=1, device="cuda", spec_kind="eagle"
        )
        fixture.forward_batch._original_batch_size = 0
        fixture.forward_batch.global_num_token_non_padded_cpu = 0
        original_cache = _clone_gdn_cache(fixture)
        output = run_gdn_fixture_eager(fixture)
        self.assertEqual(tuple(output.shape), (1, 24, 4, self.HEAD_DIM))
        torch.testing.assert_close(output, torch.zeros_like(output))
        for actual, expected in zip(_clone_gdn_cache(fixture), original_cache):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        metadata = fixture.backend.linear_attn_backend.forward_metadata
        torch.testing.assert_close(
            metadata.query_start_loc, torch.zeros_like(metadata.query_start_loc)
        )
        self.assertTrue(torch.all(metadata.mamba_cache_indices == -1).item())


if __name__ == "__main__":
    unittest.main()
