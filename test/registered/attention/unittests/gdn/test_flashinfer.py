import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    build_gdn_attention_fixture,
    make_gdn_cases,
    run_gdn_attention_case,
    run_gdn_fixture_eager,
)
from sglang.test.kits.attention_unittest.attention_methods.gdn_prewarm import (
    assert_prewarm_leaves_nothing_to_compile,
    prewarm_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_gdn_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_gdn_eagle_verify_case,
    run_gdn_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_gdn_split_op_extend_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=24, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=31, stage="base-b", runner_config="1-gpu-large")

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

        cache = fixture.runner.req_to_token_pool.mamba_pool.mamba2_layer_cache(0)
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

    def test_prewarm_leaves_no_prefill_variant_to_compile_while_serving(self):
        # Qwen3.5-like heads: FlashInfer runs one sequence on its
        # context-parallel kernel and three on its native kernel, on SM90 and
        # on SM100.
        assert_prewarm_leaves_nothing_to_compile(
            self,
            [
                prewarm_case("one_sequence", 16, 32, (0,), (192,), [129]),
                prewarm_case(
                    "three_sequences",
                    16,
                    32,
                    (0, 64, 128),
                    (64, 65, 129),
                    [64, 129, 257],
                ),
            ],
        )


@unittest.skipUnless(torch.cuda.is_available(), "the prewarm runs on a CUDA stream")
class TestFlashInferGDNPrewarmPolicy(CustomTestCase):
    """Which prefills the startup prewarm runs, with FlashInfer's prefill
    replaced by a recorder; the compile coverage itself is checked above."""

    CHECKPOINT_EVERY = 64

    def _kernel(self, prefill_fn):
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            FlashInferGDNKernel,
        )

        kernel = object.__new__(FlashInferGDNKernel)
        kernel.use_state_pool = True
        kernel._prefill_needs_fp32_state = False
        kernel._prefill_fn = prefill_fn
        return kernel

    def _prewarm(self, kernel):
        return kernel.prewarm_extend(
            num_qk_heads=2,
            num_v_heads=4,
            head_k_dim=16,
            head_v_dim=16,
            io_dtype=torch.bfloat16,
            pool_state_dtype=torch.bfloat16,
            checkpoint_every_n_tokens=self.CHECKPOINT_EVERY,
            device=torch.device("cuda"),
        )

    @staticmethod
    def _recorder(reject_cp=False, fail_native=False):
        calls = []

        def prefill(**kwargs):
            use_cp = kwargs["use_cp"]
            calls.append(
                (
                    use_cp,
                    kwargs["cu_seqlens"].numel() - 1,
                    kwargs["checkpoint_every_n_tokens"],
                )
            )
            if use_cp is True and reject_cp:
                raise ValueError("context-parallel prefill is unavailable here")
            if use_cp is False and fail_native:
                raise ValueError("native prefill is unavailable here")
            return kwargs["v"], kwargs["output_state"]

        return prefill, calls

    def _cp_counts(self, counts):
        from sglang.srt.layers.attention.linear.kernels import gdn_flashinfer

        return patch.object(
            gdn_flashinfer,
            "_context_parallel_sequence_counts",
            lambda num_heads, device: list(counts),
        )

    def test_a_rejected_context_parallel_prefill_does_not_fail_startup(self):
        # Serving asks with use_cp="auto", which falls back to the native kernel
        # when FlashInfer cannot run context-parallel; the prewarm forces each
        # kernel, so it must skip a rejected context-parallel one, not raise.
        prefill, calls = self._recorder(reject_cp=True)
        with (
            self._cp_counts([1, 2, 4]),
            self.assertLogs(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer", "WARNING"
            ) as logs,
        ):
            runs = self._prewarm(self._kernel(prefill))
        native = [call for call in calls if call[0] is False]
        self.assertEqual(native, [(False, 1, 0), (False, 1, self.CHECKPOINT_EVERY)])
        # One refused try per checkpoint mode, then that mode's counts are skipped.
        self.assertEqual(
            [call for call in calls if call[0] is True],
            [(True, 1, 0), (True, 1, self.CHECKPOINT_EVERY)],
        )
        self.assertEqual(runs, len(native))
        self.assertEqual(len(logs.records), 2)

    def test_a_native_prefill_failure_still_fails_startup(self):
        prefill, _ = self._recorder(fail_native=True)
        with self._cp_counts([1]), self.assertRaises(ValueError):
            self._prewarm(self._kernel(prefill))

    def test_no_context_parallel_prefill_runs_where_the_policy_never_picks_it(self):
        try:
            from flashinfer.gdn_kernels.delta_rule_dsl import varlen_helper
        except ImportError:
            self.skipTest("FlashInfer's automatic CP policy is not importable")
        prefill, calls = self._recorder()
        with patch.object(
            varlen_helper, "should_use_cp_host", lambda *args, **kwargs: False
        ):
            runs = self._prewarm(self._kernel(prefill))
        self.assertEqual([call for call in calls if call[0] is True], [])
        self.assertEqual(runs, 2)


if __name__ == "__main__":
    unittest.main()
