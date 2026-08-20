import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.utils import is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    _cache_indices,
    _pure_torch_gdn_reference,
    _ssm_states,
    build_gdn_attention_fixture,
    make_gdn_cases,
    run_gdn_attention_case,
    run_gdn_fixture_eager,
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

register_cuda_ci(est_time=20, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

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
    CAKE_DECODE_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_decode_b4_t1",
        backend="flashinfer",
        forward_mode=ForwardMode.DECODE,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4, 7, 10, 13),
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )
    CAKE_PREFILL_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_prefill_b5_s64",
        backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4, 7, 10, 13, 16),
        extend_lens=(64,) * 5,
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )
    CAKE_CP_PREFILL_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_cp_prefill_b1_s128",
        backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4,),
        extend_lens=(128,),
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )
    CAKE_VERIFY_CASE = GDNAttentionCase(
        name="flashinfer_cake_gdn_tp4_verify_b8_t4",
        backend="flashinfer",
        forward_mode=ForwardMode.TARGET_VERIFY,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(4, 5, 6, 7, 8, 9, 10, 11),
        extend_lens=(4,) * 8,
        linear_attn_decode_backend="flashinfer",
        linear_attn_prefill_backend="flashinfer",
    )

    def _cake_api_or_skip(self):
        try:
            from flashinfer.jit import cake_gdn_noncp_decode
        except ImportError:
            self.skipTest("public FlashInfer Cake GDN loader is unavailable")
        return cake_gdn_noncp_decode

    def _cake_cp_api_or_skip(self):
        try:
            from flashinfer.jit import cake_gdn_cp_prefill
        except ImportError:
            self.skipTest("public FlashInfer Cake GDN CP loader is unavailable")
        return cake_gdn_cp_prefill

    def test_cake_exact_decode_eager_and_cuda_graph(self):
        cake_api = self._cake_api_or_skip()
        with mock.patch.object(
            cake_api,
            "load_cake_gdn_kernel",
            wraps=cake_api.load_cake_gdn_kernel,
        ) as load_kernel:
            run_gdn_attention_case(
                self,
                self.CAKE_DECODE_CASE,
                head_k_dim=self.HEAD_DIM,
                head_v_dim=self.HEAD_DIM,
            )
            eager_load_count = load_kernel.call_count
            self.assertGreater(eager_load_count, 0)
            run_gdn_cuda_graph_decode_case(
                self,
                self.CAKE_DECODE_CASE,
                head_k_dim=self.HEAD_DIM,
                head_v_dim=self.HEAD_DIM,
                cuda_graph_capture_batch_size=4,
            )
        self.assertGreater(load_kernel.call_count, eager_load_count)

    def test_cake_exact_prefill_updates_indexed_state_in_place(self):
        cake_api = self._cake_api_or_skip()
        fixture = build_gdn_attention_fixture(
            self,
            self.CAKE_PREFILL_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=128,
        )
        initial_ssm_states = _ssm_states(fixture).clone()
        with mock.patch.object(
            cake_api,
            "load_cake_gdn_kernel",
            wraps=cake_api.load_cake_gdn_kernel,
        ) as load_kernel:
            dispatcher = fixture.backend.linear_attn_backend.kernel_dispatcher
            with mock.patch.object(
                dispatcher,
                "extend",
                wraps=dispatcher.extend,
            ) as extend:
                actual = run_gdn_fixture_eager(fixture)
        expected = _pure_torch_gdn_reference(fixture, initial_ssm_states)
        cache_indices = _cache_indices(fixture)

        self.assertGreater(load_kernel.call_count, 0)
        extend.assert_called_once()
        self.assertIs(
            extend.call_args.kwargs["seq_lens_cpu"],
            fixture.forward_batch.extend_seq_lens_cpu,
        )
        self.assertEqual(extend.call_args.kwargs["layer_id"], 0)
        torch.testing.assert_close(
            actual, expected.output, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            _ssm_states(fixture)[cache_indices],
            expected.final_states[cache_indices],
            atol=1e-2,
            rtol=1e-2,
        )

        # Prepare all stream-local Cake buffers before capture, then prove that
        # the same admitted non-CP route captures and replays on a caller stream.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        _ssm_states(fixture).copy_(initial_ssm_states)
        with (
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            fixture.backend.init_forward_metadata(fixture.forward_batch)
            fixture.actual_module(
                fixture.forward_batch,
                fixture.mixed_qkv,
                fixture.a,
                fixture.b,
            )
        capture_stream.synchronize()

        _ssm_states(fixture).copy_(initial_ssm_states)
        capture_stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with (
            torch.no_grad(),
            torch.cuda.stream(capture_stream),
            forward_context(ForwardContext(attn_backend=fixture.backend)),
        ):
            with torch.cuda.graph(graph, stream=capture_stream):
                graph_output = fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )
        capture_stream.synchronize()

        _ssm_states(fixture).copy_(initial_ssm_states)
        torch.cuda.current_stream().synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            graph_output, expected.output, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            _ssm_states(fixture)[cache_indices],
            expected.final_states[cache_indices],
            atol=1e-2,
            rtol=1e-2,
        )

    def test_public_auto_cp_prefill_is_not_intercepted(self):
        cake_api = self._cake_api_or_skip()
        cake_cp_api = self._cake_cp_api_or_skip()
        from flashinfer.gdn_kernels.blackwell import cake_gdn_cp_prefill

        fixture = build_gdn_attention_fixture(
            self,
            self.CAKE_CP_PREFILL_CASE,
            head_k_dim=self.HEAD_DIM,
            head_v_dim=self.HEAD_DIM,
            max_context_len=256,
        )
        initial_ssm_states = _ssm_states(fixture).clone()
        with (
            mock.patch.object(
                cake_api,
                "select_cake_gdn_prefill_variant",
                wraps=cake_api.select_cake_gdn_prefill_variant,
            ) as noncp_selector,
            mock.patch.object(
                cake_api,
                "load_cake_gdn_kernel",
                wraps=cake_api.load_cake_gdn_kernel,
            ) as noncp_loader,
            mock.patch.object(
                cake_gdn_cp_prefill,
                "load_cake_gdn_cp_kernel",
                wraps=cake_cp_api.load_cake_gdn_cp_kernel,
            ) as cp_loader,
        ):
            actual = run_gdn_fixture_eager(fixture)
        expected = _pure_torch_gdn_reference(fixture, initial_ssm_states)
        cache_indices = _cache_indices(fixture)

        noncp_selector.assert_not_called()
        noncp_loader.assert_not_called()
        self.assertGreater(cp_loader.call_count, 0)
        torch.testing.assert_close(
            actual, expected.output, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            _ssm_states(fixture)[cache_indices],
            expected.final_states[cache_indices],
            atol=1e-2,
            rtol=1e-2,
        )

    def test_cake_exact_verify_eager_and_cuda_graph(self):
        cake_api = self._cake_api_or_skip()
        with mock.patch.object(
            cake_api,
            "load_cake_gdn_kernel",
            wraps=cake_api.load_cake_gdn_kernel,
        ) as load_kernel:
            run_gdn_eagle_verify_case(
                self,
                self.CAKE_VERIFY_CASE,
                topk=1,
                spec_kind="frozen_kv_mtp",
                head_k_dim=self.HEAD_DIM,
                head_v_dim=self.HEAD_DIM,
            )
            eager_load_count = load_kernel.call_count
            self.assertGreater(eager_load_count, 0)
            run_gdn_eagle_verify_cuda_graph_case(
                self,
                self.CAKE_VERIFY_CASE,
                topk=1,
                spec_kind="frozen_kv_mtp",
                head_k_dim=self.HEAD_DIM,
                head_v_dim=self.HEAD_DIM,
                cuda_graph_capture_batch_size=8,
            )
        self.assertGreater(load_kernel.call_count, eager_load_count)

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


if __name__ == "__main__":
    unittest.main()
