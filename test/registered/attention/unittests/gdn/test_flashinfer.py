import sys
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.utils import is_flashinfer_available
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    build_gdn_attention_fixture,
    make_gdn_cases,
    run_gdn_attention_case,
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

register_cuda_ci(est_time=35, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-large")


@unittest.skipIf(
    not torch.cuda.is_available() or not is_flashinfer_available(),
    "CUDA + flashinfer are required",
)
class TestFlashInferGDNBackendCorrectness(CustomTestCase):
    # FlashInfer SM90 prefill kernels require value head dim in {64, 128, 256}.
    HEAD_K_DIM = 64
    HEAD_V_DIM = 64
    PREFILL_BACKEND_GETTER = (
        "sglang.srt.layers.attention.linear.gdn_backend."
        "get_linear_attn_prefill_backend"
    )
    GDN_POOL_CLASS = (
        "sglang.test.kits.attention_unittest.attention_methods.gdn_attention."
        "HybridReqToTokenPool"
    )
    PREFILL_CASE = GDNAttentionCase(
        name="flashinfer_prefill_attention_block",
        backend="flashinfer",
        forward_mode=ForwardMode.EXTEND,
        num_k_heads=4,
        num_v_heads=8,
        page_size=16,
        prefix_lens=(0, 8),
        extend_lens=(17, 9),
    )

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

    def _build_two_layer_flashinfer_prefill_fixture(self):
        layer_ids = [0, 1]

        def build_two_layer_pool(*args, **kwargs):
            kwargs["cache_params"] = replace(kwargs["cache_params"], layers=layer_ids)
            kwargs["mamba_layer_ids"] = layer_ids
            return HybridReqToTokenPool(*args, **kwargs)

        with patch(
            self.PREFILL_BACKEND_GETTER,
            return_value=LinearAttnKernelBackend.FLASHINFER,
        ), patch(self.GDN_POOL_CLASS, side_effect=build_two_layer_pool):
            return build_gdn_attention_fixture(
                self,
                self.PREFILL_CASE,
                head_k_dim=128,
                head_v_dim=128,
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

    def test_flashinfer_prefill_attention_block(self):
        with patch(
            self.PREFILL_BACKEND_GETTER,
            return_value=LinearAttnKernelBackend.FLASHINFER,
        ):
            run_gdn_attention_case(
                self,
                self.PREFILL_CASE,
                head_k_dim=128,
                head_v_dim=128,
            )

    def test_extend_prep_is_built_once_per_forward(self):
        fixture = self._build_two_layer_flashinfer_prefill_fixture()
        second_block = deepcopy(fixture.actual_module)
        second_block.attn.layer_id = 1

        pool = fixture.runner.req_to_token_pool
        first_layer_cache = pool.mamba2_layer_cache(0)
        second_layer_cache = pool.mamba2_layer_cache(1)
        self.assertNotEqual(
            first_layer_cache.temporal.data_ptr(),
            second_layer_cache.temporal.data_ptr(),
        )
        for first_conv, second_conv in zip(
            first_layer_cache.conv, second_layer_cache.conv
        ):
            second_conv.copy_(first_conv)
        second_layer_cache.temporal.copy_(first_layer_cache.temporal)
        initial_temporal = first_layer_cache.temporal.clone()

        kernel = fixture.backend.linear_attn_backend.kernel_dispatcher.extend_kernel

        with patch.object(
            kernel, "build_extend_prep", wraps=kernel.build_extend_prep
        ) as build_prep:
            with torch.no_grad(), forward_context(
                ForwardContext(attn_backend=fixture.backend)
            ):
                fixture.backend.init_forward_metadata(fixture.forward_batch)
                first_output = fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )
                second_output = second_block(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )
                self.assertEqual(build_prep.call_count, 1)
                torch.testing.assert_close(first_output, second_output)
                torch.testing.assert_close(
                    first_layer_cache.temporal, second_layer_cache.temporal
                )
                self.assertFalse(
                    torch.equal(initial_temporal, first_layer_cache.temporal)
                )
                self.assertFalse(
                    torch.equal(initial_temporal, second_layer_cache.temporal)
                )

                fixture.backend.init_forward_metadata(fixture.forward_batch)
                fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )
                self.assertEqual(build_prep.call_count, 2)

    def test_extend_prep_arch_conversions(self):
        kernel = FlashInferGDNKernel()
        cache_indices = torch.tensor([3, -1], device="cuda", dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)

        for use_state_pool, expected_indices, offsets_dtype in (
            (True, [3, 0], torch.int32),
            (False, [3, 7], torch.int64),
        ):
            with self.subTest(use_state_pool=use_state_pool), patch.object(
                kernel, "use_state_pool", use_state_pool
            ):
                prep = kernel.build_extend_prep(
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    state_pool_size=8,
                )
                torch.testing.assert_close(
                    prep.ssm_cache_indices,
                    torch.tensor(expected_indices, device="cuda", dtype=torch.int64),
                )
                self.assertEqual(prep.cu_seqlens.dtype, offsets_dtype)
                torch.testing.assert_close(
                    prep.cu_seqlens, query_start_loc.to(offsets_dtype)
                )
                if use_state_pool:
                    self.assertIs(prep.cu_seqlens, query_start_loc)

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


if __name__ == "__main__":
    unittest.main()
