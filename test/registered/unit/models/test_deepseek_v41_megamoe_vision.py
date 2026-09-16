"""V4.1 VL routing on a PD decode worker using MegaMoE."""

import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import mega_moe
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.models import deepseek_v4

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestV41MegaMoEVision(CustomTestCase):
    def test_vision_a2a_is_limited_to_pd_decode_megamoe(self):
        for backend, role, expected in (
            (MoeA2ABackend.NONE, "prefill", True),
            (MoeA2ABackend.NONE, "decode", True),
            (MoeA2ABackend.MEGAMOE, "prefill", False),
            (MoeA2ABackend.MEGAMOE, "decode", True),
            (MoeA2ABackend.DEEPEP, "decode", False),
        ):
            with (
                self.subTest(backend=backend, role=role),
                patch.object(deepseek_v4, "get_moe_a2a_backend", return_value=backend),
                patch.object(
                    deepseek_v4,
                    "get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode=role),
                ),
            ):
                self.assertEqual(deepseek_v4._v41_vision_a2a_supported(), expected)

    def test_decode_does_not_reprocess_prefill_images(self):
        model = SimpleNamespace(
            vision=object(),
            config=SimpleNamespace(image_token_id=99),
            _prepare_mm_embeddings=Mock(),
        )
        mode = SimpleNamespace(
            is_decode=lambda: True,
            is_target_verify=lambda: False,
            is_decode_or_idle=lambda: True,
        )
        ids = torch.tensor([7, 8])
        batch = SimpleNamespace(forward_mode=mode, mm_inputs=[object()])
        model_ids, embeds = (
            deepseek_v4.DeepseekV4ForCausalLM.prepare_language_model_inputs(
                model, ids, batch
            )
        )
        self.assertIs(model_ids, ids)
        self.assertIsNone(embeds)
        model._prepare_mm_embeddings.assert_not_called()

    def test_megamoe_routes_text_and_image_rows_with_separate_biases(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.mega_moe_pre_dispatch = Mock()
        deep_gemm.fp8_fp4_mega_moe = Mock()
        buffer = SimpleNamespace(
            x=object(), x_sf=object(), topk_idx=object(), topk_weights=object()
        )
        gate = Mock(return_value=torch.zeros((3, 3)))
        gate.e_score_correction_bias = torch.tensor([3.0, 0.0, 0.0])
        gate.e_score_correction_bias_vl = torch.tensor([0.0, 3.0, 0.0])
        topk = Mock()
        topk.topk_config = SimpleNamespace(
            top_k=1,
            num_fused_shared_experts=0,
            fused_shared_experts_scaling_factor=None,
            renormalize=False,
            routed_scaling_factor=1.0,
            apply_routed_scaling_factor_on_output=True,
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                moe_intermediate_size=8,
                num_experts_per_tok=1,
                image_token_id=99,
            ),
            experts=SimpleNamespace(
                num_experts=3,
                mega_l1_weights=object(),
                mega_l2_weights=object(),
                should_fuse_routed_scaling_factor_in_topk=True,
            ),
            gate=gate,
            topk=topk,
            is_hash=False,
            num_fused_shared_experts=0,
            layer_id=0,
        )

        from sglang.srt.multimodal.dsv41 import vl_routing

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            patch.object(mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer),
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            patch.object(vl_routing, "is_cuda", return_value=False),
            patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=SimpleNamespace(device_group=object()),
            ),
        ):
            for ids, expected in (
                ([7, 8, 9], [0, 0, 0]),
                ([7, 99, 8], [0, 1, 0]),
            ):
                with self.subTest(ids=ids):
                    mega_moe._run_mega_routed(
                        moe,
                        torch.zeros((3, 4)),
                        forward_batch=None,
                        input_ids_global=torch.tensor(ids),
                        num_tokens=3,
                    )
                    routed_ids = deep_gemm.mega_moe_pre_dispatch.call_args.args[1]
                    self.assertEqual(routed_ids[:, 0].tolist(), expected)
            topk.assert_not_called()
            gate.reset_mock()
            mega_moe._run_mega_routed(
                moe,
                torch.empty((0, 4)),
                forward_batch=None,
                input_ids_global=torch.empty((0,), dtype=torch.int64),
                num_tokens=0,
            )
            gate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
