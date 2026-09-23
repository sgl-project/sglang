import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.sarvam_moe import (
    AttnForwardMethod,
    SarvamMLAForCausalLM,
    SarvamMLAModel,
    SarvamMoEMLAAttention,
    SarvamMoEMLADecoderLayer,
    SarvamMoESparseMoeBlock,
    _trtllm_bypass_torch_compile_forward,
    get_attn_forward_method,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CaptureLayer(nn.Module):
    def forward(
        self,
        _positions,
        hidden_states,
        _forward_batch,
        residual,
        captured_last_layer_outputs=None,
    ):
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.append(
                hidden_states if residual is None else hidden_states + residual
            )
        return hidden_states + 1, residual


class TestSarvamDSpark(CustomTestCase):
    @staticmethod
    def _make_target(*, pp_size=1, is_last_rank=True, num_layers=32):
        target = SarvamMLAForCausalLM.__new__(SarvamMLAForCausalLM)
        nn.Module.__init__(target)
        target.pp_group = SimpleNamespace(world_size=pp_size, is_last_rank=is_last_rank)
        target.config = SimpleNamespace(num_hidden_layers=num_layers)
        target.model = nn.Module()
        target.model.dspark_layers_to_capture = None
        target.capture_aux_hidden_states = False
        return target

    def test_config_remap_treats_null_routing_values_as_missing(self):
        config = SimpleNamespace(
            n_group=None,
            topk_group=None,
            router_dtype=None,
        )

        SarvamMLAForCausalLM._remap_config(config)

        self.assertEqual(config.n_group, 1)
        self.assertEqual(config.topk_group, 1)
        self.assertEqual(config.router_dtype, "bf16_fp32")

    def test_config_remap_preserves_explicit_routing_values(self):
        config = SimpleNamespace(
            n_group=8,
            topk_group=4,
            router_dtype="fp32",
        )

        SarvamMLAForCausalLM._remap_config(config)

        self.assertEqual(config.n_group, 8)
        self.assertEqual(config.topk_group, 4)
        self.assertEqual(config.router_dtype, "fp32")

    def test_capture_hook_uses_raw_target_layer_ids(self):
        target = self._make_target()

        target.set_dspark_layers_to_capture([3, 9, 15, 21, 27])

        self.assertTrue(target.capture_aux_hidden_states)
        self.assertEqual(target.model.dspark_layers_to_capture, [3, 9, 15, 21, 27])

    def test_capture_hook_rejects_invalid_contracts(self):
        target = self._make_target()
        for layer_ids in (None, [], [3, 3], [9, 3], [-1, 3], [3, 32]):
            with self.subTest(layer_ids=layer_ids), self.assertRaises(ValueError):
                target.set_dspark_layers_to_capture(layer_ids)

        with self.assertRaises(NotImplementedError):
            self._make_target(pp_size=2).set_dspark_layers_to_capture([3, 9])

    def test_model_captures_packed_post_layer_outputs_without_id_offset(self):
        model = SarvamMLAModel.__new__(SarvamMLAModel)
        nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.start_layer = 0
        model.end_layer = 4
        model.layers = nn.ModuleList([_CaptureLayer() for _ in range(4)])
        model.norm = nn.Identity()
        model.dspark_layers_to_capture = [0, 2, 3]
        input_embeds = torch.zeros(2, 4)

        final_hidden, captured = model(
            input_ids=None,
            positions=torch.arange(2),
            forward_batch=object(),
            input_embeds=input_embeds,
        )

        torch.testing.assert_close(final_hidden, torch.full((2, 4), 4.0))
        self.assertEqual(captured.shape, (2, 12))
        for value, expected in zip(captured.chunk(3, dim=-1), (1.0, 3.0, 4.0)):
            torch.testing.assert_close(value, torch.full((2, 4), expected))

    def test_decoder_delegates_capture_to_layer_communicator(self):
        layer = SarvamMoEMLADecoderLayer.__new__(SarvamMoEMLADecoderLayer)
        nn.Module.__init__(layer)
        layer.self_attn = Mock(side_effect=lambda **kwargs: kwargs["hidden_states"])
        layer.mlp = Mock(side_effect=lambda hidden, _batch: hidden)
        layer.is_layer_sparse = True
        layer.attn_tp_size = 1
        layer.layer_communicator = Mock()
        capture = []
        prepare_and_capture = (
            layer.layer_communicator.prepare_attn_and_capture_last_layer_outputs
        )
        prepare_and_capture.side_effect = lambda h, r, _b, captured_last_layer_outputs: (
            h,
            r,
        )
        layer.layer_communicator.prepare_mlp.side_effect = lambda h, r, _b: (h, r)
        layer.layer_communicator.should_fuse_mlp_allreduce_with_next_layer.return_value = False
        layer.layer_communicator.should_use_reduce_scatter.return_value = False
        layer.layer_communicator.postprocess_layer.side_effect = lambda h, r, _b: (
            h,
            r,
        )
        runtime = SimpleNamespace(scoped=Mock(return_value=nullcontext()))
        hidden = torch.ones(2, 4)
        residual = torch.full((2, 4), 2.0)

        with patch("sglang.srt.models.sarvam_moe.get_forward", return_value=runtime):
            layer(
                hidden.new_zeros(2, dtype=torch.long),
                hidden,
                object(),
                residual,
                captured_last_layer_outputs=capture,
            )

        self.assertIs(
            prepare_and_capture.call_args.kwargs["captured_last_layer_outputs"],
            capture,
        )

    def test_target_verify_uses_decode_backend_when_requested(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        with (
            patch(
                "sglang.srt.models.sarvam_moe.attention_backends",
                return_value=("triton", "trtllm_mla"),
            ),
            patch(
                "sglang.srt.models.sarvam_moe.get_spec",
                return_value=SimpleNamespace(speculative_attention_mode="decode"),
            ),
        ):
            self.assertEqual(
                get_attn_forward_method(forward_batch),
                AttnForwardMethod.MLA_SEPARATE_ROPE,
            )

    def test_target_verify_uses_prefill_backend_when_requested(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)

        with (
            patch(
                "sglang.srt.models.sarvam_moe.attention_backends",
                return_value=("triton", "trtllm_mla"),
            ),
            patch(
                "sglang.srt.models.sarvam_moe.get_spec",
                return_value=SimpleNamespace(speculative_attention_mode="prefill"),
            ),
        ):
            self.assertEqual(
                get_attn_forward_method(forward_batch),
                AttnForwardMethod.MLA_CONCAT_ROPE,
            )

    def test_fa4_prefill_uses_separate_rope(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

        with (
            patch(
                "sglang.srt.models.sarvam_moe.attention_backends",
                return_value=("fa4", "trtllm_mla"),
            ),
            patch(
                "sglang.srt.models.sarvam_moe.get_platform",
                return_value=SimpleNamespace(is_sm100_or_sm110=True),
            ),
        ):
            self.assertEqual(
                get_attn_forward_method(forward_batch),
                AttnForwardMethod.MLA_SEPARATE_ROPE,
            )

    def test_fa4_preserves_legacy_rope_path_on_older_gpus(self):
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

        with (
            patch(
                "sglang.srt.models.sarvam_moe.attention_backends",
                return_value=("fa4", "trtllm_mla"),
            ),
            patch(
                "sglang.srt.models.sarvam_moe.get_platform",
                return_value=SimpleNamespace(is_sm100_or_sm110=False),
            ),
        ):
            self.assertEqual(
                get_attn_forward_method(forward_batch),
                AttnForwardMethod.MLA_CONCAT_ROPE,
            )

    def test_attention_value_projection_writes_flattened_layout_directly(self):
        attention = SarvamMoEMLAAttention.__new__(SarvamMoEMLAAttention)
        nn.Module.__init__(attention)
        attention.num_local_heads = 2
        attention.v_head_dim = 3
        attention.w_vc = nn.Parameter(torch.randn(2, 4, 3), requires_grad=False)
        attn_output = torch.randn(5, 2, 4)
        expected = (
            torch.bmm(attn_output.transpose(0, 1), attention.w_vc)
            .transpose(0, 1)
            .flatten(1, 2)
        )

        with (
            patch(
                "sglang.srt.models.sarvam_moe.is_in_tc_piecewise_cuda_graph",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.sarvam_moe.get_platform",
                return_value=SimpleNamespace(is_sm100_or_sm110=True),
            ),
        ):
            actual = attention._project_attention_output(attn_output)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (5, 6))

    def test_attention_value_projection_uses_legacy_path_on_older_gpus(self):
        attention = SarvamMoEMLAAttention.__new__(SarvamMoEMLAAttention)
        nn.Module.__init__(attention)
        attention.num_local_heads = 2
        attention.v_head_dim = 3
        attention.w_vc = nn.Parameter(torch.randn(2, 4, 3), requires_grad=False)
        attn_output = torch.randn(5, 2, 4)

        with (
            patch.object(
                attention,
                "_maybe_fp8_bmm",
                wraps=attention._maybe_fp8_bmm,
            ) as legacy_bmm,
            patch(
                "sglang.srt.models.sarvam_moe.get_platform",
                return_value=SimpleNamespace(is_sm100_or_sm110=False),
            ),
        ):
            actual = attention._project_attention_output(attn_output)

        self.assertEqual(legacy_bmm.call_count, 1)
        self.assertEqual(actual.shape, (5, 6))

    def test_flashinfer_bypass_keeps_piecewise_graph_wrapper(self):
        block = SarvamMoESparseMoeBlock.__new__(SarvamMoESparseMoeBlock)
        nn.Module.__init__(block)
        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 8)
        block.use_flashinfer_trtllm_bypass = True
        block.topk = SimpleNamespace(topk_config=object())
        block.experts = nn.Module()
        block.experts.forward = Mock(return_value=hidden_states)
        block.experts.forward_impl = Mock(return_value=hidden_states)
        block._router_logits = Mock(return_value=router_logits)

        with patch(
            "sglang.srt.models.sarvam_moe.is_in_tc_piecewise_cuda_graph",
            return_value=True,
        ):
            actual = block._forward_router_experts(hidden_states)

        self.assertIs(actual, hidden_states)
        block.experts.forward.assert_called_once()
        block.experts.forward_impl.assert_not_called()

    def test_sparse_moe_declares_deepseek_v3_routing_contract(self):
        config = SimpleNamespace(
            hidden_size=16,
            hidden_act="silu",
            moe_intermediate_size=32,
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=0,
        )
        captured_expert_kwargs = {}

        def make_experts(**kwargs):
            captured_expert_kwargs.update(kwargs)
            return nn.Identity()

        runtime = SimpleNamespace(moe=SimpleNamespace(ep_num_redundant_experts=0))
        with (
            patch(
                "sglang.srt.models.sarvam_moe.get_parallel",
                return_value=SimpleNamespace(tp_size=1),
            ),
            patch("sglang.srt.models.sarvam_moe.get_exec", return_value=runtime),
            patch(
                "sglang.srt.models.sarvam_moe.get_moe_runner_backend",
                return_value=SimpleNamespace(
                    is_flashinfer_trtllm=lambda: True,
                    is_flashinfer_trtllm_routed=lambda: False,
                ),
            ),
            patch(
                "sglang.srt.models.sarvam_moe.get_moe_impl_class",
                return_value=make_experts,
            ),
            patch(
                "sglang.srt.models.sarvam_moe.TopK",
                side_effect=lambda **_kwargs: nn.Identity(),
            ),
        ):
            block = SarvamMoESparseMoeBlock(config, layer_id=1)

        self.assertEqual(
            captured_expert_kwargs["routing_method_type"],
            RoutingMethodType.DeepSeekV3,
        )
        self.assertEqual(captured_expert_kwargs["routed_scaling_factor"], 2.5)
        self.assertTrue(block.fuse_routed_scaling_in_moe)
        self.assertEqual(block.gate.weight.dtype, torch.bfloat16)
        self.assertTrue(block.router_logits_fp32)

    def test_flashinfer_bypass_disables_bs1_compile_swap(self):
        config = SimpleNamespace(
            hidden_size=16,
            hidden_act="silu",
            moe_intermediate_size=32,
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=0,
        )
        runtime = SimpleNamespace(moe=SimpleNamespace(ep_num_redundant_experts=0))

        class FakeExperts(nn.Module):
            # Mirrors FusedMoE: the layer owns the quant method that the
            # fused-op compile protocol toggles.
            def __init__(self, **_kwargs):
                super().__init__()
                self.quant_method = SimpleNamespace(
                    _torch_compile_forward=lambda num_tokens: "native-swap"
                )

        for trtllm in (True, False):
            with (
                patch(
                    "sglang.srt.models.sarvam_moe.get_parallel",
                    return_value=SimpleNamespace(tp_size=1),
                ),
                patch(
                    "sglang.srt.models.sarvam_moe.get_exec",
                    return_value=runtime,
                ),
                patch(
                    "sglang.srt.models.sarvam_moe.get_moe_runner_backend",
                    return_value=SimpleNamespace(
                        is_flashinfer_trtllm=lambda trtllm=trtllm: trtllm,
                        is_flashinfer_trtllm_routed=lambda: False,
                    ),
                ),
                patch(
                    "sglang.srt.models.sarvam_moe.get_moe_impl_class",
                    return_value=FakeExperts,
                ),
                patch(
                    "sglang.srt.models.sarvam_moe.TopK",
                    side_effect=lambda **_kwargs: nn.Identity(),
                ),
            ):
                block = SarvamMoESparseMoeBlock(config, layer_id=1)

            self.assertEqual(block.use_flashinfer_trtllm_bypass, trtllm)
            hook = block.experts.quant_method._torch_compile_forward
            if trtllm:
                self.assertIs(hook, _trtllm_bypass_torch_compile_forward)
                self.assertIsNone(hook(num_tokens=1))
                self.assertIsNone(hook(num_tokens=4))
            else:
                self.assertEqual(hook(num_tokens=1), "native-swap")


if __name__ == "__main__":
    unittest.main()
