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
    SarvamMoEMLADecoderLayer,
    SarvamMoESparseMoeBlock,
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
        self.assertEqual(block.gate.weight.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
