"""Captured Kimi projections must observe every chunk of a weight reload."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def linear(in_features, out_features):
    return torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)


class TestKimiK3MergedWeightReload(unittest.TestCase):
    def test_merge_refreshes_replaced_parameter_and_padding_in_place(self):
        modules = [linear(8, 5), linear(8, 2)]
        captured, _ = kimi_k3._merge_weights_as_views(modules, pad_rows_to=8)
        pointer = captured.data_ptr()
        modules[1].weight.data = torch.full_like(modules[1].weight, 3)
        captured[-1].fill_(99)
        reloaded, _ = kimi_k3._merge_weights_as_views(
            modules, pad_rows_to=8, merged=captured
        )
        self.assertEqual(reloaded.data_ptr(), pointer)
        torch.testing.assert_close(captured[5:7], torch.full_like(captured[5:7], 3))
        self.assertEqual(captured[-1].count_nonzero().item(), 0)
        self.assertEqual(modules[1].weight.data_ptr(), captured[5:].data_ptr())

    def test_kda_chunked_reload_keeps_captured_projection_current(self):
        owner = kimi_k3.KimiK3DeltaAttention.__new__(kimi_k3.KimiK3DeltaAttention)
        torch.nn.Module.__init__(owner)
        owner.use_full_rank_gate = owner.do_fuse_qkvbfg = True
        owner._bfa_uses_block_fp8 = False
        owner.split_sizes = [24, 8]
        owner.fused_qkvg_proj = linear(16, 32)
        owner.f_a_proj, owner.b_proj = linear(16, 8), linear(16, 4)
        owner.f_b_proj = linear(8, 12)
        owner._bfa_w = owner._bfa_f_b_w = owner._qkvgbfa_layer = None
        with (
            torch.no_grad(),
            patch.object(kimi_k3, "_is_hip", True),
            patch.object(kimi_k3, "_is_npu", False),
            patch.object(
                kimi_k3.envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ, "get", return_value=True
            ),
        ):
            owner._merge_bfa_weights()
            captured = owner._qkvgbfa_layer.weight
            for module in [owner.fused_qkvg_proj, owner.f_a_proj, owner.b_proj]:
                module.weight.zero_()
            owner.f_a_proj.weight.fill_(1)
            owner._merge_bfa_weights()
            owner.fused_qkvg_proj.weight.fill_(2)
            owner.b_proj.weight.fill_(3)
            owner._merge_bfa_weights()
        self.assertEqual(captured.data_ptr(), owner._qkvgbfa_layer.weight.data_ptr())
        for section, value in [
            (captured[:32], 2),
            (captured[32:40], 1),
            (captured[40:44], 3),
        ]:
            torch.testing.assert_close(section, torch.full_like(section, value))
        self.assertIn("f_b_proj.weight", dict(owner.named_parameters()))

    def test_moe_chunked_reload_keeps_captured_front_current(self):
        owner = kimi_k3.KimiK3MoE.__new__(kimi_k3.KimiK3MoE)
        torch.nn.Module.__init__(owner)
        owner.use_latent_moe = True
        owner.shared_experts = torch.nn.Module()
        owner.shared_experts.gate_up_proj = linear(8, 12)
        owner.gate, owner.routed_expert_down_proj = linear(8, 4), linear(8, 6)
        owner._front_w = None
        with (
            torch.no_grad(),
            patch.object(kimi_k3, "_is_npu", False),
            patch.object(
                kimi_k3,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
        ):
            owner._merge_front_weights()
            captured = owner._front_w
            for module in [
                owner.shared_experts.gate_up_proj,
                owner.gate,
                owner.routed_expert_down_proj,
            ]:
                module.weight.zero_()
            owner.gate.weight.fill_(1)
            owner._merge_front_weights()
            owner.shared_experts.gate_up_proj.weight.fill_(2)
            owner.routed_expert_down_proj.weight.fill_(3)
            owner._merge_front_weights()
        self.assertEqual(captured.data_ptr(), owner._front_w.data_ptr())
        for section, value in [
            (captured[:12], 2),
            (captured[12:16], 1),
            (captured[16:], 3),
        ]:
            torch.testing.assert_close(section, torch.full_like(section, value))

    def test_split_bfa_reload_preserves_buffer_and_padding(self):
        owner = kimi_k3.KimiK3DeltaAttention.__new__(kimi_k3.KimiK3DeltaAttention)
        torch.nn.Module.__init__(owner)
        owner.use_full_rank_gate = True
        owner._bfa_uses_block_fp8 = False
        owner.f_a_proj, owner.b_proj = linear(16, 8), linear(16, 4)
        owner.f_b_proj = linear(8, 12)
        owner._bfa_w = owner._bfa_f_b_w = None
        with (
            torch.no_grad(),
            patch.object(kimi_k3, "_is_hip", False),
            patch.object(kimi_k3, "_is_npu", False),
        ):
            owner._merge_bfa_weights()
            captured = owner._bfa_w
            owner.f_a_proj.weight.zero_()
            owner.b_proj.weight.zero_()
            owner.f_a_proj.weight.fill_(2)
            owner._merge_bfa_weights()
            owner.b_proj.weight.fill_(3)
            owner._merge_bfa_weights()
        self.assertEqual(captured.data_ptr(), owner._bfa_w.data_ptr())
        torch.testing.assert_close(captured[:8], torch.full_like(captured[:8], 2))
        torch.testing.assert_close(captured[8:12], torch.full_like(captured[8:12], 3))
        self.assertEqual(captured[12:].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
