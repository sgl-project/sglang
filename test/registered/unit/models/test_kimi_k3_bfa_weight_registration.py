"""Fused KDA weight views must preserve checkpoint parameter names."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _make_attention():
    owner = kimi_k3.KimiK3DeltaAttention.__new__(kimi_k3.KimiK3DeltaAttention)
    torch.nn.Module.__init__(owner)
    owner.use_full_rank_gate = True
    owner.do_fuse_qkvbfg = True
    owner._bfa_uses_block_fp8 = False
    owner.split_sizes = [24, 8]
    owner.fused_qkvg_proj = torch.nn.Linear(16, 32, bias=False)
    owner.f_a_proj = torch.nn.Linear(16, 8, bias=False)
    owner.b_proj = torch.nn.Linear(16, 4, bias=False)
    owner.f_b_proj = torch.nn.Linear(8, 12, bias=False)
    owner._bfa_f_b_w = None
    return owner


class TestKimiK3BfaWeightRegistration(unittest.TestCase):
    def test_rocm_fused_view_preserves_reloadable_parameter(self):
        owner = _make_attention()
        with (
            patch.object(kimi_k3, "_is_hip", True),
            patch.object(kimi_k3, "_is_npu", False),
            patch.object(
                kimi_k3.envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ, "get", return_value=True
            ),
        ):
            # Weight loading can rebuild these views more than once.
            owner._merge_bfa_weights()
            owner._merge_bfa_weights()

        parameters = dict(owner.named_parameters())
        self.assertIn("f_b_proj.weight", parameters)
        self.assertNotIn("_bfa_f_b_w", owner.state_dict())
        self.assertEqual(owner._bfa_f_b_w.data_ptr(), owner.f_b_proj.weight.data_ptr())

        replacement = torch.full_like(owner.f_b_proj.weight, 0.25)
        with torch.no_grad():
            parameters["f_b_proj.weight"].copy_(replacement)
        x = torch.ones(2, 8)
        torch.testing.assert_close(
            torch.nn.functional.linear(x, owner._bfa_f_b_w),
            torch.nn.functional.linear(x, replacement),
        )


if __name__ == "__main__":
    unittest.main()
