import torch

from sglang.srt.layers.moe.topk import biased_topk_impl as native_biased_topk
from sglang.srt.layers.moe.topk import fused_topk_torch_native as native_fused_topk
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase


class TestSqrtSoftplusTopK(CustomTestCase):
    def test_sqrtsoftplus_topk_scoring_func(self):
        hidden_states = torch.ones((1, 4), device=get_device())
        gating_output = torch.tensor([[10.0, 9.0, 1.0, 0.0]], device=get_device())
        correction_bias = torch.zeros(4, device=get_device())

        _, ref_topk_ids = native_biased_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            correction_bias=correction_bias,
            topk=2,
            renormalize=False,
            scoring_func="sqrtsoftplus",
        )
        ref_topk_ids = ref_topk_ids.to(dtype=torch.int64)

        _, topk_ids = native_fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            correction_bias=correction_bias,
            topk=2,
            renormalize=False,
            scoring_func="sqrtsoftplus",
        )

        # ref_topk_ids = torch.tensor([[0, 1]], device=get_device(), dtype=torch.int64)

        torch.testing.assert_close(topk_ids, ref_topk_ids)
