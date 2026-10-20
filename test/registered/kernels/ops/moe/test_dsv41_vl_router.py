"""DSv4.1 VL routing: the fused Triton router must match the eager fallback."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.multimodal.dsv41 import vl_routing
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: ROCm takes the same fused path, compiled by the HIP Triton backend.
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
NUM_EXPERTS = 384
TOP_K = 6
IMAGE_TOKEN_ID = 99


def _moe(num_fused_shared, apply_scale_on_output):
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    bias, bias_vl = (
        (torch.randn(NUM_EXPERTS, generator=gen, device=DEVICE) * 0.1).bfloat16()
        for _ in range(2)
    )
    return SimpleNamespace(
        config=SimpleNamespace(image_token_id=IMAGE_TOKEN_ID),
        gate=SimpleNamespace(
            e_score_correction_bias=bias, e_score_correction_bias_vl=bias_vl
        ),
        topk=SimpleNamespace(
            topk_config=TopKConfig(
                top_k=TOP_K + num_fused_shared,
                num_fused_shared_experts=num_fused_shared,
                renormalize=True,
                routed_scaling_factor=1.5,
                apply_routed_scaling_factor_on_output=apply_scale_on_output,
            )
        ),
        layer_id=0,
    )


def _dense(output, num_columns, num_valid):
    """Scatter routing to [rows, experts] so slot order and tie order don't matter."""
    ids = output.topk_ids[:num_valid].long()
    dense = torch.zeros((num_valid, num_columns), device=DEVICE)
    return dense.scatter_(1, ids, output.topk_weights[:num_valid].float())


class TestDsv41VlRouter(CustomTestCase):
    def test_fused_router_matches_eager(self):
        """Image rows use the VL bias and padded rows stay empty on both paths."""
        for num_tokens in (7, 1024):
            for num_fused_shared in (0, 1):
                for apply_scale_on_output in (False, True):
                    with self.subTest(
                        num_tokens=num_tokens,
                        num_fused_shared=num_fused_shared,
                        apply_scale_on_output=apply_scale_on_output,
                    ):
                        self._check(num_tokens, num_fused_shared, apply_scale_on_output)

    def _check(self, num_tokens, num_fused_shared, apply_scale_on_output):
        gen = torch.Generator(device=DEVICE).manual_seed(num_tokens)
        moe = _moe(num_fused_shared, apply_scale_on_output)
        logits = torch.randn(num_tokens, NUM_EXPERTS, generator=gen, device=DEVICE) * 2
        is_image = torch.rand(num_tokens, generator=gen, device=DEVICE) < 0.7
        input_ids = torch.where(is_image, IMAGE_TOKEN_ID, 1000)
        num_valid = num_tokens - 3
        num_token_non_padded = torch.tensor(num_valid, dtype=torch.int32, device=DEVICE)

        def route():
            return vl_routing.vision_topk(
                moe, logits, input_ids, num_token_non_padded=num_token_non_padded
            )

        fused = route()
        with patch.object(vl_routing, "is_cuda_alike", return_value=False):
            eager = route()

        num_columns = NUM_EXPERTS + num_fused_shared
        torch.testing.assert_close(
            _dense(fused, num_columns, num_valid),
            _dense(eager, num_columns, num_valid),
            atol=1e-5,
            rtol=1e-5,
        )
        for output in (fused, eager):
            self.assertTrue((output.topk_ids[num_valid:] == -1).all())
            self.assertTrue((output.topk_weights[num_valid:] == 0).all())


if __name__ == "__main__":
    unittest.main()
