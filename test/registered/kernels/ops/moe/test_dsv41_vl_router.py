"""DSv4.1 VL routing: the fused Triton router must match the eager fallback."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe.topk import TopKConfig
from sglang.srt.multimodal.dsv41 import vl_routing
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: ROCm takes the same fused path, compiled by the HIP Triton backend.
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EXPERTS = 384
TOP_K = 6
IMAGE_TOKEN_ID = 99
NUM_TOKENS = 64
NUM_VALID = NUM_TOKENS - 3


def _route(num_fused_shared=0):
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    bias, bias_vl = (
        (torch.randn(NUM_EXPERTS, generator=gen, device=DEVICE) * 0.1).bfloat16()
        for _ in range(2)
    )
    moe = SimpleNamespace(
        config=SimpleNamespace(image_token_id=IMAGE_TOKEN_ID),
        gate=SimpleNamespace(
            e_score_correction_bias=bias, e_score_correction_bias_vl=bias_vl
        ),
        topk=SimpleNamespace(
            topk_config=TopKConfig(
                top_k=TOP_K + num_fused_shared,
                shared_experts=SharedExpertsConfig(
                    num_fused_shared_experts=num_fused_shared
                ),
                renormalize=True,
                routed_scaling_factor=1.5,
            ),
            # No EP dispatch and no bound token count outside a forward pass.
            routing_inputs=lambda: (None, None),
        ),
        layer_id=0,
    )
    logits = torch.randn(NUM_TOKENS, NUM_EXPERTS, generator=gen, device=DEVICE) * 2
    is_image = torch.rand(NUM_TOKENS, generator=gen, device=DEVICE) < 0.7
    input_ids = torch.where(is_image, IMAGE_TOKEN_ID, 1000)
    num_token_non_padded = torch.tensor(NUM_VALID, dtype=torch.int32, device=DEVICE)
    return vl_routing.vision_topk(
        moe, logits, input_ids, num_token_non_padded=num_token_non_padded
    )


def _dense(output):
    """Scatter routing to [rows, experts] so slot order and tie order don't matter."""
    ids = output.topk_ids[:NUM_VALID].long()
    dense = torch.zeros((NUM_VALID, NUM_EXPERTS), device=DEVICE)
    return dense.scatter_(1, ids, output.topk_weights[:NUM_VALID].float())


class TestDsv41VlRouter(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
    def test_fused_router_matches_eager(self):
        fused = _route()
        with patch.object(vl_routing, "is_cuda_alike", return_value=False):
            eager = _route()
        torch.testing.assert_close(_dense(fused), _dense(eager), atol=1e-5, rtol=1e-5)
        # Padded rows: no weight, and id 0 on ROCm (AITER cannot index with -1).
        padded_id = 0 if torch.version.hip else -1
        for output in (fused, eager):
            self.assertTrue((output.topk_ids[NUM_VALID:] == padded_id).all())
            self.assertTrue((output.topk_weights[NUM_VALID:] == 0).all())

    def test_eager_padded_ids_stay_zero_on_rocm(self):
        with (
            patch.object(vl_routing, "is_cuda_alike", return_value=False),
            patch.object(vl_routing, "_is_hip", True),
        ):
            out = _route()
        self.assertTrue((out.topk_ids[NUM_VALID:] == 0).all())
        self.assertTrue((out.topk_weights[NUM_VALID:] == 0).all())


if __name__ == "__main__":
    unittest.main()
