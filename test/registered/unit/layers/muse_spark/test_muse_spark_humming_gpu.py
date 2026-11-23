"""Muse Spark's NVFP4 experts on Humming W4A16 GEMMs against an FP32 reference."""

import unittest

import torch

from sglang.kernels.ops.moe.moe_align_ep_local import ep_local_align_supported
from sglang.srt.layers.muse_spark.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.layers.muse_spark.muse_spark_humming_moe import (
    HummingTuning,
    prepare_humming_experts,
)
from sglang.srt.layers.muse_spark.muse_spark_layers import (
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import make_nvfp4_weight_and_ref
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Muse Spark's expert sizes, so every spliced H200 tile is legal for these GEMMs.
HIDDEN, INTERMEDIATE, NUM_LOCAL, TOP_K = 4096, 4096, 2, 2


def _nvfp4_experts(n_stacks, size_k, up_gain=1.0):
    """Packed E2M1 weights, E4M3 group scales, FP32 per-stack global scales and
    the dequantized FP32 reference; up_gain scales the last stack's global scale
    by a power of two, which the scale and the reference both carry exactly."""
    weights, scales, global_scales, refs = [], [], [], []
    for _ in range(NUM_LOCAL):
        parts = [make_nvfp4_weight_and_ref(n, size_k, torch.bfloat16) for n in n_stacks]
        gains = [1.0] * (len(parts) - 1) + [up_gain]
        weights.append(torch.cat([part[0] for part in parts]))
        scales.append(torch.cat([part[1] for part in parts]))
        global_scales.append(
            torch.stack([part[2].float() * gain for part, gain in zip(parts, gains)])
        )
        refs.append(
            torch.cat([part[3].float() * gain for part, gain in zip(parts, gains)])
        )
    return (
        torch.stack(weights),
        torch.stack(scales),
        torch.stack(global_scales),
        torch.stack(refs),
    )


def _make_layer(use_expert_map):
    layer = MuseSpark1xFp4Experts.__new__(MuseSpark1xFp4Experts)
    torch.nn.Module.__init__(layer)
    layer.use_humming = True
    layer.params_dtype = torch.bfloat16
    layer.hidden_size = HIDDEN
    layer.intermediate_size_per_partition = INTERMEDIATE
    layer.num_local_experts = NUM_LOCAL
    layer.global_num_experts = 4 if use_expert_map else NUM_LOCAL
    layer._expert_map = (
        torch.tensor([-1, -1, 0, 1], device="cuda", dtype=torch.int32)
        if use_expert_map
        else None
    )
    # Distinct gate and up global scales: Humming keeps the gate's, as Marlin
    # does, and the post-expert norm cancels the resulting constant gain.
    w13 = _nvfp4_experts([INTERMEDIATE, INTERMEDIATE], HIDDEN, up_gain=2.0)
    w2 = _nvfp4_experts([HIDDEN], INTERMEDIATE)
    for native, (packed, scales, global_scales, _) in (
        ("gate_up_proj", w13),
        ("down_proj", w2),
    ):
        if native == "down_proj":
            global_scales = global_scales.squeeze(1)
        for name, tensor in (
            (native, packed),
            (f"{native}_weight_scale", scales),
            (f"{native}_weight_scale_2", global_scales.contiguous()),
        ):
            setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.humming_tuning = prepare_humming_experts(layer)

    norm = MuseSpark1xRMSNormWithInputScale(HIDDEN, eps=1e-6).to(
        device="cuda", dtype=torch.bfloat16
    )
    norm.weight.data.copy_(
        torch.linspace(0.5, 1.5, HIDDEN, device="cuda", dtype=torch.bfloat16)
    )
    layer._post_expert_norm = norm
    layer._post_expert_norm_weight = norm.weight.detach()
    layer._post_expert_norm_eps = norm.eps
    layer._post_expert_norm_weight_first = True
    return layer, norm, w13[3], w2[3]


def _reference(x, local_ids, weights, w13_ref, w2_ref, norm):
    experts = torch.zeros((x.shape[0], TOP_K, HIDDEN), device="cuda")
    for expert in range(NUM_LOCAL):
        rows, routes = (local_ids == expert).nonzero(as_tuple=True)
        gate, up = (x[rows].float() @ w13_ref[expert].T).chunk(2, dim=-1)
        experts[rows, routes] = (torch.nn.functional.silu(gate) * up) @ w2_ref[expert].T
    normed = norm(experts.to(torch.bfloat16)).float()
    live = weights.masked_fill(local_ids < 0, 0.0)
    return (normed * live[..., None]).sum(1)


class TestMuseSparkHummingExperts(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def _check(self, layer, norm, w13_ref, w2_ref, num_tokens):
        """The last token has no local route and must finalize to zeros."""
        torch.manual_seed(num_tokens)
        num_experts = layer.global_num_experts
        ids = torch.stack(
            [
                torch.randperm(num_experts, device="cuda")[:TOP_K]
                for _ in range(num_tokens)
            ]
        ).to(torch.int32)
        ids[-1] = -1
        local_ids = ids.clone()
        if layer._expert_map is not None:
            local_ids = layer._expert_map[ids.clamp_min(0).long()]
            local_ids[ids < 0] = -1
        weights = torch.softmax(torch.randn(num_tokens, TOP_K, device="cuda"), dim=-1)
        weights[local_ids < 0] = float("nan")
        x = torch.randn((num_tokens, HIDDEN), device="cuda", dtype=torch.bfloat16) / 20

        actual = layer._forward_w4a16(x, weights, ids, norm).float()

        expected = _reference(x, local_ids, weights, w13_ref, w2_ref, norm)
        rel_l2 = float((actual - expected).norm() / expected.norm())
        self.assertLess(rel_l2, 2e-2)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(int(torch.count_nonzero(actual[-1])), 0)

    def test_matches_reference(self):
        """On SM90, 4 tokens run the first row's MMA w13 tile, 48 a WGMMA row."""
        for use_expert_map in (False, True):
            layer, norm, w13_ref, w2_ref = _make_layer(use_expert_map)
            for num_tokens in (4, 48):
                with self.subTest(use_expert_map=use_expert_map, num_tokens=num_tokens):
                    self._check(layer, norm, w13_ref, w2_ref, num_tokens)

    def test_gemms_with_different_block_m(self):
        """Each GEMM must read routes aligned to its own tile's block M."""
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("The forced WGMMA tile needs SM90")
        layer, norm, w13_ref, w2_ref = _make_layer(use_expert_map=True)
        tables = {}
        for sublayer, block_shape, warp_shape in (
            ("w13", [8, 256, 128], [8, 64, 64]),
            ("w2", [24, 128, 256], [24, 32, 64]),
        ):
            config = {
                "mma_type": "wgmma",
                "block_shape": block_shape,
                "warp_shape": warp_shape,
                "use_stream_k": False,
                "use_f16_accum": False,
                "num_stages": 4,
            }
            tables[sublayer] = [[0, 1 << 30, config]]
        layer.humming_tuning = HummingTuning.from_tables(tables)
        self._check(layer, norm, w13_ref, w2_ref, num_tokens=4)

    def test_expert_parallel_step_above_the_one_launch_alignment(self):
        """EP prefill steps past the one-CTA aligner's bound map and align the
        routes with moe_align_block_size instead."""
        num_tokens = 640
        self.assertFalse(
            ep_local_align_supported(
                num_routes=num_tokens * TOP_K, num_local_experts=NUM_LOCAL
            )
        )
        layer, norm, w13_ref, w2_ref = _make_layer(use_expert_map=True)
        self._check(layer, norm, w13_ref, w2_ref, num_tokens)


if __name__ == "__main__":
    unittest.main()
