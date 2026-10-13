"""PPU MoE front: fusion priority, automatic overlap, and stream reuse."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu
from sglang.srt.layers.quantization.w8a8_fp8 import (
    W8A8Fp8Config,
    W8A8Fp8LinearMethod,
)
from sglang.srt.models.kimi_k3 import KimiK3MoE
from sglang.srt.utils import is_ppu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


@contextmanager
def _front_config(ppu=True):
    with patch("sglang.srt.models.kimi_k3.is_ppu", return_value=ppu):
        yield


class TestKimiK3FrontOverlapScope(CustomTestCase):
    def test_ineligible_inputs_preserve_fallback(self):
        cases = (
            {"ppu": False},
            {"m": 0},
            {"latent": False},
            {"projection": None},
            {"stream": None},
        )
        for case in cases:
            with self.subTest(case=case):
                owner = SimpleNamespace(
                    _ep_front_eligible=False,
                    use_latent_moe=case.get("latent", True),
                    routed_expert_down_proj=case.get("projection", object()),
                    alt_stream=case.get("stream", object()),
                    gate=Mock(),
                )
                x = torch.empty((case.get("m", 1), 7168), dtype=torch.bfloat16)
                with _front_config(ppu=case.get("ppu", True)):
                    self.assertIsNone(KimiK3MoE._ep_front_overlap(owner, x))
                owner.gate.assert_not_called()

    def test_original_front_strategy_remains_in_control(self):
        owner = SimpleNamespace(_ep_front_eligible=True, alt_stream=object())
        x = torch.empty((1, 7168), dtype=torch.bfloat16)
        with (
            _front_config(ppu=False),
            patch(
                "sglang.kernels.ops.moe.moe_front.get_front_strategy",
                return_value="merged_fp32",
            ) as strategy,
        ):
            self.assertIsNone(KimiK3MoE._ep_front_overlap(owner, x))
            strategy.assert_called_once_with(1, x.device)

    def test_merged_front_takes_priority_over_overlap(self):
        hidden = torch.ones(8, 16)
        latent = torch.ones(8, 4)
        topk = object()
        owner = SimpleNamespace(
            shared_experts=None,
            use_latent_moe=True,
            _use_mega_moe=False,
            _sbo_shared_overlap=False,
            _can_overlap_shared_experts_npu=Mock(return_value=False),
            _ep_front=Mock(return_value=(topk, latent)),
            _ep_front_overlap=Mock(),
            routed_expert_down_proj=Mock(),
            experts=Mock(return_value=latent),
            _reduce_latent=lambda value: value,
            routed_expert_up_proj=Mock(return_value=(hidden, None)),
        )
        actual = KimiK3MoE._forward_unfused(owner, hidden, prefix_sum=None)
        self.assertIs(actual, hidden)
        owner._ep_front_overlap.assert_not_called()
        owner.routed_expert_down_proj.assert_not_called()
        owner.experts.assert_called_once_with(latent, topk)


class _FP8LatentDown(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        w = torch.randn((3584, 7168), device="cuda", generator=generator) * 0.02
        scale = w.abs().amax(dim=1).clamp_min(1e-6) / 448
        self.weight = torch.nn.Parameter(
            (w / scale[:, None]).to(torch.float8_e4m3fn).t(), requires_grad=False
        )
        self.weight_scale = torch.nn.Parameter(scale[:, None], requires_grad=False)
        self.quant_method = W8A8Fp8LinearMethod(W8A8Fp8Config(True))

    def forward(self, x):
        return self.quant_method.apply(self, x), None


def _make_owner(fp8=True, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(42)
    gate_weight = (
        torch.randn((896, 7168), device="cuda", generator=generator) * 0.02
    ).to(dtype)
    bias = torch.randn((896,), device="cuda", generator=generator) * 0.01

    def gate(x):
        return torch.mm(x, gate_weight.T, out_dtype=torch.float32)

    def topk(x, logits):
        weights, ids = biased_grouped_topk_gpu(
            x, logits, bias, 16, True, 1, 1, routed_scaling_factor=1.0
        )
        return weights, ids, logits

    if fp8:
        down_proj = _FP8LatentDown(generator)
    else:
        weight = (
            torch.randn((3584, 7168), device="cuda", generator=generator) * 0.02
        ).to(dtype)

        def down_proj(x):
            return torch.nn.functional.linear(x, weight), None

    return SimpleNamespace(
        _ep_front_eligible=False,
        use_latent_moe=True,
        routed_expert_down_proj=down_proj,
        alt_stream=torch.cuda.Stream(),
        gate=gate,
        topk=topk,
    )


def _run_front(owner, x):
    result = KimiK3MoE._ep_front_overlap(owner, x)
    if result is None:
        topk_output = owner.topk(x, owner.gate(x))
        latent, _ = owner.routed_expert_down_proj(x)
    else:
        topk_output, latent = result
    return (*topk_output, latent)


class TestKimiK3FrontOverlapPPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not is_ppu():
            raise unittest.SkipTest("requires PPU")
        cls.owner = _make_owner()

    def test_eager_matches_serial(self):
        for m in (1, 2, 4, 8, 16, 32):
            with self.subTest(m=m):
                x = torch.randn((m, 7168), device="cuda", dtype=torch.bfloat16)
                with _front_config(ppu=False):
                    expected = tuple(t.clone() for t in _run_front(self.owner, x))
                with _front_config():
                    result = KimiK3MoE._ep_front_overlap(self.owner, x)
                    self.assertIsNotNone(result)
                    topk_output, latent = result
                    actual = (*topk_output, latent)
                torch.cuda.synchronize()
                for got, ref in zip(actual, expected):
                    self.assertTrue(torch.equal(got, ref))

    def test_dense_projections_and_fp16_inputs_match_serial(self):
        for dtype in (torch.bfloat16, torch.float16):
            owner = _make_owner(fp8=False, dtype=dtype)
            for m in (8, 32):
                with self.subTest(dtype=dtype, m=m):
                    x = torch.randn((m, 7168), device="cuda", dtype=dtype)
                    with _front_config(ppu=False):
                        expected = tuple(t.clone() for t in _run_front(owner, x))
                    with _front_config():
                        result = KimiK3MoE._ep_front_overlap(owner, x)
                    self.assertIsNotNone(result)
                    topk_output, latent = result
                    torch.cuda.synchronize()
                    for got, ref in zip((*topk_output, latent), expected):
                        self.assertTrue(torch.equal(got, ref))

    def test_graph_replay_and_reusing_shared_stream(self):
        owner = self.owner

        def run(x):
            front = _run_front(owner, x)
            current = torch.cuda.current_stream()
            owner.alt_stream.wait_stream(current)
            with torch.cuda.stream(owner.alt_stream):
                shared = x * 0.5
            current.wait_stream(owner.alt_stream)
            shared.record_stream(current)
            return (*front, shared)

        for m in (1, 2, 4, 8, 32):
            with self.subTest(m=m):
                x = torch.randn((m, 7168), device="cuda", dtype=torch.bfloat16)
                with _front_config():
                    warm = torch.cuda.Stream()
                    warm.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(warm):
                        run(x)
                    torch.cuda.current_stream().wait_stream(warm)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = run(x)
                for _ in range(3):
                    x.copy_(torch.randn_like(x))
                    with _front_config(ppu=False):
                        expected = tuple(t.clone() for t in run(x))
                    graph.replay()
                    torch.cuda.synchronize()
                    for got, ref in zip(captured, expected):
                        self.assertTrue(torch.equal(got, ref))
                graph.reset()


if __name__ == "__main__":
    unittest.main()
