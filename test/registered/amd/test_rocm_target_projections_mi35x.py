"""Correctness and graph coverage for GLM-5.2 target projection fusions."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip() and is_gfx95_supported()
if _RUNNABLE:
    try:
        from sglang.kernels.ops.attention.mla.hip_gfx950 import (
            is_target_projection_fusion_available,
            target_o_proj,
            target_q_b_proj,
            target_qkv_a_norm,
        )

        _RUNNABLE = is_target_projection_fusion_available()
    except Exception:
        _RUNNABLE = False


def _snr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    signal = expected.square().sum()
    noise = (actual - expected).square().sum().clamp_min(1e-30)
    return float((10 * torch.log10(signal / noise)).item())


@unittest.skipUnless(
    _RUNNABLE, "requires HIP gfx950 and Triton >= 3.5 with Gluon CDNA4 support"
)
class TestROCmTargetProjections(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(11)
        device = "cuda"
        cls.hidden = torch.randn(4, 6144, device=device, dtype=torch.bfloat16).mul_(0.1)
        cls.qkv_weight = torch.randn(
            2624, 6144, device=device, dtype=torch.bfloat16
        ).mul_(0.01)
        cls.q_gamma = (
            torch.randn(2048, device=device, dtype=torch.bfloat16).mul_(0.05).add_(1)
        )
        cls.kv_gamma = (
            torch.randn(512, device=device, dtype=torch.bfloat16).mul_(0.05).add_(1)
        )
        cls.qb_weight = torch.randn(
            4096, 2048, device=device, dtype=torch.bfloat16
        ).mul_(0.01)
        cls.o_weight = torch.randn(
            6144, 4096, device=device, dtype=torch.bfloat16
        ).mul_(0.01)

    def _run_stack(self):
        q, k, rope = target_qkv_a_norm(
            self.hidden,
            self.qkv_weight,
            self.q_gamma,
            self.kv_gamma,
            eps=1e-5,
        )
        qb = target_q_b_proj(q, self.qb_weight)
        output = target_o_proj(qb, self.o_weight)
        return q, k, rope, qb, output

    def test_correctness_and_graph_replay(self):
        eager = self._run_stack()
        torch.cuda.synchronize()

        projected = F.linear(self.hidden, self.qkv_weight)
        q_ref, k_ref, rope_ref = projected.split((2048, 512, 64), dim=-1)
        q_ref = F.rms_norm(q_ref.float(), (2048,), self.q_gamma.float(), 1e-5).to(
            torch.bfloat16
        )
        k_ref = F.rms_norm(k_ref.float(), (512,), self.kv_gamma.float(), 1e-5).to(
            torch.bfloat16
        )
        qb_ref = F.linear(eager[0], self.qb_weight)
        o_ref = F.linear(eager[3], self.o_weight)

        for actual, expected in zip(
            eager, (q_ref, k_ref, rope_ref, qb_ref, o_ref), strict=True
        ):
            self.assertGreater(_snr(actual, expected), 20)

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = self._run_stack()
        for tensor in captured:
            tensor.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(captured, eager, strict=True):
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main(verbosity=3)
