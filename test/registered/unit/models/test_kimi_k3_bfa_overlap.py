"""KDA bfa side-stream overlap: forward_qkvbfg_fused must produce outputs
bit-identical to the serial path, both eager and under CUDA graph
capture/replay (the overlap only engages in capture mode)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")

_H = 7168
_QKVG = 6144  # q,k,v,g slices per rank at TP8
_N_FA = 128
_N_B = 12
_BFA_W_ROWS = 144  # [f_a | b] padded to 8 rows like _merge_bfa_weights


def _make_owner(with_stream: bool):
    gen = torch.Generator(device="cuda").manual_seed(0)

    def _randn(*shape):
        return (
            torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)
            .mul(0.05)
            .to(torch.bfloat16)
        )

    qkvg_w = _randn(_QKVG, _H)

    def fused_qkvg_proj(x):
        return torch.nn.functional.linear(x, qkvg_w), None

    owner = SimpleNamespace(
        use_full_rank_gate=True,
        _bfa_w=_randn(_BFA_W_ROWS, _H).contiguous(),
        _bfa_fa_size=_N_FA,
        _bfa_b_size=_N_B,
        f_b_proj=SimpleNamespace(weight=_randn(1536, _N_FA).contiguous()),
        fused_qkvg_proj=fused_qkvg_proj,
        split_sizes=[3 * 1536, 1536],
        _bfa_alt_stream=torch.cuda.Stream() if with_stream else None,
        _bfa_bs_limit=128 if with_stream else 0,
    )
    return owner


def _run(owner, x):
    out = KimiK3DeltaAttention.forward_qkvbfg_fused(owner, x)
    return [t.clone() for t in out]


class TestKimiK3BfaOverlap(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_capture_replay_matches_serial(self):
        torch.manual_seed(0)
        for T in (1, 4, 12):
            with self.subTest(T=T):
                x = (
                    torch.randn(T, _H, device="cuda", dtype=torch.float32)
                    .mul(0.05)
                    .to(torch.bfloat16)
                )
                serial = _run(_make_owner(with_stream=False), x)

                owner = _make_owner(with_stream=True)
                with patch(
                    "sglang.srt.models.kimi_k3.get_is_capture_mode",
                    return_value=True,
                ):
                    # warm up allocations/JIT outside capture
                    _ = _run(owner, x)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = KimiK3DeltaAttention.forward_qkvbfg_fused(owner, x)
                    graph.replay()
                    torch.cuda.synchronize()
                # note: owners share the same seeded weights
                for got, ref, name in zip(
                    captured, serial, ("qkv", "beta", "forget_gate", "g")
                ):
                    self.assertTrue(torch.equal(got, ref), f"T={T} {name} mismatch")

    def test_eager_stream_branch_not_taken(self):
        x = torch.randn(3, _H, device="cuda", dtype=torch.bfloat16)
        serial = _run(_make_owner(with_stream=False), x)
        overlap = _run(_make_owner(with_stream=True), x)  # capture mode False
        for got, ref in zip(overlap, serial):
            self.assertTrue(torch.equal(got, ref))


if __name__ == "__main__":
    unittest.main()
