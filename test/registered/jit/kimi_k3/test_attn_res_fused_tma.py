"""K3 attn-residual warp-specialized TMA aggregation (attn_res_fused_tma).

Warp-specialized production forward (sm_100a): a producer warp (group)
streams rows into a double-buffered smem chunk ring via per-row
cp.async.bulk, 8 consumer warps run per-row rms/dot reductions and an online
softmax over row chunks, with cw and ow staged in TMEM and the output RMSNorm
fused into the epilogue. The launch config (chunk_rows / occupancy /
consumer_regs, incl. the setmaxnreg producer/consumer register split) is
dispatched per nvb through _TMA_BEST_CONFIG.

Pinned failure modes:
- chain math (score formula, online-softmax chunk correction across full and
  partial chunks — nvb 1..8 covers every chunk shape of the dispatch table —
  fp32-accumulated combine, fused output RMSNorm) vs an fp32 reference, on
  every launch config the table dispatches: the occupancy=2 config (nvb=1,
  T >= 128 plus its small-T fallback) and the 3/4/5-row-chunk setmaxnreg
  configs;
- row addressing: bank [T, NB, H] with NB > nvb rows must still address rows
  correctly through the runtime bank stride (guards the per-row bulk-copy
  source arithmetic);
- the persistent grid and chunk-slot ring: T large enough that every CTA
  runs >= 3 chunks exercises the token loop AND the mbarrier phase-parity
  flip (a slot is reused with flipped parity only from the third chunk on —
  for single-chunk-per-token configs that needs >= 3 tokens per CTA, at
  occupancy=2 grid size);
- read-only inputs;
- PDL under CUDA graph: a preceding kernel writes prefix_sum and the capture
  must replay bit-identically (guards capturability and the wait placement's
  basic chained correctness — the race itself is not deterministically
  testable).
"""

import unittest

import torch

from sglang.jit_kernel.kimi_k3.attn_res import attn_res_fused_tma
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_H = 7168
_EPS = 1e-6


def _reference(prefix, bank, nvb, cw, ow, eps):
    """fp32 reference of the whole aggregation chain."""
    rows = torch.cat([bank[:, :nvb].float(), prefix.unsqueeze(1).float()], dim=1)
    rms = torch.rsqrt(rows.pow(2).mean(-1) + eps)
    scores = (rows * cw.float()).sum(-1) * rms
    probs = torch.softmax(scores, dim=-1)
    mixed = (probs.unsqueeze(-1) * rows).sum(1)
    return mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + eps) * ow.float()


def _make_inputs(T: int, seed: int, num_bank_slots: int = 8):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    prefix = randn(T, _H).to(torch.bfloat16)
    bank = randn(T, num_bank_slots, _H).to(torch.bfloat16)
    cw = (randn(_H) * _H**-0.5).to(torch.bfloat16)
    ow = (1 + 0.1 * randn(_H)).to(torch.bfloat16)
    return prefix, bank, cw, ow


class TestAttnResFusedTma(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        if get_device_sm() < 100:
            raise unittest.SkipTest("attn_res_fused_tma requires SM100a+")

    def test_matches_reference_all_bank_sizes(self):
        # T = 1 and 5 run every nvb on the small-T side of the dispatch
        # (nvb=1 takes the occupancy=1 fallback); 6*SM+3 crosses the nvb=1
        # occupancy=2 threshold and gives >= 3 tokens per CTA even on its
        # 2*SM grid, so single-chunk-per-token configs still flip the
        # mbarrier phase parity (slot reuse starts at the third chunk).
        num_sm = torch.cuda.get_device_properties(0).multi_processor_count
        for nvb in range(1, 9):
            for T in (1, 5, 6 * num_sm + 3):
                with self.subTest(nvb=nvb, T=T):
                    prefix, bank, cw, ow = _make_inputs(T, seed=8 * T + nvb)
                    prefix_ref, bank_ref = prefix.clone(), bank.clone()

                    out = torch.empty_like(prefix)
                    attn_res_fused_tma(prefix, bank, cw, ow, out, nvb, _EPS)

                    ref = _reference(prefix, bank, nvb, cw, ow, _EPS)
                    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=4e-2)

                    self.assertTrue(torch.equal(prefix, prefix_ref))
                    self.assertTrue(torch.equal(bank, bank_ref))

    def test_wide_bank_row_addressing(self):
        """NB > nvb: rows must be addressed with the true bank stride."""
        prefix, bank, cw, ow = _make_inputs(7, seed=0, num_bank_slots=11)
        out = torch.empty_like(prefix)
        attn_res_fused_tma(prefix, bank, cw, ow, out, 5, _EPS)
        ref = _reference(prefix, bank, 5, cw, ow, _EPS)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=4e-2)

    def test_fused_prefix_write(self):
        """write_prefix=True snapshots the prefix row into bank[:, nvb]
        bit-exactly without touching any other row or the aggregation
        output. nvb 1..8 covers the prefix row landing in every chunk
        shape of the dispatch table; the large T exercises the persistent
        token loop (each CTA writes several tokens' rows)."""
        num_sm = torch.cuda.get_device_properties(0).multi_processor_count
        for nvb in range(1, 9):
            for T in (1, 5, 6 * num_sm + 3):
                with self.subTest(nvb=nvb, T=T):
                    prefix, bank, cw, ow = _make_inputs(
                        T, seed=8 * T + nvb, num_bank_slots=9
                    )
                    bank_ref = bank.clone()
                    out = torch.empty_like(prefix)
                    attn_res_fused_tma(
                        prefix, bank, cw, ow, out, nvb, _EPS, write_prefix=True
                    )
                    ref = _reference(prefix, bank_ref, nvb, cw, ow, _EPS)
                    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=4e-2)
                    self.assertTrue(torch.equal(bank[:, nvb], prefix))
                    self.assertTrue(torch.equal(bank[:, :nvb], bank_ref[:, :nvb]))
                    self.assertTrue(
                        torch.equal(bank[:, nvb + 1 :], bank_ref[:, nvb + 1 :])
                    )

    def test_pdl_chain_under_cuda_graph(self):
        """A preceding kernel writes prefix_sum; capture + replay (where the
        PDL launch attribute is active) must match eager execution."""
        T, nvb = 64, 8
        gen = torch.Generator(device="cuda").manual_seed(0)
        a = torch.randn(T, _H, generator=gen, device="cuda").to(torch.bfloat16)
        b = torch.randn(T, _H, generator=gen, device="cuda").to(torch.bfloat16)
        _, bank, cw, ow = _make_inputs(T, seed=1)
        prefix = torch.empty_like(a)
        out = torch.empty_like(a)

        def chain():
            torch.add(a, b, out=prefix)
            attn_res_fused_tma(prefix, bank, cw, ow, out, nvb, _EPS)

        chain()
        out_eager = out.clone()

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):
                chain()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                chain()
        torch.cuda.synchronize()
        out.zero_()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, out_eager))


if __name__ == "__main__":
    unittest.main()
