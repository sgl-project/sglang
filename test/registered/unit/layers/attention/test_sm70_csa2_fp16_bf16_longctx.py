"""fp16 vs bf16 indexer at Flash-scale context (no Hopper).

V100 cannot run bf16 kernels. This is the CPU oracle twin: last-row top-512
at 3k/9k/18k/30k, reporting all-512 Jaccard *and* whether the early needle
slice is in the shortlist under each dtype.

Distinct needle XOR must stay 0 (a unique fact must not fall out of fp16
only). Near-tie XOR is the measured SM70 tax; it is printed, not gated.

This scores one last row per length (~10 s). It is not 30 min of TG.
Live decode soak: test/manual/dsv41_v100/test_tg_soak_cohesion.py
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.attention.dsv4 import sm70_csa2_reference as R
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

torch.set_num_threads(min(4, torch.get_num_threads()))

FP16 = torch.float16
BF16 = torch.bfloat16
FLASH_TOPK = 512
FLASH_CAND_BLOCKS = 2048
FLASH_CAND_BLOCK = 8
LENGTHS = (3000, 9000, 18000, 30000)
NEEDLE = list(range(120, 200))
INDEX_HEADS = 32
INDEX_DIM = 128


def _dir(dim: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm().clamp_min(1e-8) * scale


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = {int(i) for i in a.tolist() if i >= 0}
    sb = {int(i) for i in b.tolist() if i >= 0}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def _in_topk(idx: torch.Tensor, ids) -> bool:
    sel = {int(i) for i in idx.tolist() if i >= 0}
    return any(i in sel for i in ids)


def _last_row_topk(k_fp32: torch.Tensor, q_vec: torch.Tensor, n: int):
    """Score one query against k[:n] in fp16 and bf16. Returns idx16, idxbf, jacc."""
    h, d = INDEX_HEADS, INDEX_DIM
    q16 = R.fake_quant_fp4_ue8m0(
        q_vec.view(1, 1, d).expand(1, h, d).contiguous().to(FP16)
    )
    qbf = R.fake_quant_fp4_ue8m0(
        q_vec.view(1, 1, d).expand(1, h, d).contiguous().to(BF16)
    )
    k16 = R.fake_quant_fp4_ue8m0(k_fp32[:n].to(FP16))
    kbf = R.fake_quant_fp4_ue8m0(k_fp32[:n].to(BF16))
    w16 = torch.ones(1, h, dtype=FP16)
    logits16 = R.index_scores(q16, k16, w16)
    logitsbf = R.index_scores(qbf, kbf, w16.to(BF16))
    lens = torch.tensor([[n]])
    cand16 = R.select_candidate_blocks(
        logits16, lens, FLASH_CAND_BLOCKS, FLASH_CAND_BLOCK
    )
    candbf = R.select_candidate_blocks(
        logitsbf, lens, FLASH_CAND_BLOCKS, FLASH_CAND_BLOCK
    )
    m16 = logits16.masked_fill(~cand16[:, :n], -torch.inf)
    mbf = logitsbf.masked_fill(~candbf[:, :n], -torch.inf)
    idx16 = R.topk_positions(m16, lens, FLASH_TOPK)[0]
    idxbf = R.topk_positions(mbf, lens, FLASH_TOPK)[0]
    return idx16, idxbf, _jaccard(idx16, idxbf)


class TestFp16Bf16Longctx(CustomTestCase):
    def test_distinct_and_neartie_needle_xor_vs_length(self):
        g = torch.Generator().manual_seed(90)
        d = INDEX_DIM
        needle_v = _dir(d, 1, 4.0)
        q_ask = needle_v
        k_distinct = torch.randn(LENGTHS[-1], d, generator=g) * 0.25
        k_distinct[NEEDLE] = needle_v
        k_tie = torch.randn(LENGTHS[-1], d, generator=g) * 0.25
        k_tie[NEEDLE] = _dir(d, 1, 1.0) + 0.03 * torch.randn(d, generator=g)

        print(
            f"\n--- fp16 vs bf16 last-row indexer topk={FLASH_TOPK} "
            f"cand={FLASH_CAND_BLOCKS}x{FLASH_CAND_BLOCK} ---"
        )
        print(
            f"{'n':>6} {'jacc_d':>7} {'d16':>5} {'dbf':>5} {'dxor':>5} "
            f"{'jacc_t':>7} {'t16':>5} {'tbf':>5} {'txor':>5}"
        )
        distinct_xor = []
        for n in LENGTHS:
            i16, ibf, jd = _last_row_topk(k_distinct, q_ask, n)
            t16, tbf, jt = _last_row_topk(k_tie, q_ask, n)
            d16, dbf = _in_topk(i16, NEEDLE), _in_topk(ibf, NEEDLE)
            n16, nbf = _in_topk(t16, NEEDLE), _in_topk(tbf, NEEDLE)
            dx, tx = int(d16 != dbf), int(n16 != nbf)
            distinct_xor.append(dx)
            print(
                f"{n:6d} {jd:7.3f} {str(d16):>5} {str(dbf):>5} {dx:5d} "
                f"{jt:7.3f} {str(n16):>5} {str(nbf):>5} {tx:5d}"
            )
            self.assertTrue(d16 and dbf, f"distinct needle missing at n={n}")
            self.assertGreater(jd, 0.5, f"all-512 Jaccard collapsed at n={n}")
        self.assertEqual(
            sum(distinct_xor),
            0,
            "fp16 dropped a unique fact that bf16 kept (or the reverse)",
        )


if __name__ == "__main__":
    unittest.main()
