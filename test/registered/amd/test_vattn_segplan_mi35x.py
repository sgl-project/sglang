"""Length-aware split-KV segment plan of the gfx950 asm attention kernel (the default split for bs > 1).

Guards, on a gfx950 device:
  * the planned split matches an fp32 reference as closely as the fixed split, for both GQA ratios the
    kernel ships (16 and 8), skewed and tiny lengths, bs 1 / 2 / 24 / 64 and ragged query lengths;
  * the plan itself (segment length T, work list, per-token segment count) matches a Python reference;
  * the per-forward plan cache: one plan launch per forward, reset / in-place update / other tensor
    each trigger a rebuild, cached output bit-identical to uncached.
"""

import math
import unittest

import torch
from torch.profiler import ProfilerActivity, profile

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

HD, PAGE = 256, 16
FP8 = torch.float8_e4m3fn


def _asm_available() -> bool:
    if not (torch.version.hip and torch.cuda.is_available()):
        return False
    from sglang.kernels.ops.attention.vattn_asm_gfx950 import asm_kernel_available

    return asm_kernel_available()


def make(lens, qlens, hq, hkv, seed=0):
    torch.manual_seed(seed)
    kvlens = [l + q for l, q in zip(lens, qlens)]
    npages = [(kv + PAGE - 1) // PAGE for kv in kvlens]
    total = sum(npages) + 3
    perm = torch.randperm(total)
    bt = torch.zeros(len(lens), max(npages), dtype=torch.int32)
    off = 0
    for i, n in enumerate(npages):
        bt[i, :n] = perm[off : off + n].to(torch.int32)
        off += n
    k = (torch.randn(total, PAGE, hkv, HD) / 4).to(FP8)
    v = (torch.randn(total, PAGE, hkv, HD) / 4).to(FP8)
    q = (torch.randn(sum(qlens), hq, HD) / 4).to(torch.bfloat16)
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qlens), 0)), dtype=torch.int32
    )
    seq_lens = torch.tensor(kvlens, dtype=torch.int64)
    kd = torch.full((1,), 0.9, dtype=torch.float32)
    vd = torch.full((1,), 1.1, dtype=torch.float32)
    return k, v, bt, q, cu_q, seq_lens, kd, vd


def ref(k, v, bt, q, cu_q, seq_lens, kd, vd, hq, hkv):
    gqa = hq // hkv
    outs = []
    for s in range(bt.shape[0]):
        kvlen = int(seq_lens[s])
        ql = int(cu_q[s + 1] - cu_q[s])
        pages = bt[s].long()
        kk = k[pages].reshape(-1, hkv, HD)[:kvlen].float() * kd
        vv = v[pages].reshape(-1, hkv, HD)[:kvlen].float() * vd
        qq = q[int(cu_q[s]) : int(cu_q[s + 1])].float()
        o = torch.empty(ql, hq, HD)
        for t in range(ql):
            L = kvlen - ql + t + 1
            for h in range(hq):
                kvh = h // gqa
                sc = (qq[t, h] @ kk[:L, kvh].T) / math.sqrt(HD)
                o[t, h] = torch.softmax(sc, dim=-1) @ vv[:L, kvh]
        outs.append(o)
    return torch.cat(outs)


def _cdiv(x, y):
    return -(-x // y)


CASES = []
for _hq, _hkv in ((16, 1), (16, 2)):  # GQA ratios 16 and 8, the two the kernel ships
    CASES += [
        ([70000] * 16, [4] * 16, _hq, _hkv, "uniform 16x70k"),
        (
            [248000, 120000, 76000, 60000, 34000, 20000, 9000, 3000]
            + [1500, 500, 100, 40, 17, 5, 1, 0],
            [4] * 16,
            _hq,
            _hkv,
            "agent skew + tiny",
        ),
        ([248000], [4], _hq, _hkv, "bs1 248k"),
        ([1], [4], _hq, _hkv, "bs1 len1"),
        ([200000, 3000], [4, 4], _hq, _hkv, "bs2 skew"),
        ([30000, 12000, 40000, 90000], [4, 1, 2, 3], _hq, _hkv, "ragged q 4/1/2/3"),
        ([2000 + 3000 * (i % 7) for i in range(64)], [4] * 64, _hq, _hkv, "bs64 clamp"),
        (
            [50000 + 7000 * (i % 5) for i in range(24)],
            [4] * 24,
            _hq,
            _hkv,
            "bs24 mild skew",
        ),
    ]


@unittest.skipUnless(_asm_available(), "needs a gfx950 device with ROCm clang")
class TestVattnSegPlan(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import sglang.kernels.ops.attention.vattn_asm_gfx950 as V

        cls.V = V
        torch.set_default_device("cuda")

    def _check_plan(self, lens, qlens, hkv, seq_lens, cu_q):
        V = self.V
        seg_max = V.mtp_verify_attn_seg_max(len(lens), hkv)
        plan, tok_nseg = V.build_seg_plan(seq_lens, cu_q, int(cu_q[-1]), seg_max, hkv)
        plan = plan.tolist()
        T, work = plan[0], plan[1:]
        nts = [(kv + 15) // 16 for kv in seq_lens.tolist()]
        target = V.seg_plan_target_wgs(hkv)
        lo = max(_cdiv(sum(nts), target), _cdiv(max(nts), seg_max), 1)
        slack = target - len(lens)
        hi = max(lo, _cdiv(sum(nts), slack)) if slack > 0 else lo
        while lo < hi:
            mid = (lo + hi) // 2
            if sum(_cdiv(n, mid) for n in nts) <= target:
                hi = mid
            else:
                lo = mid + 1
        self.assertEqual(T, hi)
        nseg = [_cdiv(n, T) for n in nts]
        exp_work = [(b << 16) | sg for b, n in enumerate(nseg) for sg in range(n)]
        exp_work += [-1] * (len(work) - len(exp_work))
        self.assertEqual(work, exp_work)
        self.assertLessEqual(max(nseg), seg_max)
        self.assertTrue(sum(nseg) <= target or slack <= 0)
        exp_tn = [nseg[s] for s, ql in enumerate(qlens) for _ in range(ql)]
        self.assertEqual(tok_nseg.tolist(), exp_tn)

    def test_planned_split_matches_reference(self):
        V = self.V
        for lens, qlens, hq, hkv, tag in CASES:
            with self.subTest(case=tag, hq=hq, hkv=hkv):
                k, v, bt, q, cu_q, seq_lens, kd, vd = make(lens, qlens, hq, hkv)
                scale = 1.0 / math.sqrt(HD)
                r = ref(k, v, bt, q, cu_q, seq_lens, kd, vd, hq, hkv)
                o_leg = V.mtp_verify_attn_fwd_asm(
                    q, k, v, bt, seq_lens, cu_q, kd, vd, scale, use_seg_plan=False
                ).float()
                o_plan = V.mtp_verify_attn_fwd_asm(
                    q, k, v, bt, seq_lens, cu_q, kd, vd, scale
                ).float()
                torch.cuda.synchronize()
                if len(lens) > 1:
                    self._check_plan(lens, qlens, hkv, seq_lens, cu_q)
                e_leg = (o_leg - r).abs().max().item()
                e_plan = (o_plan - r).abs().max().item()
                self.assertFalse(torch.isnan(o_plan).any().item())
                # same error budget as the fixed split (fp8 KV dominates); 0.02 floor for the tiny cases
                self.assertLessEqual(e_plan, max(2 * e_leg, 0.02))
                torch.cuda.empty_cache()

    def test_plan_cache_per_forward(self):
        V = self.V
        lens, qlens, hq, hkv = [248000, 60000, 9000, 500, 17, 0], [4] * 6, 16, 1
        k, v, bt, q, cu_q, seq_lens, kd, vd = make(lens, qlens, hq, hkv)
        scale = 1.0 / math.sqrt(HD)

        def call(sl=seq_lens, cq=cu_q):
            return V.mtp_verify_attn_fwd_asm(q, k, v, bt, sl, cq, kd, vd, scale)

        def plan_launches(fn):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            return sum(e.count for e in prof.key_averages() if "seg_plan" in e.key)

        call()
        torch.cuda.synchronize()
        V.reset_seg_plan_cache()
        self.assertEqual(plan_launches(lambda: [call() for _ in range(15)]), 1)
        V.reset_seg_plan_cache()
        self.assertEqual(plan_launches(call), 1)
        seq_lens[1] += 16  # in-place update (new _version) -> rebuild
        self.assertEqual(plan_launches(call), 1)
        sl2 = seq_lens.clone()
        self.assertEqual(plan_launches(lambda: call(sl2)), 1)
        V.reset_seg_plan_cache()
        o_cached = call().float()
        o_cached2 = call().float()  # served from the cache
        o_fresh = call(
            seq_lens.clone(), cu_q.clone()
        ).float()  # new tensors -> freshly built plan
        self.assertTrue(torch.equal(o_cached, o_cached2))
        self.assertTrue(torch.equal(o_cached, o_fresh))
        r = ref(k, v, bt, q, cu_q, seq_lens, kd, vd, hq, hkv)
        self.assertLess((o_cached - r).abs().max().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
