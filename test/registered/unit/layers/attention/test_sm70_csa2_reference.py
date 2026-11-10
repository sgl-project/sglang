"""CPU tests for the SM70 DeepSeek-V4.1 CSA2 torch reference.

Oracles used here, in order of authority:

1. Verbatim ports of the model's reference formulas (``inference/model.py``,
   ``inference/kernel.py`` of deepseek-ai/DeepSeek-V4.1-Flash) and of upstream
   SGLang ``torch_quant.py`` (PR #38798), inlined below so the test does not
   depend on the runtime port landing.
2. Brute-force loops (per query / per pair / per block) for the vectorized logic.
3. Invariants: chunking must not change selections; the two-level candidate
   scheme must collapse to plain top-k when every block fits.

Everything runs in fp16 (SM70) unless a test is explicitly about bf16 parity.
"""

import math
import unittest

import torch

from sglang.srt.layers.attention.dsv4 import sm70_csa2_reference as R
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=150, suite="base-a-test-cpu")

torch.set_num_threads(min(4, torch.get_num_threads()))

FP16 = torch.float16
BF16 = torch.bfloat16


# --------------------------------------------------------------------------- #
# Oracles: verbatim upstream / HF formulas
# --------------------------------------------------------------------------- #


def upstream_ceil_pow2(x):
    bits = x.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    has_mantissa = (bits & 0x7FFFFF) != 0
    exponent = exponent + has_mantissa.to(torch.int32)
    return ((exponent + 127) << 23).view(torch.float32)


def upstream_block_scale(x, block_size, fmax, amax_floor):
    amax = x.float().unflatten(-1, (-1, block_size)).abs().amax(dim=-1)
    amax = amax.clamp_min(amax_floor)
    return upstream_ceil_pow2(amax * (1.0 / fmax))


def upstream_round_fp4(x):
    magnitude = x.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    return torch.round(magnitude / step) * step * torch.sign(x)


def upstream_fake_quant_fp4(x, block_size=32):
    """sgl/dsv41 torch_quant.fake_quant_fp4 (indexer q/k)."""
    scale = upstream_block_scale(x, block_size, 6.0, 6 * 2.0**-126)
    scaled = x.float().unflatten(-1, (-1, block_size)) / scale.unsqueeze(-1)
    deq = upstream_round_fp4(scaled.clamp(-6.0, 6.0)) * scale.unsqueeze(-1)
    return deq.flatten(-2).to(x.dtype)


def upstream_fake_quant_compressed_kv(x):
    """sgl/dsv41 torch_quant.fake_quant_compressed_kv (main compressed KV)."""
    blocks = x.float().unflatten(-1, (-1, 16))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (amax * (1.0 / 6.0)).clamp(min=2**-9, max=448.0)
    scale = scale.to(torch.float8_e4m3fn).float()
    scaled = (blocks / scale).clamp(-6.0, 6.0)
    deq = upstream_round_fp4(scaled) * scale
    return deq.flatten(-2).to(x.dtype)


def hf_act_quant_ue8m0_inplace(x, block_size=32):
    """HF kernel.act_quant_kernel(round_scale=True, inplace=True): fp8 fake quant
    of the window K, amax floored at 1e-4, power-of-two scale."""
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    # fast_round_scale: 2 ** ceil(log2(amax / 448)) via IEEE bits
    scale = upstream_ceil_pow2((amax * (1.0 / 448.0)).contiguous())
    y = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return y.flatten(-2).to(x.dtype)


def hf_select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    import torch.nn.functional as F

    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def rows_as_sets(idx):
    return [set(r[r >= 0].tolist()) for r in idx]


def assert_topk_equivalent(tc, idx_a, idx_b, logits, rel_tie=1e-5, msg=""):
    """Selections equal up to ties: every position in exactly one of the two
    selections must score within ``rel_tie`` of the row's k-th best score (the
    selection boundary). Returns the number of rows that differed."""
    tc.assertEqual(idx_a.shape, idx_b.shape, msg)
    differing = 0
    for r, (a, b) in enumerate(zip(rows_as_sets(idx_a), rows_as_sets(idx_b))):
        if a == b:
            continue
        differing += 1
        tc.assertEqual(len(a), len(b), f"{msg} row {r}: different selection sizes")
        row = logits[r].double()
        kth = row.topk(len(a)).values[-1].item()
        scale = max(abs(kth), 1.0)
        for j in a ^ b:
            tc.assertLessEqual(
                abs(row[j].item() - kth),
                rel_tie * scale,
                f"{msg} row {r}: position {j} (score {row[j].item()}) is not a tie "
                f"at the boundary {kth}",
            )
    return differing


def brute_force_index_topk(cfg, q, k, w, positions, ratio, topk, cand_mask=None):
    """Per-query loop over compressed positions in fp64."""
    T = q.shape[0]
    n = k.shape[0]
    out = torch.full((T, topk), -1, dtype=torch.int32)
    scores_all = torch.full((T, n), -math.inf, dtype=torch.float64)
    for t in range(T):
        lens = int((positions[t] + 1) // ratio)
        s = torch.full((n,), -math.inf, dtype=torch.float64)
        for j in range(min(lens, n)):
            acc = 0.0
            for h in range(q.shape[1]):
                dot = torch.dot(q[t, h].double(), k[j].double()).item()
                acc += max(dot, 0.0) * w[t, h].double().item()
            s[j] = acc
        if cand_mask is not None:
            s[~cand_mask[t, :n]] = -math.inf
        scores_all[t] = s
        reach = (s > -math.inf).nonzero().flatten()
        kk = min(topk, reach.numel())
        if kk:
            pick = s[reach].topk(kk).indices
            sel = reach[pick].sort().values
            out[t, :kk] = sel.to(torch.int32)
    return out, scores_all


def dense_attention_oracle(q, keys, valid, sink, scale):
    """fp64 softmax with a sink column over an explicit key multiset."""
    s = torch.einsum("thd,tkd->thk", q.double(), keys.double()) * scale
    s = s.masked_fill(~valid[:, None, :], -math.inf)
    s = torch.cat([s, sink.double()[None, :, None].expand(s.shape[0], -1, 1)], dim=-1)
    p = s.softmax(dim=-1)[..., :-1]
    return torch.einsum("thk,tkd->thd", p, keys.double())


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestTopology(CustomTestCase):
    def test_hf_flash_topology(self):
        cfg = R.CSA2Config()
        self.assertEqual(cfg.num_layers, 40)
        self.assertEqual(cfg.compress_ratios[:2], (0, 0))
        self.assertEqual(cfg.compress_ratios[2:20], (2,) * 18)
        self.assertEqual(cfg.compress_ratios[20:40], (1,) * 20)
        kv_src = [cfg.kv_source_for(l) for l in range(40)]
        ix_src = [cfg.index_source_for(l) for l in range(40)]
        self.assertEqual(kv_src[:2], [None, None])
        self.assertEqual(kv_src[2:8], [2] * 6)
        self.assertEqual(kv_src[8:14], [8] * 6)
        self.assertEqual(kv_src[14:20], [14] * 6)
        self.assertEqual(kv_src[20:40], [20] * 20)
        self.assertEqual(ix_src[2:20], kv_src[2:20])
        self.assertEqual(ix_src[20:24], [20] * 4)
        self.assertEqual(ix_src[24:28], [24] * 4)
        self.assertEqual(ix_src[36:40], [36] * 4)
        self.assertTrue(cfg.is_candidate_source(20))
        self.assertEqual(
            [l for l in range(40) if cfg.uses_candidates(l)], [24, 28, 32, 36]
        )
        for l in (2, 8, 14):
            self.assertFalse(cfg.uses_candidates(l))
        self.assertAlmostEqual(cfg.index_head_weight_scale, 128**-0.5 * 32**-0.5)
        self.assertEqual(cfg.candidate_topk_blocks * cfg.candidate_block_size, 16384)


class TestFp4Fp8Primitives(CustomTestCase):
    def test_e2m1_rounding_is_nearest_even(self):
        grid = R.E2M1_VALUES
        # every exact half-way point plus random values
        halves = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
        x = torch.cat([halves, -halves, torch.rand(4096) * 12 - 6, torch.zeros(1)])
        got = R.round_e2m1(x)
        for v, g in zip(x.tolist(), got.tolist()):
            d = (grid - abs(v)).abs()
            best = d.min()
            cands = (d == best).nonzero().flatten().tolist()
            if len(cands) == 2:  # tie -> even code
                cands = [c for c in cands if c % 2 == 0]
            expect = grid[cands[0]].item() * (1 if v >= 0 else -1)
            self.assertEqual(abs(g), abs(expect), f"{v} -> {g}, expected {expect}")

    def test_ceil_pow2_bit_exact(self):
        x = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.0001, 0.7, 2.0**-126, 448.0, 1e-4])
        got = R.ceil_pow2(x)
        expect = torch.tensor([2.0 ** math.ceil(math.log2(v)) for v in x.tolist()])
        self.assertTrue(torch.equal(got, expect), (got, expect))
        self.assertTrue(torch.equal(got, upstream_ceil_pow2(x)))

    def test_fp4_ue8m0_matches_upstream(self):
        for dtype in (FP16, BF16):
            x = torch.randn(64, 128, generator=torch.Generator().manual_seed(1)) * 3
            x[3, :32] = 0  # all-zero block
            x[5, 32:64] = 6 * 2.0**-126 / 3  # under the amax floor
            x = x.to(dtype)
            self.assertTrue(
                torch.equal(R.fake_quant_fp4_ue8m0(x), upstream_fake_quant_fp4(x))
            )
            # idempotent: values already on the grid survive a second pass
            y = R.fake_quant_fp4_ue8m0(x)
            self.assertTrue(torch.equal(R.fake_quant_fp4_ue8m0(y), y))

    def test_fp4_e4m3_matches_upstream_and_hf_floor(self):
        for dtype in (FP16, BF16):
            g = torch.Generator().manual_seed(2)
            x = torch.randn(64, 512, generator=g) * 2
            x[0, :16] = 0  # zero block keeps scale 2**-9 and stays zero
            x[1, 16:32] = 1e-4  # tiny block: scale clamps to 2**-9
            x[2, 32:48] = 3000.0  # scale saturates at 448 (fp16 max ~65504 ok)
            x = x.to(dtype)
            mine = R.fake_quant_fp4_e4m3(x)
            self.assertTrue(torch.equal(mine, upstream_fake_quant_compressed_kv(x)))
            self.assertTrue(torch.all(mine[0, :16] == 0))
            self.assertTrue(torch.equal(R.fake_quant_fp4_e4m3(mine), mine))
            # HF floors amax at 6 * 2**-9 before e4m3(amax/6): identical to the clamp
            self.assertEqual(R.FP4_E4M3_AMAX_FLOOR / 6.0, R.E4M3_MIN_NORMAL_SCALE)

    def test_matches_ported_upstream_torch_quant_module(self):
        # The port of sgl/dsv41 torch_quant.py, when present in the tree.
        try:
            from sglang.srt.layers.attention.dsv4 import torch_quant as up
        except ImportError:  # pragma: no cover - port not landed
            self.skipTest("upstream torch_quant.py not ported yet")
        g = torch.Generator().manual_seed(9)
        for dtype in (FP16, BF16):
            kv = (torch.randn(48, 512, generator=g) * 2).to(dtype)
            kv[0, :16] = 0
            self.assertTrue(
                torch.equal(R.fake_quant_fp4_e4m3(kv), up.fake_quant_compressed_kv(kv))
            )
            ik = torch.randn(48, 128, generator=g).to(dtype)
            self.assertTrue(
                torch.equal(R.fake_quant_fp4_ue8m0(ik), up.fake_quant_fp4(ik))
            )
        x = torch.rand(1024) * 10 + 1e-3
        self.assertTrue(torch.equal(R.ceil_pow2(x), up.ceil_pow2(x)))

    def test_swa_fp8_matches_hf_act_quant(self):
        for dtype in (FP16, BF16):
            g = torch.Generator().manual_seed(3)
            x = (torch.randn(32, 512, generator=g) * 1.5).to(dtype)
            x[0, :32] = 0
            self.assertTrue(
                torch.equal(R.fake_quant_fp8_ue8m0(x), hf_act_quant_ue8m0_inplace(x))
            )

    def test_pack_unpack_round_trips_bit_exact(self):
        g = torch.Generator().manual_seed(4)
        kv = (torch.randn(37, 512, generator=g) * 2).to(FP16)
        kv[0, :16] = 0
        payload, scales = R.pack_kv_fp4_e4m3(kv)
        self.assertEqual((payload.shape[-1], scales.shape[-1]), (256, 32))  # 288 B/row
        self.assertEqual(payload.dtype, torch.uint8)
        self.assertTrue(
            torch.equal(
                R.unpack_kv_fp4_e4m3(payload, scales, FP16), R.fake_quant_fp4_e4m3(kv)
            )
        )

        ik = (torch.randn(37, 128, generator=g)).to(FP16)
        payload, exps = R.pack_index_fp4_ue8m0(ik)
        self.assertEqual((payload.shape[-1], exps.shape[-1]), (64, 4))  # 68 B/row
        self.assertTrue(
            torch.equal(
                R.unpack_index_fp4_ue8m0(payload, exps, FP16),
                R.fake_quant_fp4_ue8m0(ik),
            )
        )

        sk = (torch.randn(37, 512, generator=g)).to(FP16)
        payload, exps = R.pack_swa_fp8_ue8m0(sk)
        self.assertEqual((payload.shape[-1], exps.shape[-1]), (512, 16))  # 528 B/row
        self.assertTrue(
            torch.equal(
                R.unpack_swa_fp8_ue8m0(payload, exps, FP16), R.fake_quant_fp8_ue8m0(sk)
            )
        )

    def test_fp4_values_exact_in_fp16(self):
        # E2M1 x E4M3 products have <= 6 significant bits: the fp16 storage of a
        # dequantized row loses nothing (fusion may keep fp16 rows for the PV GEMM).
        g = torch.Generator().manual_seed(5)
        x = torch.randn(16, 512, generator=g) * 4
        q32 = R.fake_quant_fp4_e4m3(x)  # fp32 in, fp32 out
        self.assertTrue(torch.equal(q32.to(FP16).float(), q32))
        q32 = R.fake_quant_fp4_ue8m0(torch.randn(16, 128, generator=g))
        self.assertTrue(torch.equal(q32.to(FP16).float(), q32))

    def test_flashmla_fp8_layout_is_a_different_rounding(self):
        # Upstream SGLang requantizes the FP4 latent into FlashMLA's per-64-tile
        # FP8 layout; the model keeps FP4/E4M3-16. Both are lossy; document the gap.
        g = torch.Generator().manual_seed(6)
        lat = R.fake_quant_fp4_e4m3((torch.randn(64, 512, generator=g) * 2).to(FP16))
        via_flashmla = R.fake_quant_fp8_flashmla(lat)
        self.assertFalse(torch.equal(via_flashmla, lat))
        rel = (via_flashmla.float() - lat.float()).abs().max() / lat.float().abs().max()
        self.assertLess(rel.item(), 2**-3)  # bounded by e4m3 precision of the tile amax


class TestRope(CustomTestCase):
    def test_yarn_tables_and_inverse(self):
        cfg = R.tiny_config()
        f_swa = R.layer_freqs_cis(cfg, 0)
        f_c = R.layer_freqs_cis(cfg, 2)
        self.assertEqual(f_swa.shape, (cfg.max_seq_len, 32))
        self.assertFalse(torch.equal(f_swa, f_c))  # theta 10000 vs 160000 + YaRN
        self.assertTrue(torch.allclose(f_c.abs(), torch.ones_like(f_c.abs())))
        x = torch.randn(8, 4, 512).to(FP16)
        pos = torch.arange(100, 108)
        y = R.rope_tail(x, f_c[pos], 64)
        self.assertTrue(torch.equal(y[..., :448], x[..., :448]))
        back = R.rope_tail(y, f_c[pos], 64, inverse=True)
        self.assertTrue(torch.allclose(back.float(), x.float(), atol=4e-3))


class TestCompressor(CustomTestCase):
    def test_ratio2_pooling_matches_pair_loop(self):
        g = torch.Generator().manual_seed(7)
        kv = torch.randn(10, 2, 512, generator=g)
        sc = torch.randn(10, 2, 512, generator=g)
        got = R.pool_pairs(kv, sc)
        for i in range(10):
            for d in range(0, 512, 37):
                a, b = sc[i, 0, d].item(), sc[i, 1, d].item()
                m = max(a, b)
                pa, pb = math.exp(a - m), math.exp(b - m)
                expect = (kv[i, 0, d].item() * pa + kv[i, 1, d].item() * pb) / (pa + pb)
                self.assertAlmostEqual(got[i, d].item(), expect, places=5)

    def test_ratio2_pending_pair_across_chunks(self):
        cfg = R.tiny_config(
            num_layers=3,
            compress_ratios=(0, 0, 2),
            kv_source_layer_ids=(2,),
            index_source_layer_ids=(2,),
            candidate_source_layer_id=-1,
        )
        g = torch.Generator().manual_seed(8)
        x = torch.randn(11, cfg.hidden_size, generator=g)
        one = R.build_reference(cfg, FP16)
        one.forward_chunk(x)
        two = R.build_reference(cfg, FP16)
        two.forward_chunk(x[:5])  # ends mid-pair: token 4 pends
        self.assertIsNotNone(two.kv_state[2].pending_kv)
        two.forward_chunk(x[5:8])
        two.forward_chunk(x[8:9])
        two.forward_chunk(x[9:])
        self.assertEqual(one.kv_state[2].num_compressed, 5)
        self.assertEqual(two.kv_state[2].num_compressed, 5)
        a = torch.stack(one.kv_state[2].latents).float()
        b = torch.stack(two.kv_state[2].latents).float()
        # same math; only GEMM shape differs -> at most an FP4 step on a boundary hit
        self.assertTrue(
            torch.allclose(a, b, atol=0.0, rtol=0.0) or (a - b).abs().max() < 0.2
        )
        ka = torch.stack(one.kv_state[2].index_k).float()
        kb = torch.stack(two.kv_state[2].index_k).float()
        self.assertLess((ka - kb).abs().max().item(), 0.2)
        self.assertIsNotNone(two.kv_state[2].pending_kv)  # 11 tokens: token 10 pends
        self.assertIsNotNone(one.kv_state[2].pending_kv)


class TestIndexer(CustomTestCase):
    def _indexer_case(self, ratio, T, topk, cand_blocks, seed):
        cfg = R.tiny_config(
            num_layers=3,
            compress_ratios=(0, 0, ratio),
            kv_source_layer_ids=(2,),
            index_source_layer_ids=(2,),
            candidate_source_layer_id=2 if cand_blocks else -1,
            candidate_topk_blocks=cand_blocks or 2048,
            index_topk=topk,
        )
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(T, cfg.hidden_size, generator=g)
        ref = R.build_reference(cfg, FP16)
        out = ref.forward_chunk(x)
        return cfg, ref, x, out

    def test_visibility_and_topk_match_brute_force(self):
        for ratio, T, topk in (
            (1, 128, 64),
            (2, 128, 32),
            (1, 1024, 512),
            (2, 1024, 512),
        ):
            with self.subTest(ratio=ratio, T=T, topk=topk):
                cfg, ref, x, out = self._indexer_case(ratio, T, topk, 0, 10 + ratio)
                lw = ref.weights[2]
                positions = torch.arange(T)
                q_lora = R.rmsnorm(
                    R.linear(x.to(FP16), lw.wq_a), lw.q_norm, cfg.rms_norm_eps
                )
                q = R.index_queries(q_lora, ref.freqs[2][positions], lw, cfg)
                w = R.index_head_weights(x.to(FP16), lw, cfg)
                k = torch.stack(ref.kv_state[2].index_k)
                self.assertEqual(k.shape[0], T // ratio)
                # brute force only on a sample of query rows at 1024 (fp64 loops are slow)
                rows = (
                    list(range(T))
                    if T <= 128
                    else [0, 1, 2, 3, 127, 128, 129, 511, 512, 513, 1000, 1023]
                )
                bf_idx, bf_scores = brute_force_index_topk(
                    cfg, q[rows], k, w[rows], positions[rows], ratio, topk
                )
                got = out.topk_idx[2][rows]
                # row t sees exactly (t+1)//ratio positions
                for i, t in enumerate(rows):
                    n_vis = (t + 1) // ratio
                    self.assertEqual(
                        (got[i] >= 0).sum().item(), min(topk, n_vis), f"row {t}"
                    )
                    if n_vis:
                        self.assertLess(got[i][got[i] >= 0].max().item(), n_vis)
                assert_topk_equivalent(
                    self, got, bf_idx, bf_scores, msg=f"ratio {ratio} T {T}"
                )

    def test_candidate_two_level_restricts_consumers(self):
        # 4 blocks x 8 = 32 candidate positions, topk 16: consumers must stay inside
        cfg = R.tiny_config(
            num_layers=6,
            compress_ratios=(0, 0, 1, 1, 1, 1),
            kv_source_layer_ids=(2,),
            index_source_layer_ids=(2, 4),
            candidate_source_layer_id=2,
            candidate_topk_blocks=4,
            candidate_block_size=8,
            index_topk=16,
        )
        T = 200
        g = torch.Generator().manual_seed(20)
        x = torch.randn(T, cfg.hidden_size, generator=g)
        ref = R.build_reference(cfg, FP16)
        out = ref.forward_chunk(x)
        mask = out.candidate_mask
        self.assertEqual(mask.shape, (T, T))
        lens = torch.arange(1, T + 1)
        for t in range(T):
            n_vis = t + 1
            last_block = (n_vis - 1) // 8
            blocks = mask[t].view(-1, 8).any(-1)
            self.assertLessEqual(blocks.sum().item(), 4, f"row {t}")
            self.assertTrue(
                blocks[last_block].item(), f"row {t}: newest block not pinned"
            )
            # the mask is block-granular: the pinned partial block extends past
            # n_vis (those positions stay -inf in the logits); whole unreachable
            # blocks must not be kept
            self.assertFalse(
                blocks[last_block + 1 :].any().item(),
                f"row {t}: unreachable block kept",
            )
            sel = out.topk_idx[4][t]
            sel = sel[sel >= 0]
            self.assertTrue(
                mask[t, sel.to(torch.int64)].all().item(),
                f"row {t}: consumer left the candidates",
            )
            self.assertEqual(sel.numel(), min(16, int(mask[t, :n_vis].sum())))
        # the publisher itself selects over all positions (level one only publishes)
        pub = out.topk_idx[2][T - 1]
        self.assertEqual((pub >= 0).sum().item(), 16)
        # level one equals the HF formula on the reference's own logits
        lw = ref.weights[2]
        q_lora = R.rmsnorm(R.linear(x.to(FP16), lw.wq_a), lw.q_norm, cfg.rms_norm_eps)
        q = R.index_queries(q_lora, ref.freqs[2][torch.arange(T)], lw, cfg)
        w = R.index_head_weights(x.to(FP16), lw, cfg)
        logits = R.index_scores(q, torch.stack(ref.kv_state[2].index_k), w)
        logits = logits.masked_fill(
            torch.arange(T)[None, :] >= lens[:, None], -math.inf
        )
        self.assertTrue(
            torch.equal(hf_select_candidate_blocks(logits, lens[:, None], 4, 8), mask)
        )

    def test_candidates_collapse_to_plain_topk_when_all_blocks_fit(self):
        # Real budget (2048 x 8 = 16384) at T = 1024: level one keeps every block, so
        # layers 24..36 must select exactly what they would without a candidate source.
        common = dict(
            num_layers=6,
            compress_ratios=(0, 0, 1, 1, 1, 1),
            kv_source_layer_ids=(2,),
            index_source_layer_ids=(2, 4),
            index_topk=64,
        )
        g = torch.Generator().manual_seed(21)
        x = torch.randn(1024, 96, generator=g)
        with_c = R.build_reference(
            R.tiny_config(candidate_source_layer_id=2, **common), FP16, seed=3
        )
        without = R.build_reference(
            R.tiny_config(candidate_source_layer_id=-1, **common), FP16, seed=3
        )
        a = with_c.forward_chunk(x)
        b = without.forward_chunk(x)
        reachable = torch.ones(1024, 1024, dtype=torch.bool).tril()
        self.assertTrue(a.candidate_mask[reachable].all().item())
        self.assertTrue(torch.equal(a.topk_idx[4], b.topk_idx[4]))
        self.assertTrue(torch.equal(a.topk_idx[2], b.topk_idx[2]))
        self.assertTrue(torch.equal(a.attn_out[5], b.attn_out[5]))

    def test_bf16_score_path_matches_hf_torch_formula(self):
        # HF's torch indexer keeps einsum/relu/weighting/head-sum in bf16.
        g = torch.Generator().manual_seed(22)
        q = R.fake_quant_fp4_ue8m0((torch.randn(16, 4, 128, generator=g)).to(BF16))
        k = R.fake_quant_fp4_ue8m0((torch.randn(40, 128, generator=g)).to(BF16))
        w = (torch.randn(16, 4, generator=g) * 0.05).to(BF16)
        hf = torch.einsum("bhd,td->bht", q, k)
        hf = (hf.relu_() * w.unsqueeze(-1)).sum(dim=1).float()
        self.assertTrue(torch.equal(R.index_scores(q, k, w, score_dtype=BF16), hf))
        # fp32 accumulation (kernel path) vs the bf16 torch path: a bf16-rounding-
        # sized gap relative to the score scale, not an algorithm difference.
        fp32 = R.index_scores(q, k, w)
        gap = (fp32 - hf).abs().max().item() / hf.abs().max().item()
        self.assertLess(gap, 2**-7, f"fp32 vs bf16 indexer score gap {gap}")


class TestSparseAttention(CustomTestCase):
    def test_matches_dense_softmax_with_sink_and_duplicates(self):
        g = torch.Generator().manual_seed(30)
        T, H, D, K = 6, 4, 512, 40
        q = (torch.randn(T, H, D, generator=g) * 0.3).to(FP16)
        keys = R.fake_quant_fp4_e4m3((torch.randn(T, K, D, generator=g)).to(FP16))
        keys[:, 7] = keys[
            :, 3
        ]  # a duplicated key counts twice (SWA + compressed overlap)
        valid = torch.rand(T, K, generator=g) > 0.3
        valid[:, 0] = True
        valid[2, :] = False
        valid[2, 0] = True
        sink = torch.randn(H, generator=g)
        got = R.sparse_attention_rows(q, keys, valid, sink, D**-0.5)
        expect = dense_attention_oracle(q, keys, valid, sink, D**-0.5)
        self.assertTrue(
            torch.allclose(got.float(), expect.float(), atol=1e-2, rtol=1e-2)
        )
        # the sink steals mass: with a huge sink the output vanishes
        tiny = R.sparse_attention_rows(q, keys, valid, torch.full((H,), 30.0), D**-0.5)
        self.assertLess(tiny.float().abs().max().item(), 1e-3)
        # no valid key -> zeros, not NaN
        none = R.sparse_attention_rows(q, keys, torch.zeros_like(valid), sink, D**-0.5)
        self.assertTrue(torch.equal(none, torch.zeros_like(none)))


class TestChunkInvariance(CustomTestCase):
    """The invariant later SM70 decode / prefill kernels are held to."""

    def _compare(self, cfg, one, many, atol=1e-2, max_tie_rows=2):
        """Selections must agree up to exact score ties (the reference does not
        define tie order; neither does HF). Outputs are compared on every row
        whose selections agree for the index source the layer reads."""
        T = one.swa_valid_count.shape[0]
        agree = {}
        for lid in one.topk_idx:
            differing = assert_topk_equivalent(
                self,
                one.topk_idx[lid],
                many.topk_idx[lid],
                one.index_logits[lid],
                msg=f"layer {lid}",
            )
            self.assertLessEqual(
                differing, max_tie_rows, f"layer {lid}: {differing} rows differ"
            )
            agree[lid] = torch.tensor(
                [
                    ra == rb
                    for ra, rb in zip(
                        rows_as_sets(one.topk_idx[lid]),
                        rows_as_sets(many.topk_idx[lid]),
                    )
                ]
            )
        for lid in one.attn_out:
            src = cfg.index_source_for(lid)
            rows = agree[src] if src is not None else torch.ones(T, dtype=torch.bool)
            d = (
                (one.attn_out[lid].float() - many.attn_out[lid].float())
                .abs()[rows]
                .max()
                .item()
            )
            self.assertLess(d, atol, f"layer {lid}: max |diff| {d}")

    def _concat(self, outs):
        merged = R.ChunkOutputs(
            {}, {}, None, {}, torch.cat([o.swa_valid_count for o in outs])
        )
        for lid in outs[0].attn_out:
            merged.attn_out[lid] = torch.cat([o.attn_out[lid] for o in outs])
        for lid in outs[0].topk_idx:
            width = max(o.topk_idx[lid].shape[1] for o in outs)
            merged.topk_idx[lid] = torch.cat(
                [
                    torch.nn.functional.pad(
                        o.topk_idx[lid], (0, width - o.topk_idx[lid].shape[1]), value=-1
                    )
                    for o in outs
                ]
            )
        return merged

    def test_prefill_then_decode_equals_one_prefill_128(self):
        cfg = R.tiny_config(index_topk=32)
        g = torch.Generator().manual_seed(40)
        T = 132
        x = torch.randn(T, cfg.hidden_size, generator=g)
        one = R.build_reference(cfg, FP16).forward_chunk(x)
        ref = R.build_reference(cfg, FP16)
        outs = [ref.forward_chunk(x[:128])]
        for t in range(128, T):
            outs.append(ref.forward_chunk(x[t : t + 1]))
        self._compare(cfg, one, self._concat(outs))
        # window: 128 tokens including the query; decode rows see the full window
        self.assertEqual(one.swa_valid_count[:128].tolist(), list(range(1, 129)))
        self.assertEqual(one.swa_valid_count[128:].tolist(), [128] * 4)
        # ratio-2 visibility at the decode rows: (t+1)//2 groups
        self.assertEqual(one.compress_lens[2][128:].tolist(), [64, 65, 65, 66])
        self.assertEqual(one.compress_lens[1][128:].tolist(), [129, 130, 131, 132])

    def test_chunked_prefill_equals_one_prefill_1024(self):
        cfg = R.tiny_config(index_topk=64)
        g = torch.Generator().manual_seed(41)
        T = 1024
        x = torch.randn(T, cfg.hidden_size, generator=g)
        one = R.build_reference(cfg, FP16).forward_chunk(x)
        ref = R.build_reference(cfg, FP16)
        bounds = [
            0,
            300,
            301,
            640,
            1023,
            1024,
        ]  # odd boundaries exercise the ratio-2 pair state
        outs = [ref.forward_chunk(x[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
        self._compare(cfg, one, self._concat(outs))
        self.assertEqual(one.compress_lens[2][-1].item(), 512)
        self.assertEqual(ref.kv_state[2].num_compressed, 512)
        self.assertEqual(ref.kv_state[20].num_compressed, 1024)
        reachable = torch.ones(T, T, dtype=torch.bool).tril()
        self.assertTrue(
            one.candidate_mask[reachable].all().item()
        )  # everything fits at 1k
        self.assertEqual(one.topk_idx[20].shape, (T, 64))


class TestFp16VersusBf16(CustomTestCase):
    def test_fp16_storage_tracks_bf16_reference(self):
        # Same bf16-exact weights and inputs; only the activation/storage dtype
        # differs. This quantifies the SM70 (fp16) gap against the upstream (bf16) path.
        cfg = R.tiny_config(index_topk=32)
        g = torch.Generator().manual_seed(50)
        x = torch.randn(256, cfg.hidden_size, generator=g).to(BF16).float()
        a = R.build_reference(cfg, FP16, seed=7).forward_chunk(x)
        b = R.build_reference(cfg, BF16, seed=7).forward_chunk(x)
        overlaps, agree = [], {}
        for lid in a.topk_idx:
            same = []
            for ra, rb in zip(
                rows_as_sets(a.topk_idx[lid]), rows_as_sets(b.topk_idx[lid])
            ):
                if ra or rb:
                    overlaps.append(len(ra & rb) / max(len(ra | rb), 1))
                same.append(ra == rb)
            agree[lid] = torch.tensor(same)
        mean_overlap = sum(overlaps) / len(overlaps)
        rel_all, rel_agree = [], []
        for lid in a.attn_out:
            da, db = a.attn_out[lid].float(), b.attn_out[lid].float()
            rel_all.append(((da - db).norm() / db.norm()).item())
            src = cfg.index_source_for(lid)
            rows = (
                agree[src]
                if src is not None
                else torch.ones(da.shape[0], dtype=torch.bool)
            )
            rel_agree.append(((da[rows] - db[rows]).norm() / db[rows].norm()).item())
        print(
            f"\n[fp16 vs bf16] top-k Jaccard {mean_overlap:.4f}; attn out rel L2: "
            f"max {max(rel_all):.4f} (all rows), max {max(rel_agree):.4f} (rows with equal selections)"
        )
        # Selections flip only at near-ties. Where they agree the gap is dominated
        # by FP4 grid flips: a bf16-vs-fp16 rounding difference before the quant
        # moves ~1% of channels across an E2M1 boundary (a whole step), so the
        # attention output sits a few percent apart, not a few tenths of a percent.
        self.assertGreater(mean_overlap, 0.9)
        self.assertLess(max(rel_agree), 0.05)


if __name__ == "__main__":
    unittest.main()
