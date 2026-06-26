"""Unit tests for the RoPE-aware Double Sparsity selection score.

Covers the ported recovery fix that adds the RoPE term ``q_rope·k_pe[t]`` to the
absorbed selection score, behind the off-by-default ``rope_aware_score`` flag:

- Config layer (CPU): the flag defaults off, parses, is allowed, must be a bool, and
  requires ``scorer_norm="off"`` (the fail-closed config rules).
- Kernel (CUDA): the rope-ON graph-safe kernel matches an eager fp32 reference
  (no-PE absorbed dot + rope, head-agg max); the rope-OFF launch is byte-identical to
  a launch with no ``q_pe``/``k_pe``; the knob is non-vacuous.
- Graph safety (CUDA): the rope region is zero-new-allocation under CUDA-graph replay.

The full cross-TP SUM identity (post-reduce == sum-over-ranks of pre-reduce) needs a
real TP=8 run and is validated by the development-only captured-TP harness; the SUM
reduce path itself is unchanged by this fix, and the kernel test below confirms the
rope term is added per head BEFORE the head reduce (the property the SUM composes over).
"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.double_sparsity import validate_double_sparsity
from sglang.srt.layers.attention.double_sparsity.config import (
    DoubleSparsityConfig,
    _ALLOWED_FIELDS,
    parse_double_sparsity_config,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()

# Real-shaped paged inputs (GLM-5.1-FP8 selection shapes): 8 heads, 512 latent, 64 rope.
BS, H, LORA, ROPE, BLK = 3, 8, 512, 64, 128
NBLK = LORA // BLK
SEQ, TOPK = 4096, 2048
T = BS * SEQ + 128


class TestRopeAwareConfig(unittest.TestCase):
    """Config-layer fail-closed rules (AC-5, config layer). CPU-only."""

    def test_default_off(self):
        c = DoubleSparsityConfig(channel_mask_path="x")
        self.assertIs(c.rope_aware_score, False)

    def test_allowed_field_present(self):
        self.assertIn("rope_aware_score", _ALLOWED_FIELDS)

    def test_parse_roundtrip(self):
        c = parse_double_sparsity_config(
            '{"channel_mask_path":"x","scorer_norm":"off","rope_aware_score":true}'
        )
        self.assertIs(c.rope_aware_score, True)

    def test_non_bool_rejected(self):
        with self.assertRaises(ValueError):
            DoubleSparsityConfig(
                channel_mask_path="x", scorer_norm="off", rope_aware_score="yes"
            )

    def test_rope_requires_scorer_norm_off(self):
        # cosine + rope is not a defined scorer -> must fail closed.
        with self.assertRaises(ValueError):
            DoubleSparsityConfig(
                channel_mask_path="x", scorer_norm="cosine", rope_aware_score=True
            )

    def test_rope_with_off_is_accepted(self):
        c = DoubleSparsityConfig(
            channel_mask_path="x", scorer_norm="off", rope_aware_score=True
        )
        self.assertIs(c.rope_aware_score, True)
        self.assertEqual(c.scorer_norm, "off")


def _rope_payload(**extra):
    d = {
        "channel_mask_path": "/tmp/cm.safetensors",
        "scorer_norm": "off",
        "rope_aware_score": True,
        "page_size": 64,
    }
    d.update(extra)
    return json.dumps(d)


def _server_args(**kwargs):
    defaults = dict(
        enable_double_sparsity=True,
        enable_hisparse=False,
        enable_hierarchical_cache=False,
        disaggregation_mode=None,
        double_sparsity_config=_rope_payload(),
        page_size=64,
        device="cuda",
        disable_cuda_graph=False,
        speculative_algorithm=None,
        dcp_size=1,
        kv_cache_dtype="fp8_e4m3",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestRopeAwareStartupGate(unittest.TestCase):
    """Init/startup fail-closed gate in validate_double_sparsity (AC-5, init layer).

    Covers the scope leaks the in-loop selection-site guard cannot reach (CPU/NPU and
    graphs-off decode bypass _select_topk_indices entirely). CPU-only.
    """

    def setUp(self):
        # validate_double_sparsity reaches a model-capability check only AFTER the rope
        # gate; the rope gate fires first, so the adapter env is a harmless safety net.
        os.environ["SGLANG_DS_ALLOW_NO_ADAPTER"] = "1"

    def tearDown(self):
        os.environ.pop("SGLANG_DS_ALLOW_NO_ADAPTER", None)

    def test_graphs_off_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(_server_args(disable_cuda_graph=True))
        self.assertIn("rope_aware_score", str(ctx.exception))
        self.assertIn("graph", str(ctx.exception).lower())

    def test_non_cuda_device_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(_server_args(device="cpu"))
        self.assertIn("rope_aware_score", str(ctx.exception))

    def test_speculative_rejected(self):
        # include_current_slot=false so the rope gate (not the current-slot gate) fires.
        args = _server_args(
            double_sparsity_config=_rope_payload(include_current_slot=False),
            speculative_algorithm="EAGLE",
        )
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(args)
        self.assertIn("rope_aware_score", str(ctx.exception))

    def test_dcp_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(_server_args(dcp_size=2))
        self.assertIn("rope_aware_score", str(ctx.exception))

    def test_bf16_kv_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(_server_args(kv_cache_dtype="bfloat16"))
        self.assertIn("rope_aware_score", str(ctx.exception))


def _dequant(latent_fp8, scales, phys):
    g = latent_fp8[phys].to(torch.float32)
    s = scales[phys].to(torch.float32)
    return (g.view(-1, NBLK, BLK) * s.view(-1, NBLK, 1)).view(-1, LORA)


def _eager_score(v, latent_fp8, scales, q_pe, k_pe, rtt, rpi, seq_lens, use_rope, dev):
    out = torch.full((BS, SEQ), float("-inf"), device=dev)
    for b in range(BS):
        sl = int(seq_lens[b])
        phys = rtt[rpi[b], :sl].long()
        c_kv = _dequant(latent_fp8, scales, phys)
        acc = torch.einsum("hl,sl->sh", v[b], c_kv)
        if use_rope:
            kpe = k_pe[phys].to(torch.float32)
            acc = acc + torch.einsum("hr,sr->sh", q_pe[b].float(), kpe)
        out[b, :sl] = acc.amax(dim=-1)
    return out


def _cos(a, b):
    m = torch.isfinite(a) & torch.isfinite(b)
    x, y = a[m].double(), b[m].double()
    return float((x @ y) / (x.norm() * y.norm() + 1e-12))


def _topk_set(scores, sl):
    return set(torch.topk(scores[:sl], min(TOPK, sl)).indices.tolist())


@unittest.skipUnless(_HAS_CUDA, "rope-aware DS kernel requires CUDA/Triton")
class TestRopeAwareKernel(unittest.TestCase):
    """Kernel numeric correctness + off byte-identical (AC-1, AC-2, AC-4 additivity)."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.double_sparsity.absorbed_latent_kernel import (
            absorbed_score_paged_fp8,
        )

        cls.fn = staticmethod(absorbed_score_paged_fp8)
        dev = "cuda"
        torch.manual_seed(0)
        cls.dev = dev
        cls.v = torch.randn(BS, H, LORA, device=dev)
        cls.latent_fp8 = (torch.randn(T, LORA, device=dev) * 0.3).to(
            torch.float8_e4m3fn
        )
        cls.scales = torch.rand(T, NBLK, device=dev) * 0.1 + 0.05
        cls.q_pe = torch.randn(BS, H, ROPE, device=dev)
        cls.k_pe = torch.randn(T, ROPE, device=dev).to(torch.bfloat16)
        perm = torch.randperm(T, device=dev)
        rtt = torch.zeros(BS, SEQ, dtype=torch.int32, device=dev)
        for b in range(BS):
            rtt[b] = perm[b * SEQ : (b + 1) * SEQ].to(torch.int32)
        cls.rtt = rtt
        cls.rpi = torch.arange(BS, dtype=torch.int32, device=dev)
        cls.seq_lens = torch.tensor(
            [SEQ, SEQ - 7, SEQ - 64], dtype=torch.int32, device=dev
        )

    def _kernel(self, **kw):
        return self.fn(
            self.v,
            self.latent_fp8,
            self.scales,
            self.rpi,
            self.rtt,
            self.seq_lens,
            SEQ,
            head_agg="max",
            **kw,
        )

    def test_rope_on_matches_eager(self):
        k_on = self._kernel(q_pe=self.q_pe, k_pe=self.k_pe)
        e_on = _eager_score(
            self.v,
            self.latent_fp8,
            self.scales,
            self.q_pe,
            self.k_pe,
            self.rtt,
            self.rpi,
            self.seq_lens,
            True,
            self.dev,
        )
        cos_on = min(_cos(k_on[b], e_on[b]) for b in range(BS))
        self.assertGreaterEqual(cos_on, 0.999, f"cos(rope-ON vs eager)={cos_on}")
        recalls = []
        for b in range(BS):
            sl = int(self.seq_lens[b])
            recalls.append(
                len(_topk_set(k_on[b], sl) & _topk_set(e_on[b], sl))
                / len(_topk_set(e_on[b], sl))
            )
        self.assertGreaterEqual(min(recalls), 0.99, f"top-{TOPK} recall={min(recalls)}")

    def test_off_byte_identical(self):
        # AC-1: rope-OFF launch == launch with no q_pe/k_pe == eager no-PE-only.
        k_off = self._kernel()
        k_off2 = self._kernel(q_pe=None, k_pe=None)
        self.assertTrue(torch.equal(k_off, k_off2))
        e_off = _eager_score(
            self.v,
            self.latent_fp8,
            self.scales,
            self.q_pe,
            self.k_pe,
            self.rtt,
            self.rpi,
            self.seq_lens,
            False,
            self.dev,
        )
        cos_off = min(_cos(k_off[b], e_off[b]) for b in range(BS))
        self.assertGreaterEqual(cos_off, 0.999, f"cos(rope-OFF vs eager)={cos_off}")

    def test_knob_is_non_vacuous(self):
        # AC-2 negative: rope-ON selection must differ from rope-OFF.
        k_on = self._kernel(q_pe=self.q_pe, k_pe=self.k_pe)
        k_off = self._kernel()
        differ = any(
            _topk_set(k_on[b], int(self.seq_lens[b]))
            != _topk_set(k_off[b], int(self.seq_lens[b]))
            for b in range(BS)
        )
        self.assertTrue(differ, "rope-ON and rope-OFF selections are identical")


@unittest.skipUnless(_HAS_CUDA, "rope-aware DS graph audit requires CUDA")
class TestRopeAwareGraphAlloc(unittest.TestCase):
    """Zero new allocations under CUDA-graph replay (AC-3)."""

    @staticmethod
    def _rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    def test_zero_alloc_replay(self):
        from sglang.srt.layers.attention.double_sparsity.absorbed_latent_kernel import (
            absorbed_score_paged_fp8,
        )

        dev = "cuda"
        torch.manual_seed(0)
        v = torch.randn(BS, H, LORA, device=dev)
        latent_fp8 = (torch.randn(T, LORA, device=dev) * 0.3).to(torch.float8_e4m3fn)
        scales = torch.rand(T, NBLK, device=dev) * 0.1 + 0.05
        q_pe = torch.randn(BS, H, ROPE, device=dev)
        k_pe = torch.randn(T, ROPE, device=dev).to(torch.bfloat16)
        cos = torch.randn(BS, 1, ROPE, device=dev)
        sin = torch.randn(BS, 1, ROPE, device=dev)
        perm = torch.randperm(T, device=dev)
        rtt = torch.zeros(BS, SEQ, dtype=torch.int32, device=dev)
        for b in range(BS):
            rtt[b] = perm[b * SEQ : (b + 1) * SEQ].to(torch.int32)
        rpi = torch.arange(BS, dtype=torch.int32, device=dev)
        seq_lens = torch.tensor([SEQ, SEQ - 7, SEQ - 64], dtype=torch.int32, device=dev)
        out_buf = torch.empty((BS, SEQ), dtype=torch.float32, device=dev)

        def region():
            _qp = q_pe.contiguous()
            c = _qp.clone()
            z = torch.zeros_like(_qp[:, :1, :])
            q_rot = c * cos + self._rotate_half(c) * sin
            q_rot = q_rot + z.sum() * 0.0
            absorbed_score_paged_fp8(
                v,
                latent_fp8,
                scales,
                rpi,
                rtt,
                seq_lens,
                SEQ,
                head_agg="max",
                q_pe=q_rot,
                k_pe=k_pe,
                out=out_buf,
            )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                region()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            region()
        torch.cuda.synchronize()

        m0 = torch.cuda.memory_allocated()
        for _ in range(100):
            g.replay()
        torch.cuda.synchronize()
        m1 = torch.cuda.memory_allocated()
        self.assertEqual(m1 - m0, 0, f"grew {m1 - m0} bytes over 100 replays")


if __name__ == "__main__":
    unittest.main()
