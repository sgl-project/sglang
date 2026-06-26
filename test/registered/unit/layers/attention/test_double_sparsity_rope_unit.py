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


class TestRopeAwareFailClosedFallback(unittest.TestCase):
    """Selection/capture entry points never silently score no-PE (AC-5). CPU-only.

    The deep ``_select_topk_indices`` per-step guards (bf16 resident KV, missing
    q_pe/positions/rotary_emb, is_nextn, non-decode) are defense-in-depth behind the
    AUTHORITATIVE startup gate (TestRopeAwareStartupGate, which rejects device/graphs/
    spec/DCP/dtype up front); reaching them in a unit test needs the full forward-context
    + graph-state + KV-pool environment, so they are exercised by the serve/integration
    path. The externally-reachable rope entry points are covered here.
    """

    def test_retrieve_topk_graph_safe_fallback_raises_with_rope(self):
        # No scratch + CPU => the eager fallback; it does not score the rope term, so
        # passing q_pe/k_pe must raise rather than return a no-PE selection.
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            retrieve_topk_graph_safe,
        )

        bs, num_heads, label = 1, 2, 4
        common = dict(
            queries=torch.zeros(bs, num_heads, label),
            written=torch.ones(1, 8, dtype=torch.bool),
            channel_selection=torch.zeros(1, num_heads, label, dtype=torch.long),
            channel_weights=torch.ones(1, num_heads, label),
            layer_id=0,
            req_pool_indices=torch.zeros(bs, dtype=torch.int32),
            req_to_token=torch.zeros(bs, 8, dtype=torch.int32),
            seq_lens=torch.full((bs,), 4, dtype=torch.int32),
            max_seq_len=4,
            max_top_k=2,
            out_indices=torch.zeros(bs, 2, dtype=torch.int32),
            out_lengths=torch.zeros(bs, dtype=torch.int32),
            absorbed_w_sel=torch.zeros(1, num_heads, label, label),
        )
        with self.assertRaises(RuntimeError) as ctx:
            retrieve_topk_graph_safe(**common, q_pe=torch.zeros(bs, num_heads, ROPE))
        self.assertIn("rope", str(ctx.exception).lower())

    def test_capture_decode_step_raises_when_rope_on(self):
        from sglang.srt.layers.attention.double_sparsity.cuda_graph import (
            capture_decode_step,
        )

        selector = SimpleNamespace(config=SimpleNamespace(rope_aware_score=True))
        state = SimpleNamespace(max_seq_len=4)
        with self.assertRaises(RuntimeError) as ctx:
            capture_decode_step(
                selector,
                state=state,
                queries=torch.zeros(1, 2, 4),
                layer_id=0,
                req_pool_indices=torch.zeros(1, dtype=torch.int32),
                sparse_mask=torch.ones(1, 4, dtype=torch.bool),
                seq_lens=torch.full((1,), 4, dtype=torch.int32),
            )
        self.assertIn("rope_aware_score", str(ctx.exception))


def _reduce_worker(rank, world_size, port, out_q):
    # Module-level (picklable) worker: init a gloo group, run the production
    # reduce on a per-rank-distinct score row, report the result.
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            reduce_token_scores,
        )

        # rank r contributes [r+1, (r+1)*10]; SUM over 2 ranks = [3, 30].
        scores = torch.tensor(
            [[float(rank + 1), float((rank + 1) * 10)]], dtype=torch.float32
        )
        reduce_token_scores(scores, process_group=dist.group.WORLD, use_bf16=False)
        out_q.put((rank, scores.tolist()))
    finally:
        dist.destroy_process_group()


class TestRopeAwareCrossTPReduce(unittest.TestCase):
    """The cross-rank score reduce is SUM, not MAX/local (AC-4). CPU/gloo.

    The rope term is added per head BEFORE the reduce (kernel test), and the SUM
    reduce composes per-rank partial scores. This proves the reduce is a true SUM:
    a SUM->MAX or SUM->local regression changes the result. (Full TP=8 captured
    selected-index identity stays dev-only per DEC-5; this is the CI guard.)
    """

    def test_reduce_token_scores_is_sum(self):
        import socket

        import torch.multiprocessing as mp

        if not torch.distributed.is_available():
            self.skipTest("torch.distributed unavailable")
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        procs = [
            ctx.Process(target=_reduce_worker, args=(r, 2, port, out_q))
            for r in range(2)
        ]
        for p in procs:
            p.start()
        results = [out_q.get(timeout=60) for _ in range(2)]
        for p in procs:
            p.join(timeout=60)

        # SUM over ranks: [1,10] + [2,20] = [3,30] on every rank.
        for _, row in results:
            self.assertEqual(
                row,
                [[3.0, 30.0]],
                f"reduce is not SUM (got {row}); SUM->MAX/local regression",
            )


class TestRopeSelectionSiteGuards(unittest.TestCase):
    """Selection-site fail-closed preconditions, unit-tested directly via the pure
    helpers _select_topk_indices delegates to (AC-5, selection-site layer). CPU-only.
    """

    def _ok_kwargs(self):
        # The validated config: all preconditions satisfied -> must NOT raise.
        return dict(
            is_nextn=False,
            dcp_world_size=1,
            forward_mode_is_decode=True,
            q_pe=torch.zeros(1, 2, ROPE),
            positions=torch.zeros(1, dtype=torch.int64),
            rotary_emb=object(),  # any non-None stand-in
        )

    def test_supported_config_does_not_raise(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            assert_rope_selection_supported,
        )

        assert_rope_selection_supported(**self._ok_kwargs())  # no raise

    def test_each_unsupported_precondition_raises(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            assert_rope_selection_supported,
        )

        for override in (
            {"is_nextn": True},
            {"dcp_world_size": 2},
            {"forward_mode_is_decode": False},
            {"q_pe": None},
            {"positions": None},
            {"rotary_emb": None},
        ):
            kwargs = self._ok_kwargs()
            kwargs.update(override)
            with self.assertRaises(
                RuntimeError, msg=f"override {override} did not raise"
            ):
                assert_rope_selection_supported(**kwargs)

    def test_fp8_resident_required(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            assert_rope_fp8_resident,
        )

        # bf16 resident KV must raise; fp8 must not.
        with self.assertRaises(RuntimeError):
            assert_rope_fp8_resident(torch.bfloat16)
        assert_rope_fp8_resident(torch.float8_e4m3fn)  # no raise


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
        self.assertGreaterEqual(
            min(recalls), 0.9995, f"top-{TOPK} recall={min(recalls)} (plan ~0.9995)"
        )

    def test_non_power_of_two_rope_dim_rejected(self):
        # ROPE_DIM must be a power of two (tl.arange); q_pe rope_dim=48 must raise.
        q_pe_bad = torch.randn(BS, H, 48, device=self.dev)
        k_pe_bad = torch.randn(T, 48, device=self.dev).to(torch.bfloat16)
        with self.assertRaises(AssertionError):
            self._kernel(q_pe=q_pe_bad, k_pe=k_pe_bad)

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

    def test_off_selected_indices_match_no_rope(self):
        # AC-1: with the flag off, the SELECTED top-k indices are identical to a
        # no-rope launch (the byte-identical scores produce an identical selection).
        k_off = self._kernel()
        k_off2 = self._kernel(q_pe=None, k_pe=None)
        for b in range(BS):
            sl = int(self.seq_lens[b])
            self.assertEqual(_topk_set(k_off[b], sl), _topk_set(k_off2[b], sl))

    def test_half_wired_rope_pair_rejected(self):
        # AC-5: a half-wired rope pair (exactly one of q_pe/k_pe) must raise, not
        # silently launch no-PE.
        with self.assertRaises(ValueError):
            self._kernel(q_pe=self.q_pe)  # k_pe missing
        with self.assertRaises(ValueError):
            self._kernel(k_pe=self.k_pe)  # q_pe missing

    def test_padded_heads_no_rope_leak(self):
        # AC-2 negative: with a non-power-of-two head count (6 -> H_POW2=8, 2 pad
        # heads), the rope-ON kernel still matches the eager 6-head reference — the
        # pad heads contribute zero rope (h_mask), no leak into the head-max.
        h6 = 6
        torch.manual_seed(1)
        v6 = torch.randn(BS, h6, LORA, device=self.dev)
        q_pe6 = torch.randn(BS, h6, ROPE, device=self.dev)
        k_on = self.fn(
            v6,
            self.latent_fp8,
            self.scales,
            self.rpi,
            self.rtt,
            self.seq_lens,
            SEQ,
            head_agg="max",
            q_pe=q_pe6,
            k_pe=self.k_pe,
        )
        e_on = _eager_score(
            v6,
            self.latent_fp8,
            self.scales,
            q_pe6,
            self.k_pe,
            self.rtt,
            self.rpi,
            self.seq_lens,
            True,
            self.dev,
        )
        cos = min(_cos(k_on[b], e_on[b]) for b in range(BS))
        self.assertGreaterEqual(cos, 0.999, f"padded-head rope leak: cos={cos}")


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

        def _snapshot():
            st = torch.cuda.memory_stats()
            return (
                torch.cuda.memory_allocated(),
                torch.cuda.memory_reserved(),
                st.get("allocation.all.allocated", 0),
                st.get("segment.all.allocated", 0),
            )

        before = _snapshot()
        for _ in range(100):
            g.replay()
        torch.cuda.synchronize()
        after = _snapshot()
        # AC-3: not just bytes flat — the allocator must perform zero new
        # allocations and zero new segments over the replays.
        labels = ("memory_allocated", "memory_reserved", "alloc_count", "segment_count")
        for name, b, a in zip(labels, before, after):
            self.assertEqual(a, b, f"{name} changed across 100 replays: {b} -> {a}")


if __name__ == "__main__":
    unittest.main()
