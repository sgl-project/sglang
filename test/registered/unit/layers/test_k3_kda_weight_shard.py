"""Kimi-K3 KDA weight sharding for attention-DP prefill (--kda-weight-shard-size).

CPU part (N simulated ranks in one process, synchronous copy_ mover): head-aligned shard
layout (rank r's shard == tensor-parallel rank r's slices, as KDA TP loads them),
load/gather roundtrip (bit-identical to the replicated weights for N = 2, 4, 8),
missing/doubled slices fail check_loaded, the gather schedule over three forwards of the K3
layer pattern (93 layers, 69 KDA) with prefetch depth 1 and 2, issue-at-core, the
shared-expert store next to the KDA store, and the MoE-stage wait modes
(SGLANG_K3_KDA_WS_MOE_WAIT).

CUDA part: the same schedule with real streams on one GPU (N fake ranks whose shards live
on one device): a GEMM chain that reads each layer's weight views on the compute stream
while gathers run on the copy stream is bit-identical to the chain over replicated
weights; the gather launches no kernels (copy engine only, the o_proj column slices as
2-D copies); K3-sized shards gather bitwise from cuMem-allocated peers.
"""

import os
import random
import types
import unittest
import unittest.mock
from unittest import TestCase, skipUnless

import torch

from sglang.srt.layers.communication import k3_kda_weight_shard as ws
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

H, P = 64, 48  # small K3-shaped layer: qkvg (4P x H), o_proj (H x P)
SHAPES = ws.kda_unit_shapes(P, H)


def k3_kda_layers(num_layers=93):
    full = [4 * i for i in range(1, 24)] + [93]  # 1-indexed full-attention layers
    return [i for i in range(num_layers) if (i + 1) not in full]


def make_group(
    n,
    kda_ids,
    num_layers,
    device="cpu",
    dtype=torch.float32,
    depth=1,
    shapes=SHAPES,
    **kw,
):
    shards = {}
    alloc_fn = kw.pop("alloc", None)

    def alloc_for(r):
        def alloc(numel):
            shards[r] = (
                alloc_fn(numel)
                if alloc_fn
                else torch.zeros(numel, dtype=dtype, device=device)
            )
            return shards[r]

        return alloc

    ranks = []
    for r in range(n):
        ranks.append(
            ws.LayerWeightShard(
                units=[(layer, shapes) for layer in kda_ids],
                num_layers=num_layers,
                group_rank=r,
                group_size=n,
                dtype=dtype,
                device=torch.device(device),
                alloc_shard=alloc_for(r),
                # resolved lazily: peers are read only at gather time
                peer_shards=lambda t: _LazyPeers(shards, n),
                prefetch_depth=depth,
                **kw,
            )
        )
    return ranks


def vmm_alloc(device, dtype):
    """One cuMem allocation per shard, as ws._vmm_alloc_and_peers makes (its single-process
    half)."""

    def alloc(numel):
        vmm = ws._Vmm(torch.device(device))
        es = torch.empty((), dtype=dtype).element_size()
        g = vmm.granularity()
        nb = -(-numel * es // g) * g
        return vmm.map_flat([vmm.create(nb)], [nb], dtype)[:numel].zero_()

    return alloc


class _LazyPeers(list):
    def __init__(self, shards, n):
        super().__init__([None] * n)
        self._shards = shards
        self.read = set()  # ranks whose shard a gather read

    def __getitem__(self, p):
        self.read.add(p)
        return self._shards[p]


def full_weights(num_kda, seed=0, device="cpu", dtype=torch.float32, shapes=SHAPES):
    if device != "cpu":
        g = torch.Generator(device=device).manual_seed(seed)
        return [
            {
                name: torch.randn(shape, generator=g, device=device).to(dtype)
                for name, shape, *_ in shapes
            }
            for _ in range(num_kda)
        ]
    g = torch.Generator().manual_seed(seed)
    return [
        {
            name: torch.randn(shape, generator=g).to(device=device, dtype=dtype)
            for name, shape, *_ in shapes
        }
        for _ in range(num_kda)
    ]


def load_all(rank, weights, order_seed=0):
    rng = random.Random(order_seed)
    for k, w in enumerate(weights):
        pieces = [("qkvg", i) for i in range(4)] + [("o_proj", None)]
        rng.shuffle(pieces)
        p = w["qkvg"].shape[0] // 4
        for name, sid in pieces:
            if name == "qkvg":
                rank.load(
                    k, "qkvg", w["qkvg"][sid * p : (sid + 1) * p], row_offset=sid * p
                )
            else:
                rank.load(k, "o_proj", w["o_proj"])


K3_SHAPES = ws.kda_unit_shapes(12288, 7168)


class TestLayout(TestCase):
    def test_alignment(self):
        for n in (2, 4, 8):
            specs, size = ws.layer_layout(SHAPES)
            pieces, shard = ws.unit_pieces(specs, n)
            for s in specs:
                self.assertEqual(s.offset % ws.ALIGN, 0)
            for pc in pieces:
                self.assertEqual(pc.off % ws.ALIGN, 0)
            self.assertGreaterEqual(size, specs[-1].offset + specs[-1].numel)
            self.assertEqual(sum(pc.rows * pc.cols for pc in pieces) * n, 5 * P * H)

    def test_k3_sizes(self):
        specs, size = ws.layer_layout(K3_SHAPES)
        self.assertEqual(specs[1].offset, 49152 * 7168)
        self.assertEqual(size, 49152 * 7168 + 7168 * 12288)
        pieces, shard = ws.unit_pieces(specs, 8)
        self.assertEqual(shard, size // 8)  # no padding at K3 sizes
        self.assertEqual(69 * shard * 2, 7_596_933_120)  # shard bytes per rank
        self.assertEqual(
            [(pc.name, pc.block, pc.rows, pc.cols) for pc in pieces],
            [("qkvg", b, 1536, 7168) for b in range(4)] + [("o_proj", 0, 7168, 1536)],
        )

    def test_splits_must_divide(self):
        with self.assertRaisesRegex(ValueError, "do not split"):
            ws.unit_pieces(ws.layer_layout([("qkvg", (4 * 6, H), "rows", 4)])[0], 4)
        with self.assertRaisesRegex(ValueError, "columns do not split"):
            ws.unit_pieces(ws.layer_layout([("o_proj", (H, 6), "cols", 1)])[0], 4)


class TestTensorParallelSlices(TestCase):
    """Rank r's shard is exactly tensor-parallel rank r's KDA weights: the [q|k|v|g]
    in-projection rows MergedColumnParallelLinear.weight_loader keeps, and the o_proj
    columns RowParallelLinear.weight_loader keeps (the loaders KDA TP's layer uses, with
    rank r of N), so KDA TP rank r could read its shard in place."""

    def test_shard_equals_tensor_parallel_rank(self):
        torch.manual_seed(1)
        F = P
        for n in (2, 4, 8):
            ranks = make_group(n, [0], 1)
            w = full_weights(1, seed=n)[0]
            for rk in ranks:
                load_all(rk, [w])
                rk.check_loaded()
            for r, rk in enumerate(ranks):
                f = F // n
                # Tensor parallelism's slicing: rank r keeps rows [r*f, (r+1)*f) of each of
                # q, k, v, g (MergedColumnParallelLinear, shard ids 0..3) and o_proj columns
                # [r*f, (r+1)*f) (RowParallelLinear, input dim).
                tp_qkvg = torch.cat(
                    [w["qkvg"][i * F + r * f : i * F + (r + 1) * f] for i in range(4)]
                )
                tp_o = w["o_proj"][:, r * f : (r + 1) * f]
                self.assertTrue(torch.equal(rk.own_view(0, "qkvg"), tp_qkvg), (n, r))
                self.assertTrue(torch.equal(rk.own_view(0, "o_proj"), tp_o), (n, r))

    def test_matches_the_model_tp_loaders(self):
        # The same through sglang's tensor-parallel weight loaders (the KDA TP layer's
        # fused_qkvg_proj / o_proj), with rank r of N.
        from sglang.srt.layers.linear import (
            MergedColumnParallelLinear,
            RowParallelLinear,
        )

        F = P
        n = 4
        ranks = make_group(n, [0], 1)
        w = full_weights(1, seed=3)[0]
        for rk in ranks:
            load_all(rk, [w])
        for r, rk in enumerate(ranks):
            par = types.SimpleNamespace(tp_group=None, tp_size=n, tp_rank=r)
            with unittest.mock.patch(
                "sglang.srt.layers.linear.get_parallel", return_value=par
            ):
                qkvg = MergedColumnParallelLinear(
                    H, [F] * 4, bias=False, params_dtype=torch.float32
                )
                o = RowParallelLinear(
                    F, H, bias=False, reduce_results=False, params_dtype=torch.float32
                )
                for i in range(4):
                    qkvg.weight.weight_loader(
                        qkvg.weight, w["qkvg"][i * F : (i + 1) * F], i
                    )
                o.weight.weight_loader(o.weight, w["o_proj"])
            self.assertTrue(torch.equal(rk.own_view(0, "qkvg"), qkvg.weight.data))
            self.assertTrue(torch.equal(rk.own_view(0, "o_proj"), o.weight.data))


class TestGroupLayout(TestCase):
    """Shard groups: runs of consecutive ranks under attention TP 1; under attention TP > 1
    the DP ranks holding the same attention-TP slice."""

    def _layout(self, *args):
        from sglang.srt.layers.communication.k3_kda_weight_shard import (
            shard_group_layout,
        )

        return shard_group_layout(*args)

    def test_attention_tp1_matches_consecutive_runs(self):
        ranks = list(range(8))
        for size in (2, 4, 8):
            for r in ranks:
                members, g, gr = self._layout(ranks, 1, size, r)
                self.assertEqual(
                    members, [ranks[i : i + size] for i in range(0, 8, size)]
                )
                self.assertEqual((g, gr), (r // size, r % size))

    def test_attention_tp2_groups_share_a_slice(self):
        ranks = list(range(8))
        members, _, _ = self._layout(ranks, 2, 4, 0)
        self.assertEqual(members, [[0, 2, 4, 6], [1, 3, 5, 7]])
        for r in ranks:
            _, g, gr = self._layout(ranks, 2, 4, r)
            self.assertEqual(members[g][gr], r)
        members, _, _ = self._layout(ranks, 2, 2, 0)
        self.assertEqual(members, [[0, 2], [4, 6], [1, 3], [5, 7]])
        for r in ranks:
            _, g, gr = self._layout(ranks, 2, 2, r)
            self.assertEqual(members[g][gr], r)

    def test_size_must_divide_attention_dp(self):
        with self.assertRaises(ValueError):
            self._layout(list(range(8)), 2, 8, 0)


class TestLoadGather(TestCase):
    def _roundtrip(self, n):
        kda = [0, 1, 2, 4, 5, 6]
        ranks = make_group(n, kda, 8)
        w = full_weights(len(kda))
        for r, rank in enumerate(ranks):
            load_all(rank, w, order_seed=r)
            rank.check_loaded()
        for rank in ranks:
            for k in range(len(kda)):
                rank.wait(k)
                for name, *_ in SHAPES:
                    self.assertTrue(
                        torch.equal(rank.param_view(k, name), w[k][name]), (n, k, name)
                    )

    def test_roundtrip(self):
        for n in (2, 4, 8):
            self._roundtrip(n)

    def test_own_pieces_come_from_the_own_shard(self):
        # No replica: a rank stores only its own pieces and copies them from its own shard
        # last; every peer's shard is read once per gather.
        ranks = make_group(4, [0], 1)
        w = full_weights(1)
        for rk in ranks:
            load_all(rk, w)
        self.assertEqual(ranks[2].shard.numel(), ranks[2].shard_size[0])
        ops = ranks[2]._copy_ops(0)
        own = [
            op
            for op in ops
            if op[1].data_ptr() >= ranks[2].shard.data_ptr()
            and op[1].data_ptr()
            < ranks[2].shard.data_ptr() + ranks[2].shard.numel() * 4
        ]
        self.assertEqual(len(own), len(ranks[2].pieces[0]))
        self.assertEqual(ops[-len(own) :], own)
        ranks[2].peers.read.clear()
        ranks[2].wait(0)
        self.assertEqual(ranks[2].peers.read, {0, 1, 3})
        for name, *_ in SHAPES:
            self.assertTrue(torch.equal(ranks[2].param_view(0, name), w[0][name]))
        ranks[2].shard.zero_()
        ranks[2]._pending.clear()
        ranks[2]._slot_owner = [None] * ranks[2].num_slots
        ranks[2]._running = None
        ranks[2].wait(0)
        self.assertFalse(torch.equal(ranks[2].param_view(0, "qkvg"), w[0]["qkvg"]))

    def test_missing_and_double_loads_fail(self):
        kda = [0, 1]
        ranks = make_group(4, kda, 2)
        w = full_weights(2)
        load_all(ranks[1], w)
        ranks[1].check_loaded()
        # missing o_proj of layer 1 on rank 3 (o_proj sits in the tail shards)
        r3 = ranks[3]
        for k in range(2):
            for i in range(4):
                r3.load(k, "qkvg", w[k]["qkvg"][i * P : (i + 1) * P], row_offset=i * P)
        r3.load(0, "o_proj", w[0]["o_proj"])
        with self.assertRaises(RuntimeError):
            r3.check_loaded()
        r3.load(1, "o_proj", w[1]["o_proj"])
        r3.check_loaded()
        r3.load(1, "o_proj", w[1]["o_proj"])
        with self.assertRaises(RuntimeError):
            r3.check_loaded()

    def test_bad_shapes_rejected(self):
        ranks = make_group(2, [0], 1)
        with self.assertRaises(ValueError):
            ranks[0].load(0, "qkvg", torch.zeros(P, H + 1), row_offset=0)
        with self.assertRaises(ValueError):
            ranks[0].load(0, "qkvg", torch.zeros(P, H), row_offset=4 * P)


class TestMixedUnits(TestCase):
    def test_heterogeneous_units_roundtrip(self):
        # Units of different shapes (e.g. KDA and a later MLA registration) share the two
        # slots, which are sized to the largest unit.
        small = [("a", (40, 24), "rows", 1), ("b", (24, 40), "cols", 1)]
        units = [(0, SHAPES), (1, small), (2, SHAPES), (4, small)]
        n = 4
        shards = {}

        def alloc_for(r):
            def alloc(numel):
                shards[r] = torch.zeros(numel)
                return shards[r]

            return alloc

        ranks = [
            ws.LayerWeightShard(
                units=units,
                num_layers=5,
                group_rank=r,
                group_size=n,
                dtype=torch.float32,
                device=torch.device("cpu"),
                alloc_shard=alloc_for(r),
                peer_shards=lambda t: _LazyPeers(shards, n),
            )
            for r in range(n)
        ]
        g = torch.Generator().manual_seed(7)
        w = [
            {name: torch.randn(shape, generator=g) for name, shape, *_ in sh}
            for _, sh in units
        ]
        for rank in ranks:
            for u, (_, sh) in enumerate(units):
                for name, *_ in sh:
                    rank.load(u, name, w[u][name])
            rank.check_loaded()
        self.assertEqual(ranks[0].slots.shape[1], max(ranks[0].e_size))
        for rank in ranks:
            for layer in range(5):
                rank.on_layer_start(layer)
                u = rank.unit_index.get(layer)
                if u is not None:
                    for name, *_ in units[u][1]:
                        self.assertTrue(
                            torch.equal(rank.param_view(u, name), w[u][name])
                        )


class TestSchedule(TestCase):
    def _run(self, num_layers, kda, forwards=3, n=4, prefetched=1, depth=1):
        ranks = make_group(n, kda, num_layers, depth=depth)
        w = full_weights(len(kda), seed=1)
        for rank in ranks:
            load_all(rank, w)
        rank = ranks[2]
        kidx = {l: k for k, l in enumerate(kda)}
        issued_at = {}  # unit -> decoder layer whose start issued it (this forward)
        orig_issue = rank.issue

        def issue(u, _orig=orig_issue):
            if u not in rank._pending:
                issued_at[u] = cur_layer[0]
            _orig(u)

        rank.issue = issue
        cur_layer = [None]
        for f in range(forwards):
            waited = []
            orig_wait = rank.wait

            def wait(k, _orig=orig_wait):
                waited.append(k)
                _orig(k)

            rank.wait = wait
            for layer in range(num_layers):
                cur_layer[0] = layer
                rank.on_layer_start(layer)
                if layer in kidx:
                    k = kidx[layer]
                    for name, *_ in SHAPES:
                        self.assertTrue(
                            torch.equal(rank.param_view(k, name), w[k][name]),
                            (f, layer),
                        )
                    self.assertNotIn(k, rank.pending)
                self.assertLessEqual(len(rank.pending), depth)
                self.assertEqual(
                    len({rank.slot_of(p) for p in rank.pending}), len(rank.pending)
                )
            rank.wait = orig_wait
            self.assertEqual(waited, list(range(len(kda))))
            # The next forward's first KDA layer is already in flight (unless the last
            # decoder layer is a KDA layer sharing its slot).
            self.assertEqual(rank.pending, list(range(prefetched)))
            # Every gather of a unit issued in this forward started at a layer start no
            # more than `depth` sharded layers before it (prefetch depth honored).
            for u, layer in issued_at.items():
                if layer is not None and layer <= kda[u]:
                    self.assertLessEqual(
                        sum(1 for l in kda if layer <= l < kda[u]), depth
                    )
            issued_at.clear()

    def test_k3_pattern(self):
        kda = k3_kda_layers(93)
        self.assertEqual(len(kda), 69)
        self._run(93, kda)

    def test_truncated_pattern(self):
        self._run(8, k3_kda_layers(8))

    def test_kda_last_layer(self):
        # last decoder layer is KDA: the wrap prefetch falls to the next forward's start
        self._run(4, [0, 2, 3], prefetched=0)

    def test_k3_pattern_depth2(self):
        kda = k3_kda_layers(93)
        self._run(93, kda, depth=2, prefetched=2)

    def test_depth2_lead_time(self):
        # Steady state (second forward on): every KDA gather is in flight for at least
        # `depth` attention windows (layer starts) before its layer runs, so with depth 2
        # the MLA layer's window also carries gather traffic.
        kda = k3_kda_layers(93)
        for depth in (1, 2):
            ranks = make_group(4, kda, 93, depth=depth)
            for rk in ranks:
                load_all(rk, full_weights(len(kda), seed=2))
            rank = ranks[0]
            step, issued_step = 0, {}
            for f in range(2):
                for layer in range(93):
                    before = set(rank.pending)
                    rank.on_layer_start(layer)
                    for u in set(rank.pending) - before:
                        issued_step[u] = step
                    if f == 1 and layer in rank.unit_index:
                        u = rank.unit_index[layer]
                        self.assertGreaterEqual(
                            step - issued_step[u], depth, (depth, layer)
                        )
                    step += 1

    def test_truncated_pattern_depth2(self):
        self._run(8, k3_kda_layers(8), depth=2, prefetched=2)


class TestIssueAtCore(TestCase):
    """SGLANG_K3_KDA_WS_ISSUE_AT=core: a KDA layer issues the next layer's gather at its
    core (after the in-projection), non-KDA layers at their start; MoE-start waits on what
    the layer issued; every KDA layer still finds exactly its weights."""

    def _run(self, kda_units, num_layers=93, forwards=3, depth=1):
        ranks = make_group(4, kda_units, num_layers, depth=depth, issue_at="core")
        w = full_weights(len(kda_units), seed=4)
        for rk in ranks:
            load_all(rk, w)
        rank = ranks[1]
        self.assertTrue(all(rank.needs_layer_hook(l) for l in range(num_layers)))
        for f in range(forwards):
            for layer in range(num_layers):
                before = set(rank.pending)
                rank.on_layer_start(layer)
                u = rank.unit_index.get(layer)
                if u is not None:
                    # nothing new issued at a KDA layer's start in core mode
                    self.assertTrue(set(rank.pending) <= before)
                    for name, *_ in SHAPES:
                        self.assertTrue(
                            torch.equal(rank.param_view(u, name), w[u][name]),
                            (f, layer),
                        )
                    rank.on_core_start(layer)
                rank.on_moe_start(layer)
                self.assertLessEqual(len(rank.pending), depth)

    def test_k3_pattern(self):
        self._run(k3_kda_layers(93))

    def test_k3_pattern_depth2(self):
        self._run(k3_kda_layers(93), depth=2)


class TestSharedExpertStore(TestCase):
    """KDA store + shared-expert store (one unit per MoE layer) driven together by the
    module-level hooks, as KimiK3DecoderLayer does."""

    I = 40
    SHARED = ws.shared_expert_unit_shapes(40, H)

    def _stores(self, n=4):
        kda = k3_kda_layers()
        moe = list(range(1, 93))
        kda_ranks = make_group(n, kda, 93)
        sh_ranks = make_group(n, moe, 93, shapes=self.SHARED)
        wk = full_weights(len(kda), seed=3)
        wsh = full_weights(len(moe), seed=4, shapes=self.SHARED)
        for r in kda_ranks:
            load_all(r, wk)
        for r in sh_ranks:
            for k, w in enumerate(wsh):
                for sid in (1, 0):  # up before gate: any order
                    r.load(
                        k,
                        "gate_up",
                        w["gate_up"][sid * self.I : (sid + 1) * self.I],
                        row_offset=sid * self.I,
                    )
                r.load(k, "down", w["down"])
        return kda, moe, kda_ranks, sh_ranks, wk, wsh

    def setUp(self):
        self._saved = (ws._ACTIVE, ws._ACTIVE_SHARED)

    def tearDown(self):
        ws._ACTIVE, ws._ACTIVE_SHARED = self._saved

    def test_two_stores_schedule(self):
        kda, moe, kda_ranks, sh_ranks, wk, wsh = self._stores()
        ws._ACTIVE, ws._ACTIVE_SHARED = kda_ranks[1], sh_ranks[1]
        ws.check_all_loaded()
        kidx = {l: k for k, l in enumerate(kda)}
        sidx = {l: k for k, l in enumerate(moe)}
        for f in range(3):
            for layer in range(93):
                hooked = ws.needs_layer_hook(layer)
                self.assertEqual(
                    hooked, layer in kidx or layer in sidx or layer > 92, layer
                )
                if hooked:
                    ws.on_layer_start(layer)
                if layer in kidx:
                    for name, *_ in SHAPES:
                        self.assertTrue(
                            torch.equal(
                                ws._ACTIVE.param_view(kidx[layer], name),
                                wk[kidx[layer]][name],
                            )
                        )
                if hooked:
                    ws.on_moe_start(layer)
                if layer in sidx:
                    for name, *_ in self.SHARED:
                        self.assertTrue(
                            torch.equal(
                                ws._ACTIVE_SHARED.param_view(sidx[layer], name),
                                wsh[sidx[layer]][name],
                            ),
                            (f, layer, name),
                        )

    def test_lean_issue_needs_cuda(self):
        ranks = make_group(2, [0, 1], 2)
        self.assertFalse(ranks[0].lean_issue)

    def test_missing_shared_piece_fails_check(self):
        kda = k3_kda_layers()
        sh = make_group(2, [1, 2], 93, shapes=self.SHARED)[0]
        k = make_group(2, kda, 93)[0]
        load_all(k, full_weights(len(kda)))
        w = full_weights(2, shapes=self.SHARED)
        for u in range(2):
            sh.load(
                u, "gate_up", w[u]["gate_up"][: self.I], row_offset=0
            )  # up half missing
            sh.load(u, "down", w[u]["down"])
        ws._ACTIVE, ws._ACTIVE_SHARED = k, sh
        with self.assertRaises(RuntimeError):
            ws.check_all_loaded()


class TestMoeWait(TestCase):
    """SGLANG_K3_KDA_WS_MOE_WAIT: which gathers the MoE stage waits for, and that skipping
    the MoE-stage wait keeps the slot invariants over the full K3 pattern."""

    I = TestSharedExpertStore.I
    SHARED = TestSharedExpertStore.SHARED

    def setUp(self):
        self._saved = (ws._ACTIVE, ws._ACTIVE_SHARED)

    def tearDown(self):
        ws._ACTIVE, ws._ACTIVE_SHARED = self._saved

    def test_parse(self):
        self.assertEqual(ws.parse_moe_wait("always"), 0)
        self.assertEqual(ws.parse_moe_wait(" Never "), None)
        self.assertEqual(ws.parse_moe_wait("2048"), 2048)
        self.assertEqual(ws.parse_moe_wait("0"), 0)
        for bad in ("", "sometimes", "-1", "2k", "1.5"):
            with self.assertRaises(ValueError, msg=bad):
                ws.parse_moe_wait(bad)
        with self.assertRaises(ValueError):
            make_group(2, [0], 1, moe_wait_min_tokens=-1)

    def test_env_default_always(self):
        try:
            from sglang.srt.environ import envs
        except ImportError:
            self.skipTest("sglang not importable (standalone module run)")
        with unittest.mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_K3_KDA_WS_MOE_WAIT", None)
            self.assertEqual(ws.parse_moe_wait(envs.SGLANG_K3_KDA_WS_MOE_WAIT.get()), 0)

    def test_threshold_needs_token_count(self):
        rank = make_group(2, [0, 1, 2], 3, moe_wait_min_tokens=2048)[0]
        self.assertTrue(rank.moe_wait_needs_tokens())
        rank.on_layer_start(0)  # issues unit 1
        with self.assertRaisesRegex(ValueError, "token threshold"):
            rank.on_moe_start(0)
        for m in (0, None):
            rank = make_group(2, [0], 1, moe_wait_min_tokens=m)[0]
            self.assertFalse(rank.moe_wait_needs_tokens())

    def _drive(self, min_tokens, num_tokens, depth=1, forwards=3):
        """Both stores through the module hooks over the 93-layer pattern. Returns, per
        store, the (layer, units issued at its start, units the MoE stage waited for)."""
        kda = k3_kda_layers()
        moe = list(range(1, 93))
        kw = dict(moe_wait_min_tokens=min_tokens)
        k_ranks = make_group(4, kda, 93, depth=depth, **kw)
        s_ranks = make_group(4, moe, 93, shapes=self.SHARED, **kw)
        wk = full_weights(len(kda), seed=6)
        wsh = full_weights(len(moe), seed=7, shapes=self.SHARED)
        for r in k_ranks:
            load_all(r, wk)
        for r in s_ranks:
            for k, w in enumerate(wsh):
                for sid in (0, 1):
                    rows = w["gate_up"][sid * self.I : (sid + 1) * self.I]
                    r.load(k, "gate_up", rows, row_offset=sid * self.I)
                r.load(k, "down", w["down"])
        k_rank, s_rank = k_ranks[2], s_ranks[2]
        ws._ACTIVE, ws._ACTIVE_SHARED = k_rank, s_rank
        ws.check_all_loaded()
        self.assertEqual(ws.moe_wait_needs_tokens(), min_tokens not in (0, None))
        stores = {"kda": (k_rank, wk), "shared": (s_rank, wsh)}
        log = {name: [] for name in stores}
        waited = {name: [] for name in stores}  # this forward's layer-start waits
        for name, (st, _) in stores.items():
            orig_issue, orig_units, orig_wait = st.issue, st.moe_wait_units, st.wait

            def issue(u, st=st, orig=orig_issue):
                if u not in st._pending:
                    # The slot's previous unit is neither pending nor running.
                    prev = [p for p in st._pending if st.slot_of(p) == st.slot_of(u)]
                    self.assertEqual(prev, [], (u, prev))
                    if st._running is not None:
                        self.assertNotEqual(st.slot_of(st._running), st.slot_of(u), u)
                orig(u)

            def units(n=None, name=name, orig=orig_units):
                out = orig(n)
                log[name][-1][2] = out
                return out

            def wait(k, orig=orig_wait, name=name):
                waited[name].append(k)
                orig(k)

            st.issue, st.moe_wait_units, st.wait = issue, units, wait
        for f in range(forwards):
            for name in stores:
                waited[name].clear()
            for layer in range(93):
                before = {name: set(st.pending) for name, (st, _) in stores.items()}
                for name in stores:
                    log[name].append([layer, None, []])
                if ws.needs_layer_hook(layer):
                    ws.on_layer_start(layer)
                for name, (st, w) in stores.items():
                    log[name][-1][1] = sorted(set(st.pending) - before[name])
                    u = st.unit_index.get(layer)
                    if u is not None:
                        self.assertNotIn(u, st.pending)
                        for pname in w[u]:
                            got = st.param_view(u, pname)
                            self.assertTrue(
                                torch.equal(got, w[u][pname]), (name, f, layer)
                            )
                    self.assertLessEqual(len(st.pending), st.prefetch_depth)
                    slots = {st.slot_of(p) for p in st.pending}
                    self.assertEqual(len(slots), len(st.pending))
                if ws.needs_layer_hook(layer):
                    ws.on_moe_start(layer, num_tokens)
                for name, (st, _) in stores.items():
                    # Whatever the MoE stage did not wait for is still pending through it.
                    issued, moe_waited = log[name][-1][1], log[name][-1][2]
                    self.assertTrue(set(issued) - set(moe_waited) <= set(st.pending))
            for name, (st, _) in stores.items():
                self.assertEqual(waited[name], list(range(st.num_units)), (name, f))
        return log

    def test_always_waits_for_what_the_layer_issued(self):
        for min_tokens, num_tokens in ((0, None), (2048, 2048), (2048, 8192)):
            log = self._drive(min_tokens, num_tokens)
            for name, rows in log.items():
                self.assertTrue(any(issued for _, issued, _ in rows), name)
                for layer, issued, moe_waited in rows:
                    self.assertEqual(moe_waited, issued, (min_tokens, name, layer))

    def test_skipped_moe_wait_keeps_slots_safe(self):
        for depth in (1, 2):
            for min_tokens, num_tokens in ((None, None), (None, 8192), (2048, 1024)):
                log = self._drive(min_tokens, num_tokens, depth=depth)
                for name, rows in log.items():
                    self.assertTrue(any(issued for _, issued, _ in rows), name)
                    moe_waits = [m for _, _, m in rows if m]
                    self.assertEqual(moe_waits, [], (name, min_tokens))

    def test_issue_at_core_never(self):
        kda = k3_kda_layers()
        ranks = make_group(4, kda, 93, issue_at="core", moe_wait_min_tokens=None)
        w = full_weights(len(kda), seed=8)
        for r in ranks:
            load_all(r, w)
        rank = ranks[1]
        for f in range(2):
            for layer in range(93):
                rank.on_layer_start(layer)
                u = rank.unit_index.get(layer)
                if u is not None:
                    for name, *_ in SHAPES:
                        got = rank.param_view(u, name)
                        self.assertTrue(torch.equal(got, w[u][name]), (f, layer))
                    rank.on_core_start(layer)
                self.assertEqual(rank.moe_wait_units(), [])
                self.assertLessEqual(len(rank.pending), 1)


@skipUnless(torch.cuda.is_available(), "CUDA")
class TestCudaStreams(TestCase):
    def test_gemm_chain_bitwise(self):
        torch.manual_seed(0)
        dev = "cuda"
        kda = k3_kda_layers(16)
        n = 4
        ranks = make_group(n, kda, 16, device=dev, dtype=torch.bfloat16)
        w = full_weights(len(kda), seed=3, device=dev, dtype=torch.bfloat16)
        for rank in ranks:
            load_all(rank, w)
            rank.check_loaded()
        rank = ranks[1]
        kidx = {l: k for k, l in enumerate(kda)}
        x0 = torch.randn(4096, H, device=dev, dtype=torch.bfloat16)
        big = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)

        def chain(get_w):
            x = x0
            for layer in range(16):
                if layer in kidx:
                    k = kidx[layer]
                    for _ in range(
                        4
                    ):  # keep the compute stream busy past the next issue
                        big @ big
                    y = x @ get_w(k, "qkvg").t()  # [T, 4P]
                    x = (y[:, :P] @ get_w(k, "o_proj").t()).tanh()  # [T, H]
            return x

        ref = chain(lambda k, name: w[k][name])
        for _ in range(2):

            def get_w(k, name):
                return rank.param_view(k, name)

            x = x0
            for layer in range(16):
                rank.on_layer_start(layer)
                if layer in kidx:
                    k = kidx[layer]
                    for _ in range(4):
                        big @ big
                    y = x @ get_w(k, "qkvg").t()
                    x = (y[:, :P] @ get_w(k, "o_proj").t()).tanh()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(x, ref))

    def test_moe_wait_modes_bitwise(self):
        # A MoE-stage stand-in (big GEMMs on the compute stream) between attention and the
        # next layer: with the MoE-stage wait off or thresholded out, the gathers run through
        # it and are waited on at their own layer's start; outputs stay bitwise equal to the
        # replicated chain, and the stats show where the wait happened.
        torch.manual_seed(0)
        dev = "cuda"
        kda = k3_kda_layers(16)
        w = full_weights(len(kda), seed=9, device=dev, dtype=torch.bfloat16)
        kidx = {l: k for k, l in enumerate(kda)}
        x0 = torch.randn(4096, H, device=dev, dtype=torch.bfloat16)
        big = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)

        def run(get_w, layer_hook, moe_hook):
            x = x0
            for layer in range(16):
                layer_hook(layer)
                if layer in kidx:
                    k = kidx[layer]
                    y = x @ get_w(k, "qkvg").t()
                    x = (y[:, :P] @ get_w(k, "o_proj").t()).tanh()
                moe_hook(layer)
                for _ in range(2):
                    big @ big
            torch.cuda.synchronize()
            return x

        ref = run(lambda k, name: w[k][name], lambda layer: None, lambda layer: None)
        modes = (
            (0, None, True),
            (None, None, False),
            (2048, 1024, False),
            (2048, 4096, True),
        )
        for min_tokens, num_tokens, moe_waits in modes:
            kw = dict(device=dev, dtype=torch.bfloat16, moe_wait_min_tokens=min_tokens)
            ranks = make_group(4, kda, 16, **kw)
            for rank in ranks:
                load_all(rank, w)
                rank.check_loaded()
            rank = ranks[3]
            rank.stats = True

            def moe_hook(layer, rank=rank, num_tokens=num_tokens):
                rank.on_moe_start(layer, num_tokens)

            for _ in range(2):
                out = run(rank.param_view, rank.on_layer_start, moe_hook)
                self.assertTrue(torch.equal(out, ref), (min_tokens, num_tokens))
            rank._fold_stalls()
            mode = (min_tokens, num_tokens)
            self.assertEqual(rank.moe_wait_count > 0, moe_waits, mode)
            self.assertGreater(rank.layer_wait_count, 0)

    def test_lean_issue_bitwise_and_reuses_events(self):
        # Mover "ce" on CUDA issues from per-unit copy views and
        # reused events by default; both issue paths gather the same bytes, and the lean path
        # records each unit's gather on the same Event object every forward.
        dev = "cuda"
        kda = k3_kda_layers(16)
        w = full_weights(len(kda), seed=5, device=dev, dtype=torch.bfloat16)
        kidx = {l: k for k, l in enumerate(kda)}
        x0 = torch.randn(1024, H, device=dev, dtype=torch.bfloat16)
        outs = {}
        for lean in (True, False):
            ranks = make_group(4, kda, 16, device=dev, dtype=torch.bfloat16)
            for rank in ranks:
                load_all(rank, w)
                rank.check_loaded()
            rank = ranks[2]
            self.assertTrue(rank.lean_issue)
            rank.lean_issue = lean
            seen = []
            for _ in range(2):
                x = x0
                events = {}
                for layer in range(16):
                    rank.on_layer_start(layer)
                    events.update({u: id(ev) for u, ev in rank._pending.items()})
                    if layer in kidx:
                        k = kidx[layer]
                        x = (x @ rank.param_view(k, "qkvg").t())[
                            :, :P
                        ] @ rank.param_view(k, "o_proj").t()
                        x = x.tanh()
                seen.append(events)
                torch.cuda.synchronize()
            outs[lean] = x
            if lean:
                common = set(seen[0]) & set(seen[1])
                self.assertTrue(common)
                self.assertTrue(all(seen[0][u] == seen[1][u] for u in common))
        self.assertTrue(torch.equal(outs[True], outs[False]))


@skipUnless(torch.cuda.is_available(), "CUDA")
class TestCudaHeadAligned(TestCase):
    """Head-aligned gathers on one GPU: K3-sized units from cuMem-allocated shards (as
    _vmm_alloc_and_peers makes them) are bitwise, the copies run on the copy engines only
    (no kernel in the profile; the o_proj column slices are cuMemcpy2DAsync), and a K3
    rank holds 69 shards + 2 slots (7.60 + 1.76 GB)."""

    dt = torch.bfloat16

    def test_k3_shapes_bitwise_from_cumem_shards(self):
        kda, n = [0, 1, 2], 8
        ranks = make_group(
            n,
            kda,
            3,
            device="cuda",
            dtype=self.dt,
            shapes=K3_SHAPES,
            alloc=vmm_alloc("cuda", self.dt),
        )
        w = full_weights(
            len(kda), seed=11, device="cuda", dtype=self.dt, shapes=K3_SHAPES
        )
        for rk in ranks:
            load_all(rk, w)
            rk.check_loaded()
        for rk in (ranks[0], ranks[5]):
            for layer in range(3):
                rk.on_layer_start(layer)
                torch.cuda.synchronize()
                for name, *_ in K3_SHAPES:
                    self.assertTrue(
                        torch.equal(rk.param_view(layer, name), w[layer][name])
                    )

    def test_gather_launches_no_kernels(self):
        kda, n = [0, 1], 4
        ranks = make_group(n, kda, 2, device="cuda", dtype=self.dt)
        w = full_weights(len(kda), seed=2, device="cuda", dtype=self.dt)
        for rk in ranks:
            load_all(rk, w)
        rk = ranks[1]
        torch.cuda.synchronize()
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            rk.on_layer_start(0)
            rk.on_layer_start(1)
            torch.cuda.synchronize()
        kinds = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
        self.assertTrue(kinds)
        self.assertTrue(all("emcpy" in k or "Memcpy" in k for k in kinds), kinds)
        for layer in range(2):
            for name, *_ in SHAPES:
                self.assertTrue(torch.equal(rk.param_view(layer, name), w[layer][name]))

    def test_memory_per_rank(self):
        specs, size = ws.layer_layout(K3_SHAPES)
        _, shard = ws.unit_pieces(specs, 8)
        self.assertAlmostEqual(69 * shard * 2 / 1e9, 7.60, places=2)
        self.assertAlmostEqual(2 * size * 2 / 1e9, 1.76, places=2)


if __name__ == "__main__":
    unittest.main()
