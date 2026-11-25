"""Order of the shared experts' all-reduce around a MegaMoE call under SGLANG_K3_AR_FUSION
(no GPU: KimiK3MoE._forward_fused runs on a stand-in module with recording streams)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import os
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.communication import k3_ar_fusion
from sglang.srt.layers.moe import k3_deepgemm_routed as routed
from sglang.srt.layers.moe import k3_die_local_moe as policy
from sglang.srt.models import kimi_k3

ENV = "SGLANG_K3_ROUTED_MOE_DEEPGEMM_SHARED_AR_ORDER"
PULL = "SGLANG_K3_ROUTED_MOE_DEEPGEMM_FUSED_PULL_"
H, LATENT = (
    64,
    32,
)  # hidden = 2 x latent, as in K3 (7168 / 3584): the buffer is 3 norm-width rows per token


class _Stream:
    def __init__(self, trace, name):
        self.trace, self.name, self.deps = trace, name, set()

    def wait_stream(self, other):
        # everything issued on `other` so far precedes whatever this stream issues next
        self.deps |= set(self.trace.issued[other.name])
        self.trace.waits.append((self.name, other.name))


class _Trace:
    """Kernels as (name -> stream, ancestors): an op follows the earlier ops of its stream and the ops its stream waited for."""

    def __init__(self):
        self.issued = {"main": [], "alt": []}
        self.before, self.stream_of, self.order, self.waits = {}, {}, [], []
        self.main, self.alt = _Stream(self, "main"), _Stream(self, "alt")
        self.current = self.main

    def op(self, name):
        s = self.current
        direct = set(s.deps) | set(self.issued[s.name])
        self.before[name] = (
            set(direct).union(*(self.before[d] for d in direct)) if direct else set()
        )
        self.stream_of[name] = s.name
        self.issued[s.name].append(name)
        self.order.append(name)

    def ordered(self, a, b):
        return a in self.before[b] or b in self.before[a]

    @contextlib.contextmanager
    def stream(self, s):
        prev, self.current = self.current, s
        try:
            yield
        finally:
            self.current = prev


def _run(
    order,
    *,
    mega,
    sharded,
    tail=True,
    rows=80,
    fuse_norm=True,
    defer=True,
    alt=True,
    got=None,
    env_extra=None,
):
    """One _forward_fused call. Returns the trace and the b_stream the gemm_ag tail got (None if the tail did not run).
    `got` (a dict) also receives what the collectives and the tail were called with."""
    tr = _Trace()
    got = {} if got is None else got
    buffers = []

    def symm_buffer(name, n, width, dtype):
        buffers.append(torch.zeros(n, width, dtype=dtype))
        return buffers[-1]

    def reduce_norm(x, w, eps, num_tokens, **geometry):
        # the latent alone ([rows, LATENT]) or the whole [latent | shared] pair viewed at the norm width
        whole = x.numel() == buffers[-1].numel()
        got.setdefault("norm_calls", []).append(
            dict(
                shape=tuple(x.shape),
                num_tokens=num_tokens,
                geometry=geometry,
                whole=whole,
                at_buffer_start=x.data_ptr() == buffers[-1].data_ptr(),
            )
        )
        tr.op("fused_ar" if whole else "latent_ar")

    def reduce_plain(x):
        whole = x.numel() == buffers[-1].numel()
        got.setdefault("plain_calls", []).append(
            dict(shape=tuple(x.shape), whole=whole)
        )
        tr.op("fused_ar" if whole else "latent_ar")

    layer = (
        types.SimpleNamespace(mega_l1_weights=(1, 1))
        if mega
        else types.SimpleNamespace()
    )

    def fake_tail(latent, weight, b, c, *, b_stream=None):
        got["b_stream"] = b_stream
        got["tail_b_is_shared"] = b.data_ptr() == buffers[
            -1
        ].data_ptr() + rows * LATENT * b.element_size() and tuple(b.shape) == (rows, H)
        tr.op("slice_gemm")
        if b_stream is not None:
            tr.current.wait_stream(b_stream)
        tr.op("gather_add3")
        return torch.zeros(rows, H)

    fusion = types.SimpleNamespace(
        MOE_LATENT_SHARED="moe_latent_shared",
        NORM_DIM=LATENT,
        enabled=lambda: True,
        symm_buffer=symm_buffer,
        finalize_push_fits=lambda n: n <= 73,
        all_reduce_low_sm=lambda x, **kw: tr.op("shared_ar"),
        all_reduce_norm=reduce_norm,
        all_reduce=reduce_plain,
        finalize_all_reduce_push_norm=lambda *a: tr.op("latent_ar"),
        gemm_ag_up_fits=lambda n: tail,
        gemm_ag_up_proj=fake_tail,
    )

    class UpProj:
        weight = torch.zeros(H, LATENT)

        def __call__(self, latent):
            tr.op("up_proj")
            return torch.zeros(rows, H), None

    me = types.SimpleNamespace(
        moe_hidden_size=LATENT,
        alt_stream=tr.alt if alt else None,
        experts=layer,
        _defer_moe_finalize=defer,
        fuse_ar_norm=fuse_norm,
        _gemm_ag_up_eligible=fuse_norm,
        routed_expert_up_proj=UpProj(),
        _sharded_front_covers=lambda n: sharded,
        _sharded_front_gemm=lambda hs: (tr.op("front_gemm"), ("gate_up", "block"))[1],
        _fused_front=lambda hs: (
            tr.op("front_gemm"),
            ("gate_up", "logits", "routed_in"),
        )[1],
        _sharded_front_gather=lambda block, hs: (
            tr.op("front_gather"),
            ("logits", "routed_in"),
        )[1],
        _forward_shared=lambda gate_up, out: tr.op("shared_gemm"),
        _forward_routed=lambda hs, logits, x, latent: tr.op("routed"),
        _forward_routed_deferred=lambda hs, logits, x: (
            tr.op("routed"),
            types.SimpleNamespace(
                gemm2_out=None, expanded_idx_to_permuted_idx=None, expert_weights=None
            ),
        )[1],
        _get_fused_norm_params=lambda: (torch.ones(LATENT), 1e-6),
        _latent_norm=lambda latent: (tr.op("latent_norm"), latent)[1],
    )
    env = {k: v for k, v in os.environ.items() if k != ENV and not k.startswith(PULL)}
    if order is not None:
        env[ENV] = order
    env.update(env_extra or {})
    routed.configured_shared_ar_order.cache_clear()
    routed.fused_pull_tuning.cache_clear()
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.object(kimi_k3, "k3_ar_fusion", fusion),
        mock.patch.object(policy, "takes", lambda rows: True),
        mock.patch.object(
            kimi_k3,
            "_add3",
            lambda out, b, c, prefetch_bc=False: (tr.op("add3"), out)[1],
        ),
        mock.patch.object(torch.cuda, "current_stream", lambda: tr.current),
        mock.patch.object(torch.cuda, "stream", tr.stream),
    ):
        try:
            kimi_k3.KimiK3MoE._forward_fused(me, torch.zeros(rows, H), prefix_sum=None)
        finally:
            routed.configured_shared_ar_order.cache_clear()
            routed.fused_pull_tuning.cache_clear()
    return tr, got.get("b_stream", "no tail")


class TestSharedArOrderPolicy(unittest.TestCase):
    def setUp(self):
        routed.configured_shared_ar_order.cache_clear()
        self.addCleanup(routed.configured_shared_ar_order.cache_clear)

    def test_default_and_values(self):
        mega = types.SimpleNamespace(mega_l1_weights=(1, 1))
        stock = types.SimpleNamespace()
        for value, want in (
            (None, "fused"),
            ("after", "after"),
            ("before", "before"),
            ("overlap", "overlap"),
            ("fused", "fused"),
        ):
            env = {k: v for k, v in os.environ.items() if k != ENV}
            if value is not None:
                env[ENV] = value
            routed.configured_shared_ar_order.cache_clear()
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch.object(policy, "takes", lambda rows: True),
            ):
                self.assertEqual(routed.shared_ar_order(mega, 80), want)
                # a call trtllm-gen takes keeps the stock order whatever the switch says
                self.assertEqual(routed.shared_ar_order(stock, 80), "overlap")

    def test_invalid_value(self):
        with (
            mock.patch.dict(os.environ, {ENV: "sometimes"}),
            mock.patch.object(policy, "takes", lambda rows: True),
        ):
            with self.assertRaises(ValueError):
                routed.shared_ar_order(
                    types.SimpleNamespace(mega_l1_weights=(1, 1)), 80
                )

    def test_prepared_layer_follows_the_width_table(self):
        prepared = types.SimpleNamespace(mega_l1_weights=(1, 1))
        with mock.patch.object(policy, "takes", lambda rows: rows >= 80):
            self.assertEqual(routed.shared_ar_order(prepared, 160), "fused")
            self.assertEqual(routed.shared_ar_order(prepared, 40), "overlap")


class TestForwardFusedOrder(unittest.TestCase):
    def test_stock_call_is_unchanged(self):
        # trtllm-gen call, every switch value: the stock sequence, shared AR queued before and unordered with the routed call
        for order in (None, "after", "before", "overlap", "fused"):
            for rows in (40, 80):
                tr, b_stream = _run(order, mega=False, sharded=True, rows=rows)
                self.assertEqual(
                    tr.order,
                    [
                        "front_gemm",
                        "shared_gemm",
                        "front_gather",
                        "shared_ar",
                        "routed",
                        "latent_ar",
                        "slice_gemm",
                        "gather_add3",
                    ],
                )
                self.assertEqual(
                    tr.waits, [("alt", "main"), ("alt", "main"), ("main", "alt")]
                )
                self.assertFalse(tr.ordered("shared_ar", "routed"))
                self.assertIn("shared_ar", tr.before["latent_ar"])
                self.assertIsNone(b_stream)

    def test_overlap_reproduces_the_stock_order_for_a_mega_call(self):
        tr, b_stream = _run("overlap", mega=True, sharded=True)
        ref, _ = _run("after", mega=False, sharded=True)
        self.assertEqual(
            (tr.order, tr.waits, tr.stream_of), (ref.order, ref.waits, ref.stream_of)
        )
        self.assertFalse(
            tr.ordered("shared_ar", "routed")
        )  # the measured regression: the two race for the SMs

    def test_after(self):
        tr, b_stream = _run("after", mega=True, sharded=True)
        self.assertEqual(tr.stream_of["shared_gemm"], "alt")
        self.assertEqual(tr.stream_of["shared_ar"], "alt")
        self.assertEqual(tr.stream_of["routed"], "main")
        # the kernel starts behind the shared GEMM and the front gather, and nothing else is in flight with it
        self.assertLessEqual({"shared_gemm", "front_gather"}, tr.before["routed"])
        for other in tr.order:
            if other != "routed":
                self.assertTrue(tr.ordered("routed", other), other)
        # shared AR: behind the latent AR (same pull semaphores), next to the tail's local GEMM, joined by the gather
        self.assertIn("latent_ar", tr.before["shared_ar"])
        self.assertFalse(tr.ordered("shared_ar", "slice_gemm"))
        self.assertIn("shared_ar", tr.before["gather_add3"])
        self.assertIn("latent_ar", tr.before["slice_gemm"])
        self.assertIs(b_stream, tr.alt)
        self.assertEqual(tr.order.count("shared_ar"), 1)

    def test_after_without_the_gemm_ag_tail_joins_before_the_add(self):
        tr, b_stream = _run("after", mega=True, sharded=True, tail=False)
        self.assertEqual(b_stream, "no tail")
        self.assertIn("latent_ar", tr.before["shared_ar"])
        self.assertIn("shared_ar", tr.before["up_proj"])
        self.assertIn("shared_ar", tr.before["add3"])

    def test_before(self):
        tr, b_stream = _run("before", mega=True, sharded=True)
        self.assertLessEqual(
            {"shared_gemm", "front_gather", "shared_ar"}, tr.before["routed"]
        )
        self.assertIn(
            "front_gather", tr.before["shared_ar"]
        )  # the AR still waits for the gather
        self.assertIn("shared_ar", tr.before["latent_ar"])
        self.assertIsNone(b_stream)
        for other in tr.order:
            if other != "routed":
                self.assertTrue(tr.ordered("routed", other), other)

    def test_unsharded_front(self):
        for order in ("after", "before"):
            tr, b_stream = _run(order, mega=True, sharded=False)
            self.assertLessEqual({"routed"}, tr.before["shared_gemm"])
            self.assertLessEqual({"routed", "shared_gemm"}, tr.before["shared_ar"])
            self.assertIn("shared_ar", tr.before["latent_ar"])
            self.assertIsNone(b_stream)
        tr, _ = _run("overlap", mega=True, sharded=False)
        self.assertFalse(tr.ordered("routed", "shared_ar"))

    def test_the_two_pull_all_reduces_never_overlap(self):
        for order in ("after", "before", "overlap", "fused"):
            for mega in (True, False):
                for sharded in (True, False):
                    for tail in (True, False):
                        tr, _ = _run(order, mega=mega, sharded=sharded, tail=tail)
                        reader = "gather_add3" if tail else "add3"
                        if order == "fused" and mega:
                            # one collective: there is no second pull to meet
                            self.assertEqual(
                                [op for op in tr.order if op.endswith("_ar")],
                                ["fused_ar"],
                                (mega, sharded, tail),
                            )
                            self.assertIn(
                                "fused_ar", tr.before[reader], (mega, sharded, tail)
                            )
                            continue
                        self.assertNotIn("fused_ar", tr.order)
                        self.assertTrue(
                            tr.ordered("shared_ar", "latent_ar"),
                            (order, mega, sharded, tail),
                        )
                        self.assertIn(
                            "shared_ar", tr.before[reader], (order, mega, sharded, tail)
                        )
                        self.assertIn(
                            "latent_ar", tr.before[reader], (order, mega, sharded, tail)
                        )


class TestFusedPull(unittest.TestCase):
    """SGLANG_K3_ROUTED_MOE_DEEPGEMM_SHARED_AR_ORDER=fused: one pull over the [latent | shared] pair behind the MegaMoE kernel."""

    def test_sequence_and_streams(self):
        got = {}
        tr, b_stream = _run("fused", mega=True, sharded=True, got=got)
        self.assertEqual(
            tr.order,
            [
                "front_gemm",
                "shared_gemm",
                "front_gather",
                "routed",
                "fused_ar",
                "slice_gemm",
                "gather_add3",
            ],
        )
        self.assertEqual(tr.stream_of["shared_gemm"], "alt")
        for op in ("front_gather", "routed", "fused_ar", "slice_gemm", "gather_add3"):
            self.assertEqual(tr.stream_of[op], "main", op)
        # the kernel starts behind the shared GEMM and the gather; nothing is in flight with it
        self.assertLessEqual({"shared_gemm", "front_gather"}, tr.before["routed"])
        for other in tr.order:
            if other != "routed":
                self.assertTrue(tr.ordered("routed", other), other)
        # the pull follows both producers; the whole tail follows the pull on its own stream: no second stream to join
        self.assertLessEqual({"shared_gemm", "routed"}, tr.before["fused_ar"])
        self.assertIn("fused_ar", tr.before["slice_gemm"])
        self.assertIn("fused_ar", tr.before["gather_add3"])
        self.assertIsNone(b_stream)
        self.assertTrue(got["tail_b_is_shared"])
        self.assertNotIn(
            ("alt", "main"), tr.waits[1:]
        )  # the side stream is not used again after the shared GEMM
        self.assertEqual(tr.issued["alt"], ["shared_gemm"])

    def test_the_collective_is_the_kernel_contract(self):
        # all_reduce_pull_norm's contract (ar_fusion.cuh): rows of the latent width, N latent rows then the 2N half-rows of
        # the shared output, num_norm_rows = N. That is the flat buffer from its first element.
        for rows in (80, 160, 235):
            got = {}
            _run("fused", mega=True, sharded=True, rows=rows, got=got)
            (call,) = got["norm_calls"]
            self.assertEqual(call["shape"], (3 * rows, LATENT))
            self.assertEqual(call["num_tokens"], rows)
            self.assertTrue(call["whole"] and call["at_buffer_start"])
            self.assertEqual(call["geometry"], {})  # default: the kernel's own table
            self.assertNotIn("plain_calls", got)

    def test_geometry_override(self):
        got = {}
        _run(
            "fused",
            mega=True,
            sharded=True,
            got=got,
            env_extra={PULL + "BLOCKS": "16", PULL + "UNROLL": "4"},
        )
        self.assertEqual(
            got["norm_calls"][0]["geometry"], {"num_blocks": 16, "unroll": 4}
        )
        got = {}
        _run(
            "fused", mega=True, sharded=True, got=got, env_extra={PULL + "BLOCKS": "32"}
        )
        self.assertEqual(got["norm_calls"][0]["geometry"], {"num_blocks": 32})
        for bad in (
            {PULL + "UNROLL": "3"},
            {PULL + "BLOCKS": "0"},
            {PULL + "BLOCKS": "x"},
        ):
            with self.assertRaises(ValueError):
                _run("fused", mega=True, sharded=True, env_extra=bad)
        # the override belongs to the fused pull only: the other orders' latent pull keeps the table
        for order in ("after", "before", "overlap"):
            got = {}
            _run(
                order,
                mega=True,
                sharded=True,
                got=got,
                env_extra={PULL + "BLOCKS": "16"},
            )
            self.assertEqual([c["geometry"] for c in got["norm_calls"]], [{}])
            self.assertFalse(got["norm_calls"][0]["whole"])

    def test_bad_geometry_fails_with_the_order(self):
        # configured_shared_ar_order validates the pull geometry too, so a bad value stops the first
        # kernel-taken call instead of a later one
        with mock.patch.dict(os.environ, {ENV: "fused", PULL + "UNROLL": "3"}):
            routed.configured_shared_ar_order.cache_clear()
            routed.fused_pull_tuning.cache_clear()
            with self.assertRaises(ValueError):
                routed.configured_shared_ar_order()
        routed.configured_shared_ar_order.cache_clear()
        routed.fused_pull_tuning.cache_clear()

    def test_trtllm_call_is_the_stock_sequence(self):
        ref, _ = _run("overlap", mega=False, sharded=True, rows=40)
        for rows in (40, 80):
            got = {}
            tr, b_stream = _run("fused", mega=False, sharded=True, rows=rows, got=got)
            self.assertEqual(
                (tr.order, tr.waits, tr.stream_of),
                (ref.order, ref.waits, ref.stream_of),
            )
            self.assertNotIn("fused_ar", tr.order)
            self.assertNotIn("plain_calls", got)
            self.assertFalse(
                any(call["whole"] for call in got.get("norm_calls", []))
            )  # 40 rows: the finalize-fused push
        # a prepared layer: the width table decides per call
        prepared = types.SimpleNamespace(mega_l1_weights=(1, 1))
        with (
            mock.patch.dict(os.environ, {ENV: "fused"}),
            mock.patch.object(policy, "takes", lambda rows: rows >= 80),
        ):
            routed.configured_shared_ar_order.cache_clear()
            self.assertEqual(routed.shared_ar_order(prepared, 160), "fused")
            self.assertEqual(routed.shared_ar_order(prepared, 40), "overlap")

    def test_unsharded_front(self):
        got = {}
        tr, b_stream = _run("fused", mega=True, sharded=False, got=got)
        self.assertEqual(
            tr.order,
            [
                "front_gemm",
                "shared_gemm",
                "routed",
                "fused_ar",
                "slice_gemm",
                "gather_add3",
            ],
        )
        self.assertLessEqual({"shared_gemm"}, tr.before["routed"])
        self.assertLessEqual({"shared_gemm", "routed"}, tr.before["fused_ar"])
        self.assertEqual(tr.order.count("shared_gemm"), 1)
        self.assertIsNone(b_stream)

    def test_without_the_gemm_ag_tail(self):
        tr, b_stream = _run("fused", mega=True, sharded=True, tail=False)
        self.assertEqual(b_stream, "no tail")
        self.assertEqual(
            tr.order[-3:], ["fused_ar", "up_proj", "add3"]
        )  # the norm ran in the pull
        self.assertIn("fused_ar", tr.before["up_proj"])
        self.assertIn("fused_ar", tr.before["add3"])

    def test_without_the_fused_norm(self):
        # no norm epilogue available: a plain collective over the pair, then the unfused tail, as in the single-stream branch
        got = {}
        tr, b_stream = _run("fused", mega=True, sharded=True, fuse_norm=False, got=got)
        self.assertEqual(b_stream, "no tail")
        self.assertEqual(
            tr.order,
            [
                "front_gemm",
                "shared_gemm",
                "front_gather",
                "routed",
                "fused_ar",
                "latent_norm",
                "up_proj",
                "add3",
            ],
        )
        self.assertEqual(
            got["plain_calls"], [dict(shape=(80 * (LATENT + H),), whole=True)]
        )
        self.assertNotIn("norm_calls", got)

    def test_matches_the_single_stream_branch(self):
        # the bundle's own branch without a side stream issues the same collective on the same view
        got = {}
        tr, _ = _run("fused", mega=True, sharded=True, got=got)
        ref_got = {}
        ref, _ = _run(
            "overlap", mega=True, sharded=False, alt=False, got=ref_got
        )  # stock code path, untouched by the patches
        self.assertEqual(got["norm_calls"], ref_got["norm_calls"])
        self.assertEqual([op for op in ref.order if op.endswith("_ar")], ["fused_ar"])
        self.assertEqual(ref.waits, [])

    def test_a_deferred_finalize_call_falls_back_to_before(self):
        # not reachable today (a call the kernel takes is never deferred); pins the fallback should that change
        src = __import__("inspect").getsource(kimi_k3.KimiK3MoE._forward_fused)
        guard = src.index('if shared_ar == "fused" and defer_finalize:')
        self.assertLess(guard, src.index('fused_pull = shared_ar == "fused"'))
        self.assertIn(
            'shared_ar = "before"',
            src[guard : src.index('fused_pull = shared_ar == "fused"')],
        )


class TestGemmAgTailWaitsForB(unittest.TestCase):
    def _call(self, rows, b_stream):
        from sglang.kernels.ops.communication import gemm_ag as mod

        tr = _Trace()
        b_stream = tr.alt if b_stream else None
        with tr.stream(tr.alt):
            tr.op("shared_ar")
        state = types.SimpleNamespace(world_size=8, rank=1)
        with (
            mock.patch.object(k3_ar_fusion, "_get_state", lambda: state),
            mock.patch.object(
                k3_ar_fusion, "slice_gemm", lambda x, w, *a: (tr.op("slice_gemm"), x)[1]
            ),
            mock.patch.object(mod, "gather_add3", lambda *a: tr.op("gather_add3")),
            mock.patch.object(mod, "gemm_ag_up_proj", lambda *a: tr.op("gemv_tail")),
            mock.patch.object(torch.cuda, "current_stream", lambda: tr.current),
        ):
            x, w, b = (
                torch.zeros(rows, LATENT),
                torch.zeros(64, LATENT),
                torch.zeros(rows, 64),
            )
            if b_stream is None:
                k3_ar_fusion.gemm_ag_up_proj(x, w, b, None)
            else:
                k3_ar_fusion.gemm_ag_up_proj(x, w, b, None, b_stream=b_stream)
        return tr, mod.GEMV_MAX_TOKENS

    def test_gather_waits_and_the_local_gemm_does_not(self):
        tr, _ = self._call(80, True)
        self.assertEqual(tr.order, ["shared_ar", "slice_gemm", "gather_add3"])
        self.assertNotIn("shared_ar", tr.before["slice_gemm"])
        self.assertIn("shared_ar", tr.before["gather_add3"])

    def test_fused_gemv_tail_waits(self):
        tr, gemv_max = self._call(4, True)
        self.assertLessEqual(4, gemv_max)
        self.assertEqual(tr.order, ["shared_ar", "gemv_tail"])
        self.assertIn("shared_ar", tr.before["gemv_tail"])

    def test_default_adds_no_wait(self):
        for rows in (4, 80):
            tr, _ = self._call(rows, False)
            self.assertEqual(tr.waits, [])


class TestSequenceParallelPass(unittest.TestCase):
    """_forward_fused(scatters=True) is the sequence-parallel prefill pass. The MegaMoE kernel never takes
    prefill, and there is no flat [latent | shared] buffer to pull: the pass keeps the stock order whatever the switch says."""

    def _scatter(self, order, sharded):
        tr = _Trace()
        sp = types.SimpleNamespace(
            SP_RS_OVERLAP_BLOCKS=4,
            sp_partial_rows=lambda name, like, width: torch.zeros(like.shape[0], width),
            sp_reduce_scatter=lambda x, blocks=None: (
                tr.op("shared_rs" if x.shape[1] == H else "latent_rs"),
                x,
            )[1],
        )
        boom = lambda *a, **k: self.fail(
            "an all-reduce of the [latent | shared] buffer ran on a sequence-parallel pass"
        )
        fusion = types.SimpleNamespace(
            MOE_LATENT_SHARED="moe_latent_shared",
            NORM_DIM=LATENT,
            enabled=lambda: True,
            symm_buffer=boom,
            finalize_push_fits=lambda n: True,
            all_reduce_low_sm=boom,
            all_reduce_norm=boom,
            all_reduce=boom,
            finalize_all_reduce_push_norm=boom,
            gemm_ag_up_fits=lambda n: True,
            gemm_ag_up_proj=boom,
        )

        class UpProj:
            weight = torch.zeros(H, LATENT)

            def __call__(self, latent):
                tr.op("up_proj")
                return torch.zeros(latent.shape[0], H), None

        me = types.SimpleNamespace(
            moe_hidden_size=LATENT,
            alt_stream=tr.alt,
            experts=types.SimpleNamespace(mega_l1_weights=(1, 1)),
            _defer_moe_finalize=True,
            fuse_ar_norm=True,
            _gemm_ag_up_eligible=True,
            routed_expert_up_proj=UpProj(),
            _sharded_front_covers=lambda n: sharded,
            _sharded_front_gemm=lambda hs: (tr.op("front_gemm"), ("gate_up", "block"))[
                1
            ],
            _fused_front=lambda hs: (
                tr.op("front_gemm"),
                ("gate_up", "logits", "routed_in"),
            )[1],
            _sharded_front_gather=lambda block, hs: (
                tr.op("front_gather"),
                ("logits", "routed_in"),
            )[1],
            _forward_shared=lambda gate_up, out: tr.op("shared_gemm"),
            _forward_routed=lambda hs, logits, x, latent: tr.op("routed"),
            _forward_routed_deferred=lambda *a: self.fail(
                "deferred finalize on a sequence-parallel pass"
            ),
            _get_fused_norm_params=lambda: (torch.ones(LATENT), 1e-6),
            _latent_norm=lambda latent: (tr.op("latent_norm"), latent)[1],
        )
        env = {k: v for k, v in os.environ.items() if k != ENV}
        if order is not None:
            env[ENV] = order
        routed.configured_shared_ar_order.cache_clear()
        self.addCleanup(routed.configured_shared_ar_order.cache_clear)
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(kimi_k3, "k3_ar_fusion", fusion),
            mock.patch.object(kimi_k3, "k3_sp_collective", sp),
            mock.patch.object(policy, "takes", lambda rows: True),
            mock.patch.object(
                kimi_k3,
                "_add3",
                lambda out, b, c, prefetch_bc=False: (tr.op("add3"), out)[1],
            ),
            mock.patch.object(torch.cuda, "current_stream", lambda: tr.current),
            mock.patch.object(torch.cuda, "stream", tr.stream),
        ):
            kimi_k3.KimiK3MoE._forward_fused(
                me, torch.zeros(80, H), prefix_sum=None, scatters=True
            )
        return tr

    def test_every_switch_value_gives_the_same_sequence(self):
        for sharded in (True, False):
            ref = self._scatter(None, sharded)
            self.assertIn("shared_rs", ref.order)
            self.assertIn("latent_rs", ref.order)
            self.assertEqual(ref.stream_of["shared_rs"], "alt")
            self.assertFalse(
                ref.ordered("shared_rs", "routed")
            )  # stock: the side-stream scatter overlaps the routed call
            self.assertIn("shared_rs", ref.before["latent_rs"])
            for order in ("after", "before", "overlap", "fused"):
                tr = self._scatter(order, sharded)
                self.assertEqual(
                    (tr.order, tr.waits, tr.stream_of),
                    (ref.order, ref.waits, ref.stream_of),
                )


if __name__ == "__main__":
    unittest.main()
