"""Order of the shared experts' all-reduce around a MegaMoE call under AR fusion (no
GPU: KimiK3MoE._forward_fused runs on a stand-in module with recording streams)."""

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
from sglang.test.test_utils import CustomTestCase

ENV = "SGLANG_K3_ROUTED_MOE_DEEPGEMM_SHARED_AR_ORDER"
PULL = "SGLANG_K3_ROUTED_MOE_DEEPGEMM_FUSED_PULL_"
BLOCKS, UNROLL = PULL + "BLOCKS", PULL + "UNROLL"
# hidden = 2 x latent, as in K3 (7168 / 3584): the buffer is 3 norm-width rows per token
H, LATENT = 64, 32
ORDERS = ("after", "before", "overlap", "fused")
MEGA = types.SimpleNamespace(mega_l1_weights=(1, 1))  # a layer `finish` prepared
STOCK = types.SimpleNamespace()  # a layer only trtllm-gen runs
FRONT = ["front_gemm", "shared_gemm", "front_gather"]  # sharded front
TAIL = ["slice_gemm", "gather_add3"]  # the gemm_ag tail
STOCK_ORDER = FRONT + ["shared_ar", "routed", "latent_ar"] + TAIL
# what a deferred-finalize runner call hands the push all-reduce
DEFERRED = types.SimpleNamespace(
    gemm2_out=None, expanded_idx_to_permuted_idx=None, expert_weights=None
)


class _Stream:
    def __init__(self, trace, name):
        self.trace, self.name, self.deps = trace, name, set()

    def wait_stream(self, other):
        # everything issued on `other` so far precedes whatever this stream issues next
        self.deps |= set(self.trace.issued[other.name])
        self.trace.waits.append((self.name, other.name))


class _Trace:
    """An op follows the earlier ops of its stream and the ops its stream waited for."""

    def __init__(self):
        self.issued = {"main": [], "alt": []}
        self.before, self.stream_of, self.order, self.waits = {}, {}, [], []
        self.main, self.alt = _Stream(self, "main"), _Stream(self, "alt")
        self.current = self.main

    def op(self, name, result=None):
        s = self.current
        direct = set(s.deps) | set(self.issued[s.name])
        self.before[name] = (
            set(direct).union(*(self.before[d] for d in direct)) if direct else set()
        )
        self.stream_of[name] = s.name
        self.issued[s.name].append(name)
        self.order.append(name)
        return result

    def ordered(self, a, b):
        return a in self.before[b] or b in self.before[a]

    @contextlib.contextmanager
    def stream(self, s):
        prev, self.current = self.current, s
        try:
            yield
        finally:
            self.current = prev


@contextlib.contextmanager
def _env(order, extra=None, takes=lambda rows: True):
    """The order switch and the pull geometry cleared, then set as given."""
    env = {k: v for k, v in os.environ.items() if k != ENV and not k.startswith(PULL)}
    if order is not None:
        env[ENV] = order
    env.update(extra or {})
    with contextlib.ExitStack() as stack:
        for cached in (routed.configured_shared_ar_order, routed.fused_pull_tuning):
            cached.cache_clear()
            stack.callback(cached.cache_clear)
        stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
        stack.enter_context(mock.patch.object(policy, "takes", takes))
        yield


def _run(
    order,
    *,
    mega=True,
    sharded=True,
    tail=True,
    rows=80,
    fuse_norm=True,
    alt=True,
    scatters=False,
    env_extra=None,
):
    """One _forward_fused call. Returns the trace and what the collectives and the
    gemm_ag tail were called with (no "b_stream" key: the tail did not run)."""
    tr, got, buffers = _Trace(), {}, []

    def symm_buffer(name, n, width, dtype):
        buffers.append(torch.zeros(n, width, dtype=dtype))
        return buffers[-1]

    def reduce(x, *norm_params, num_tokens=None, **geometry):
        # the latent alone ([rows, LATENT]) or the whole pair from its first element
        whole = x.numel() == buffers[-1].numel()
        got.setdefault("norm_calls" if norm_params else "plain_calls", []).append(
            dict(
                shape=tuple(x.shape),
                num_tokens=num_tokens,
                geometry=geometry,
                whole=whole and x.data_ptr() == buffers[-1].data_ptr(),
            )
        )
        tr.op("fused_ar" if whole else "latent_ar")

    def fake_tail(latent, weight, b, c, *, b_stream=None):
        got["b_stream"] = b_stream
        offset = (b.data_ptr() - buffers[-1].data_ptr()) // b.element_size()
        got["tail_b"] = (offset, tuple(b.shape))  # where in the buffer the tail reads b
        tr.op("slice_gemm")
        if b_stream is not None:
            tr.current.wait_stream(b_stream)
        tr.op("gather_add3")
        return torch.zeros(rows, H)

    def boom(*args, **kwargs):
        raise AssertionError("an all-reduce or a deferred finalize on a scatter pass")

    collectives = dict(
        symm_buffer=symm_buffer,
        all_reduce_low_sm=lambda x, **kw: tr.op("shared_ar"),
        all_reduce_norm=reduce,
        all_reduce=reduce,
        finalize_all_reduce_push_norm=lambda *a: tr.op("latent_ar"),
        gemm_ag_up_proj=fake_tail,
    )
    if scatters:
        # prefill: no flat [latent | shared] buffer to all-reduce, no deferred finalize
        collectives = dict.fromkeys(collectives, boom)
    fusion = types.SimpleNamespace(
        MOE_LATENT_SHARED="moe_latent_shared",
        NORM_DIM=LATENT,
        enabled=lambda: True,
        finalize_push_fits=lambda n: n <= 73,
        gemm_ag_up_fits=lambda n: tail,
        **collectives,
    )
    sp = types.SimpleNamespace(
        SP_RS_OVERLAP_BLOCKS=4,
        sp_partial_rows=lambda name, like, width: torch.zeros(like.shape[0], width),
        sp_reduce_scatter=lambda x, blocks=None: tr.op(
            "shared_rs" if x.shape[1] == H else "latent_rs", x
        ),
    )

    class UpProj:
        weight = torch.zeros(H, LATENT)

        def __call__(self, latent):
            tr.op("up_proj")
            return torch.zeros(rows, H), None

    deferred = boom if scatters else lambda hs, logits, x: tr.op("routed", DEFERRED)
    me = types.SimpleNamespace(
        moe_hidden_size=LATENT,
        alt_stream=tr.alt if alt else None,
        experts=MEGA if mega else STOCK,
        _defer_moe_finalize=True,
        fuse_ar_norm=fuse_norm,
        _gemm_ag_up_eligible=fuse_norm,
        routed_expert_up_proj=UpProj(),
        _sharded_front_covers=lambda n: sharded,
        _sharded_front_gemm=lambda hs: tr.op("front_gemm", ("gate_up", "block")),
        _fused_front=lambda hs: tr.op("front_gemm", ("gate_up", "logits", "x")),
        _sharded_front_gather=lambda b, hs: tr.op("front_gather", ("logits", "x")),
        _forward_shared=lambda gate_up, out: tr.op("shared_gemm"),
        _forward_routed=lambda hs, logits, x, latent: tr.op("routed"),
        _forward_routed_deferred=deferred,
        _get_fused_norm_params=lambda: (torch.ones(LATENT), 1e-6),
        _latent_norm=lambda latent: tr.op("latent_norm", latent),
    )
    with (
        _env(order, env_extra),
        mock.patch.object(kimi_k3, "k3_ar_fusion", fusion),
        mock.patch.object(kimi_k3, "k3_sp_collective", sp),
        mock.patch.object(
            kimi_k3, "_add3", lambda out, b, c, prefetch_bc=False: tr.op("add3", out)
        ),
        mock.patch.object(torch.cuda, "current_stream", lambda: tr.current),
        mock.patch.object(torch.cuda, "stream", tr.stream),
    ):
        kimi_k3.KimiK3MoE._forward_fused(
            me, torch.zeros(rows, H), prefix_sum=None, scatters=scatters
        )
    return tr, got


class TestSharedArOrderPolicy(CustomTestCase):
    def test_default_and_values(self):
        for value, want in ((None, "fused"), *zip(ORDERS, ORDERS)):
            with self.subTest(value=value), _env(value):
                self.assertEqual(routed.shared_ar_order(MEGA, 80), want)
                # a call trtllm-gen takes keeps the stock order whatever the switch says
                self.assertEqual(routed.shared_ar_order(STOCK, 80), "overlap")
            with _env(value, takes=lambda rows: rows >= 80):  # the width table decides
                self.assertEqual(routed.shared_ar_order(MEGA, 40), "overlap")

    def test_invalid_order_or_pull_geometry(self):
        # a bad geometry fails with the order, on the first kernel-taken call
        for order, extra in (("sometimes", None), ("fused", {UNROLL: "3"})):
            with _env(order, extra), self.assertRaises(ValueError):
                routed.shared_ar_order(MEGA, 80)


class TestForwardFusedOrder(CustomTestCase):
    def assert_kernel_runs_alone(self, tr, behind):
        self.assertLessEqual(behind, tr.before["routed"])
        for other in tr.order:
            if other != "routed":
                self.assertTrue(tr.ordered("routed", other), other)

    def test_stock_call_is_unchanged(self):
        # trtllm-gen call: the shared AR is queued before the routed call, unordered
        # with it, whatever the switch says
        for order in (None, *ORDERS):
            for rows in (40, 80):
                with self.subTest(order=order, rows=rows):
                    tr, got = _run(order, mega=False, rows=rows)
                    self.assertEqual(tr.order, STOCK_ORDER)
                    self.assertEqual(
                        tr.waits, [("alt", "main"), ("alt", "main"), ("main", "alt")]
                    )
                    self.assertFalse(tr.ordered("shared_ar", "routed"))
                    self.assertIn("shared_ar", tr.before["latent_ar"])
                    self.assertIsNone(got["b_stream"])

    def test_overlap_reproduces_the_stock_order_for_a_mega_call(self):
        tr, _ = _run("overlap")
        ref, _ = _run("after", mega=False)
        self.assertEqual(
            (tr.order, tr.waits, tr.stream_of), (ref.order, ref.waits, ref.stream_of)
        )
        # the measured regression: the two race for the SMs
        self.assertFalse(tr.ordered("shared_ar", "routed"))

    def test_after(self):
        tr, got = _run("after")
        self.assertEqual(tr.issued["alt"], ["shared_gemm", "shared_ar"])
        self.assert_kernel_runs_alone(tr, behind={"shared_gemm", "front_gather"})
        # shared AR: behind the latent AR (same pull semaphores), next to the tail's
        # local GEMM, joined by the gather
        self.assertIn("latent_ar", tr.before["shared_ar"])
        self.assertFalse(tr.ordered("shared_ar", "slice_gemm"))
        self.assertIn("shared_ar", tr.before["gather_add3"])
        self.assertIn("latent_ar", tr.before["slice_gemm"])
        self.assertIs(got["b_stream"], tr.alt)
        # without the gemm_ag tail the shared AR joins before the up projection
        tr, got = _run("after", tail=False)
        self.assertNotIn("b_stream", got)
        self.assertIn("latent_ar", tr.before["shared_ar"])
        self.assertIn("shared_ar", tr.before["up_proj"])

    def test_before(self):
        tr, got = _run("before")
        self.assert_kernel_runs_alone(
            tr, behind={"shared_gemm", "front_gather", "shared_ar"}
        )
        # the AR still waits for the gather
        self.assertIn("front_gather", tr.before["shared_ar"])
        self.assertIn("shared_ar", tr.before["latent_ar"])
        self.assertIsNone(got["b_stream"])

    def test_unsharded_front(self):
        for order in ("after", "before"):
            tr, got = _run(order, sharded=False)
            self.assertLessEqual({"routed", "shared_gemm"}, tr.before["shared_ar"])
            self.assertIn("routed", tr.before["shared_gemm"])
            self.assertIn("shared_ar", tr.before["latent_ar"])
            self.assertIsNone(got["b_stream"])
        tr, _ = _run("overlap", sharded=False)
        self.assertFalse(tr.ordered("routed", "shared_ar"))

    def test_the_two_pull_all_reduces_never_overlap(self):
        for order in ORDERS:
            for case in range(8):
                mega, sharded, tail = (bool(case & bit) for bit in (4, 2, 1))
                with self.subTest(order=order, mega=mega, sharded=sharded, tail=tail):
                    tr, _ = _run(order, mega=mega, sharded=sharded, tail=tail)
                    reader = "gather_add3" if tail else "add3"
                    ars = [op for op in tr.order if op.endswith("_ar")]
                    if order == "fused" and mega:
                        # one collective behind both producers: no second pull to meet
                        self.assertEqual(ars, ["fused_ar"])
                        self.assertLessEqual(
                            {"shared_gemm", "routed"}, tr.before["fused_ar"]
                        )
                        self.assertIn("fused_ar", tr.before[reader])
                        continue
                    self.assertNotIn("fused_ar", ars)
                    self.assertTrue(tr.ordered("shared_ar", "latent_ar"))
                    self.assertLessEqual({"shared_ar", "latent_ar"}, tr.before[reader])

    def test_sequence_parallel_pass_keeps_the_stock_order(self):
        # scatters=True is prefill, which the MegaMoE kernel never takes
        for sharded in (True, False):
            ref, _ = _run(None, sharded=sharded, scatters=True)
            self.assertEqual(ref.stream_of["shared_rs"], "alt")
            # stock: the side-stream scatter overlaps the routed call
            self.assertFalse(ref.ordered("shared_rs", "routed"))
            self.assertIn("shared_rs", ref.before["latent_rs"])
            for order in ORDERS:
                tr, _ = _run(order, sharded=sharded, scatters=True)
                self.assertEqual(
                    (tr.order, tr.waits, tr.stream_of),
                    (ref.order, ref.waits, ref.stream_of),
                )

    def test_fused_sequence_and_streams(self):
        tr, got = _run("fused")
        self.assertEqual(tr.order, FRONT + ["routed", "fused_ar"] + TAIL)
        # the side stream runs the shared GEMM and is not used again
        self.assertEqual(tr.issued["alt"], ["shared_gemm"])
        self.assertNotIn(("alt", "main"), tr.waits[1:])
        self.assert_kernel_runs_alone(tr, behind={"shared_gemm", "front_gather"})
        # the whole tail follows the pull on its own stream and reads the shared half
        # of the buffer: no second stream to join
        self.assertIn("fused_ar", tr.before["slice_gemm"])
        self.assertIsNone(got["b_stream"])
        self.assertEqual(got["tail_b"], (80 * LATENT, (80, H)))

    def test_fused_collective_and_its_geometry(self):
        # all_reduce_pull_norm (ar_fusion.cuh): rows of the latent width, N latent rows
        # then the 2N half-rows of the shared output, num_norm_rows = N
        for rows, extra, geometry in (
            (80, None, {}),  # default: the kernel's own table
            (235, None, {}),
            (80, {BLOCKS: "16", UNROLL: "4"}, {"num_blocks": 16, "unroll": 4}),
            (80, {BLOCKS: "32"}, {"num_blocks": 32}),
        ):
            _, got = _run("fused", rows=rows, env_extra=extra)
            want = dict(
                shape=(3 * rows, LATENT), num_tokens=rows, geometry=geometry, whole=True
            )
            self.assertEqual(got["norm_calls"], [want])
            self.assertNotIn("plain_calls", got)
        for bad in ({UNROLL: "3"}, {BLOCKS: "0"}, {BLOCKS: "x"}):
            with self.assertRaises(ValueError):
                _run("fused", env_extra=bad)
        # the override belongs to the fused pull: the latent pull keeps the table
        for order in ("after", "before", "overlap"):
            _, got = _run(order, env_extra={BLOCKS: "16"})
            (call,) = got["norm_calls"]
            self.assertEqual((call["geometry"], call["whole"]), ({}, False))

    def test_fused_without_the_fused_norm(self):
        # no norm epilogue: a plain collective over the pair, then the unfused tail
        tr, got = _run("fused", fuse_norm=False)
        self.assertEqual(
            tr.order, FRONT + ["routed", "fused_ar", "latent_norm", "up_proj", "add3"]
        )
        (call,) = got["plain_calls"]
        self.assertEqual((call["shape"], call["whole"]), ((80 * (LATENT + H),), True))
        self.assertNotIn("norm_calls", got)


class TestGemmAgTailWaitsForB(CustomTestCase):
    def _call(self, rows, b_stream):
        from sglang.kernels.ops.communication import gemm_ag as mod

        tr = _Trace()
        with tr.stream(tr.alt):
            tr.op("shared_ar")
        state = types.SimpleNamespace(world_size=8, rank=1)
        with (
            mock.patch.object(k3_ar_fusion, "_get_state", lambda: state),
            mock.patch.object(
                k3_ar_fusion, "slice_gemm", lambda x, w, *a: tr.op("slice_gemm", x)
            ),
            mock.patch.object(mod, "gather_add3", lambda *a: tr.op("gather_add3")),
            mock.patch.object(mod, "gemm_ag_up_proj", lambda *a: tr.op("gemv_tail")),
            mock.patch.object(torch.cuda, "current_stream", lambda: tr.current),
        ):
            x, w = torch.zeros(rows, LATENT), torch.zeros(H, LATENT)
            kwargs = dict(b_stream=tr.alt) if b_stream else {}  # default: no keyword
            k3_ar_fusion.gemm_ag_up_proj(x, w, torch.zeros(rows, H), None, **kwargs)
        self.assertLessEqual(4, mod.GEMV_MAX_TOKENS)  # 4 rows take the fused GEMV tail
        return tr

    def test_only_the_reader_of_b_waits_for_its_stream(self):
        tr = self._call(80, True)
        self.assertEqual(tr.order, ["shared_ar", "slice_gemm", "gather_add3"])
        self.assertNotIn("shared_ar", tr.before["slice_gemm"])
        self.assertIn("shared_ar", tr.before["gather_add3"])
        tr = self._call(4, True)
        self.assertEqual(tr.order, ["shared_ar", "gemv_tail"])
        self.assertIn("shared_ar", tr.before["gemv_tail"])

    def test_default_adds_no_wait(self):
        for rows in (4, 80):
            self.assertEqual(self._call(rows, False).waits, [])


if __name__ == "__main__":
    unittest.main()
