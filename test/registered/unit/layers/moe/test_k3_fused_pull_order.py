"""Order of the all-reduces around a K3 routed-expert call under AR fusion: a call on
the MegaMoE kernel reduces latent and shared in one pull behind the kernel, every other
call keeps the stock order (no GPU: KimiK3MoE._forward_fused runs on a stand-in module
with recording streams)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe import k3_die_local_moe as policy
from sglang.srt.models import kimi_k3
from sglang.test.test_utils import CustomTestCase

# hidden = 2 x latent, as in K3 (7168 / 3584): the buffer is 3 norm-width rows per token
H, LATENT = 64, 32
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


def _run(
    *,
    mega=True,
    sharded=True,
    tail=True,
    rows=80,
    fuse_norm=True,
    scatters=False,
    takes=lambda rows: True,
):
    """One _forward_fused call; `takes` stands in for the width table. Returns the trace
    and what the collectives, the runner and the gemm_ag tail were called with."""
    tr, got, buffers = _Trace(), {}, []

    def symm_buffer(name, n, width, dtype):
        buffers.append(torch.zeros(n, width, dtype=dtype))
        return buffers[-1]

    def where(x):  # (element offset in the [latent | shared] buffer, shape)
        offset = (x.data_ptr() - buffers[-1].data_ptr()) // x.element_size()
        return offset, tuple(x.shape)

    def reduce(x, *norm_params, num_tokens=None):
        # the latent alone ([rows, LATENT]) or the whole pair from its first element
        whole = x.numel() == buffers[-1].numel()
        got.setdefault("norm_calls" if norm_params else "plain_calls", []).append(
            dict(where=where(x), num_tokens=num_tokens)
        )
        tr.op("fused_ar" if whole else "latent_ar")

    def fake_tail(latent, weight, b, c):
        got["tail"] = (where(latent), where(b))
        tr.op("slice_gemm")
        tr.op("gather_add3")
        return torch.zeros(rows, H)

    def deferred(hs, logits, x):
        got["deferred"] = True
        return tr.op("routed", DEFERRED)

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

    me = types.SimpleNamespace(
        moe_hidden_size=LATENT,
        alt_stream=tr.alt,
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
        _forward_routed_deferred=boom if scatters else deferred,
        _get_fused_norm_params=lambda: (torch.ones(LATENT), 1e-6),
        _latent_norm=lambda latent: tr.op("latent_norm", latent),
    )
    with (
        mock.patch.object(policy, "takes", takes),
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


def _tail_args(rows):
    """The gemm_ag tail's inputs: the latent and the shared halves of the buffer."""
    return (0, (rows, LATENT)), (rows * LATENT, (rows, H))


class TestFusedPullOrder(CustomTestCase):
    def assert_same_trace(self, tr, ref):
        self.assertEqual(
            (tr.order, tr.waits, tr.stream_of), (ref.order, ref.waits, ref.stream_of)
        )

    def test_stock_call_keeps_the_upstream_order(self):
        # trtllm-gen call: the shared AR is queued before the routed call, overlapping
        for rows in (40, 80):
            with self.subTest(rows=rows):
                tr, got = _run(mega=False, rows=rows)
                self.assertEqual(tr.order, STOCK_ORDER)
                self.assertEqual(
                    tr.waits, [("alt", "main"), ("alt", "main"), ("main", "alt")]
                )
                self.assertEqual(tr.issued["alt"], ["shared_gemm", "shared_ar"])
                self.assertFalse(tr.ordered("shared_ar", "routed"))
                self.assertIn("shared_ar", tr.before["latent_ar"])
                # up to 73 rows the finalize is folded into the push all-reduce
                self.assertEqual("deferred" in got, rows == 40)
                self.assertEqual(got["tail"], _tail_args(rows))
        # unsharded front: the shared GEMM and its AR follow on the side stream
        tr, _ = _run(mega=False, sharded=False)
        self.assertEqual(
            tr.order,
            ["front_gemm", "routed", "shared_gemm", "shared_ar", "latent_ar"] + TAIL,
        )
        self.assertFalse(tr.ordered("shared_ar", "routed"))
        self.assertIn("shared_ar", tr.before["latent_ar"])

    def test_width_outside_the_table_keeps_the_stock_order(self):
        for rows, sharded in ((40, True), (79, True), (79, False)):
            ref, ref_got = _run(mega=False, rows=rows, sharded=sharded)
            tr, got = _run(rows=rows, sharded=sharded, takes=lambda rows: rows >= 80)
            self.assert_same_trace(tr, ref)
            self.assertEqual(got, ref_got)

    def test_kernel_call_is_one_pull_behind_the_kernel(self):
        for sharded, rows in ((True, 80), (True, 235), (False, 80), (True, 40)):
            with self.subTest(sharded=sharded, rows=rows):
                tr, got = _run(sharded=sharded, rows=rows)
                front = FRONT if sharded else ["front_gemm", "shared_gemm"]
                self.assertEqual(tr.order, front + ["routed", "fused_ar"] + TAIL)
                # the side stream runs the shared GEMM and is not used again
                self.assertEqual(tr.issued["alt"], ["shared_gemm"])
                self.assertEqual(tr.waits[1:], [("main", "alt"), ("main", "alt")])
                # the kernel starts behind the shared GEMM with nothing else in flight
                for other in tr.order:
                    if other != "routed":
                        self.assertTrue(tr.ordered("routed", other), other)
                # all_reduce_pull_norm (ar_fusion.cuh): rows of the latent width, N
                # latent rows then the 2N half-rows of the shared output, norm on N
                want = dict(where=(0, (3 * rows, LATENT)), num_tokens=rows)
                self.assertEqual(got["norm_calls"], [want])
                self.assertNotIn("plain_calls", got)
                # the kernel finalizes in-kernel: never deferred, also at 40 rows
                self.assertNotIn("deferred", got)
                # the tail follows the pull and reads the normed latent from the buffer
                self.assertIn("fused_ar", tr.before["slice_gemm"])
                self.assertEqual(got["tail"], _tail_args(rows))

    def test_kernel_call_without_the_tail_or_the_fused_norm(self):
        tr, got = _run(tail=False)
        self.assertNotIn("tail", got)
        self.assertEqual(tr.order[-3:], ["fused_ar", "up_proj", "add3"])
        self.assertEqual(len(got["norm_calls"]), 1)
        # no norm epilogue: a plain collective over the pair, then the unfused tail
        tr, got = _run(fuse_norm=False)
        self.assertEqual(
            tr.order, FRONT + ["routed", "fused_ar", "latent_norm", "up_proj", "add3"]
        )
        want = dict(where=(0, (80 * (LATENT + H),)), num_tokens=None)
        self.assertEqual(got["plain_calls"], [want])
        self.assertNotIn("norm_calls", got)

    def test_sequence_parallel_pass_keeps_the_stock_order(self):
        # scatters=True is prefill, which the MegaMoE kernel never takes
        for sharded in (True, False):
            ref, _ = _run(mega=False, sharded=sharded, scatters=True)
            self.assertEqual(ref.stream_of["shared_rs"], "alt")
            # stock: the side-stream scatter overlaps the routed call
            self.assertFalse(ref.ordered("shared_rs", "routed"))
            self.assertIn("shared_rs", ref.before["latent_rs"])
            tr, _ = _run(sharded=sharded, scatters=True)
            self.assert_same_trace(tr, ref)


if __name__ == "__main__":
    unittest.main()
