"""Bounded DSA replay metadata kernels write exactly what the unbounded ones write."""

import unittest
from functools import partial

import torch

from sglang.kernels.ops.attention import dsa_metadata_bounded as bounded
from sglang.kernels.ops.attention.dsa import metadata as unbounded
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_metadata_manager import (
    DSAMetadataManagementMixin,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.dsa_bounded_metadata_kit import (
    PAGE,
    TOPK,
    WIDTH_1M,
    MetadataWorld,
    clone_bufs,
    decode_bufs,
    live_columns,
    run_decode,
    run_draft_extend,
    run_verify,
    verify_bufs,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PROD = dict(
    program_budget=bounded.DEFAULT_PROGRAM_BUDGET,
    max_splits_per_row=bounded.DEFAULT_MAX_SPLITS_PER_ROW,
)
NOCAP = dict(program_budget=bounded.DEFAULT_PROGRAM_BUDGET, max_splits_per_row=0)
# Fewer programs than live blocks: every program loops over several blocks.
TINY = dict(program_budget=7, max_splits_per_row=256)
# Verify rows per request of EAGLE 4/1/5 and 5/1/6.
NEXT_NS = (5, 6)
BATCH_SIZES = (1, 2, 4, 8, 16)
EDGE_LENS = [1, 2, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049, 8192]


def kv_sets(width, seed=1):
    g = torch.Generator().manual_seed(seed)

    def rnd(lo, hi, n):
        return torch.randint(lo, hi, (n,), generator=g).tolist()

    return {
        "1k": lambda n: rnd(1024, 2048, n),
        "64k": lambda n: rnd(65536, 66560, n),
        "edge": lambda n: [EDGE_LENS[i % len(EDGE_LENS)] for i in range(n)],
        # At and past the table width: both grids clamp to the width.
        "wide": lambda n: ([width - 7, width - 1, width + 300] + rnd(1, 3000, n))[:n],
    }


def distinct_budgets(budgets, rows, width):
    seen, out = set(), []
    for budget in budgets:
        splits = bounded.bounded_num_splits(rows, width, **budget)
        if splits not in seen:
            seen.add(splits)
            out.append(budget)
    return out


class _BufferAsserts:
    def assert_same(self, expected, actual, tag):
        for name, ref in expected.items():
            got = actual[name]
            if torch.equal(ref, got):
                continue
            diff = (ref != got).nonzero()
            first = tuple(diff[0].tolist())
            self.fail(
                f"{tag}.{name}: {diff.shape[0]} elements differ, first at {first}: "
                f"{ref[first].item()} vs {got[first].item()}"
            )

    def assert_same_live(self, expected, actual, seq, rows_per_req, kv_add, width, tag):
        """Page tables on each row's live prefix; graph buffers keep stale tails."""
        for name, ref in expected.items():
            if ref.dim() == 1 or name == "paged_mqa_ctx_lens_2d":
                self.assert_same({name: ref}, {name: actual[name]}, tag)
        for r, kv_len in enumerate(seq.tolist()):
            cols, real_cols = live_columns(kv_len + kv_add, width)
            for row in range(r * rows_per_req, (r + 1) * rows_per_req):
                self.assert_same(
                    {
                        "page_table_1": expected["page_table_1"][row, :cols],
                        "real_page_table": expected["real_page_table"][row, :real_cols],
                    },
                    {
                        "page_table_1": actual["page_table_1"][row, :cols],
                        "real_page_table": actual["real_page_table"][row, :real_cols],
                    },
                    f"{tag}.row{row}",
                )


class TestBoundedMetadataKernels(CustomTestCase, _BufferAsserts):
    """Whole destination buffers, after identical sentinel fills, at a 1M-token
    table width: the bounded grid must write the same elements with the same
    values, including padded rows, lengths past the width and multi-pass loops."""

    @classmethod
    def setUpClass(cls):
        cls.world = MetadataWorld(WIDTH_1M, pool=40)

    def _batches(self, world, batch_sizes, seq_dtype=torch.int64):
        for bs in batch_sizes:
            for set_name, gen in kv_sets(world.width).items():
                kv = gen(bs)
                real = bs - 1 if bs > 2 else bs  # the last row is CUDA-graph padding
                seq, req = world.batch(kv[:real], pad_to=bs, seq_dtype=seq_dtype)
                yield bs, set_name, seq, req

    def test_decode(self):
        world = self.world
        for bs, set_name, seq, req in self._batches(world, BATCH_SIZES):
            for budget in distinct_budgets([PROD, NOCAP, TINY], bs, world.width):
                ref = decode_bufs(world, bs)
                got = clone_bufs(ref)
                run_decode(
                    unbounded.fused_dsa_decode_metadata, world, seq, req, bs, ref
                )
                run_decode(
                    bounded.fused_dsa_decode_metadata_bounded,
                    world,
                    seq,
                    req,
                    bs,
                    got,
                    **budget,
                )
                self.assert_same(ref, got, f"decode.bs{bs}.{set_name}.{budget}")

    def test_target_verify(self):
        world = self.world
        for next_n in NEXT_NS:
            for bs, set_name, seq, req in self._batches(world, BATCH_SIZES):
                for budget in distinct_budgets([PROD, NOCAP, TINY], bs, world.width):
                    for with_ctx in (True, False):
                        ref = verify_bufs(world, bs, next_n, with_ctx)
                        got = clone_bufs(ref)
                        run_verify(
                            unbounded.fused_dsa_target_verify_metadata,
                            world,
                            seq,
                            req,
                            bs,
                            next_n,
                            ref,
                        )
                        run_verify(
                            bounded.fused_dsa_target_verify_metadata_bounded,
                            world,
                            seq,
                            req,
                            bs,
                            next_n,
                            got,
                            **budget,
                        )
                        self.assert_same(
                            ref,
                            got,
                            f"verify.n{next_n}.bs{bs}.{set_name}.ctx{int(with_ctx)}.{budget}",
                        )

    def test_draft_extend_static_length(self):
        """The bounded kernel takes the static extend length as a scalar
        (extend_seq_lens=None) or from the tensor; both match the unbounded one."""
        world = self.world
        for next_n in NEXT_NS:
            for bs, set_name, seq, req in self._batches(world, BATCH_SIZES):
                ext = torch.full((bs,), next_n, dtype=torch.int32, device="cuda")
                for budget in distinct_budgets([PROD, NOCAP, TINY], bs, world.width):
                    ref = verify_bufs(world, bs, next_n)
                    scalar, tensor = clone_bufs(ref), clone_bufs(ref)
                    run_draft_extend(
                        unbounded.fused_dsa_draft_extend_metadata,
                        world,
                        seq,
                        req,
                        bs,
                        next_n,
                        ref,
                        ext,
                    )
                    for got, ext_arg in ((scalar, None), (tensor, ext)):
                        run_draft_extend(
                            bounded.fused_dsa_draft_extend_metadata_bounded,
                            world,
                            seq,
                            req,
                            bs,
                            next_n,
                            got,
                            ext_arg,
                            **budget,
                        )
                    tag = f"draft_extend.n{next_n}.bs{bs}.{set_name}.{budget}"
                    self.assert_same(ref, scalar, tag + ".scalar")
                    self.assert_same(ref, tensor, tag + ".tensor")

    def test_int32_seq_lens_and_tiny_budgets(self):
        world = MetadataWorld(5000, pool=20, seed=3)
        budgets = [
            dict(program_budget=1, max_splits_per_row=256),
            PROD,
            dict(program_budget=2048, max_splits_per_row=2),
        ]
        for bs, set_name, seq, req in self._batches(world, (3,), torch.int32):
            for budget in budgets:
                tag = f"int32.{set_name}.{budget}"
                ref = decode_bufs(world, bs)
                got = clone_bufs(ref)
                run_decode(
                    unbounded.fused_dsa_decode_metadata, world, seq, req, bs, ref
                )
                run_decode(
                    bounded.fused_dsa_decode_metadata_bounded,
                    world,
                    seq,
                    req,
                    bs,
                    got,
                    **budget,
                )
                self.assert_same(ref, got, tag + ".decode")
                ref = verify_bufs(world, bs, 6, with_ctx=True)
                got = clone_bufs(ref)
                run_verify(
                    unbounded.fused_dsa_target_verify_metadata,
                    world,
                    seq,
                    req,
                    bs,
                    6,
                    ref,
                )
                run_verify(
                    bounded.fused_dsa_target_verify_metadata_bounded,
                    world,
                    seq,
                    req,
                    bs,
                    6,
                    got,
                    **budget,
                )
                self.assert_same(ref, got, tag + ".verify")


class TestBoundedMetadataGraphReplay(CustomTestCase, _BufferAsserts):
    """The grid depends only on (rows, width, budget): one capture serves any
    replay-time KV lengths and request slots."""

    def test_replay_with_new_lengths(self):
        world = MetadataWorld(WIDTH_1M, pool=40, seed=5)
        next_n = 6
        for bs in (1, 3, 16):
            seq_buf, req_buf = world.batch([1000] * bs)
            vbufs = verify_bufs(world, bs, next_n, with_ctx=True)
            ebufs = verify_bufs(world, bs, next_n)
            dbufs = decode_bufs(world, bs)

            def body():
                run_verify(
                    bounded.fused_dsa_target_verify_metadata_bounded,
                    world,
                    seq_buf,
                    req_buf,
                    bs,
                    next_n,
                    vbufs,
                )
                run_draft_extend(
                    bounded.fused_dsa_draft_extend_metadata_bounded,
                    world,
                    seq_buf,
                    req_buf,
                    bs,
                    next_n,
                    ebufs,
                    None,
                )
                run_decode(
                    bounded.fused_dsa_decode_metadata_bounded,
                    world,
                    seq_buf,
                    req_buf,
                    bs,
                    dbufs,
                )

            body()  # compile outside the capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body()
            for trial, kv in enumerate(
                ([9000] * bs, [3] * bs, [70000] + [129] * (bs - 1))
            ):
                seq, req = world.batch(kv)
                seq_buf.copy_(seq)
                req_buf.copy_(req)
                graph.replay()
                torch.cuda.synchronize()
                tag = f"graph.bs{bs}.t{trial}"
                ref = verify_bufs(world, bs, next_n, with_ctx=True)
                run_verify(
                    unbounded.fused_dsa_target_verify_metadata,
                    world,
                    seq,
                    req,
                    bs,
                    next_n,
                    ref,
                )
                self.assert_same_live(
                    ref, vbufs, seq, next_n, next_n, world.width, tag + ".verify"
                )
                ref = verify_bufs(world, bs, next_n)
                ext = torch.full((bs,), next_n, dtype=torch.int32, device="cuda")
                run_draft_extend(
                    unbounded.fused_dsa_draft_extend_metadata,
                    world,
                    seq,
                    req,
                    bs,
                    next_n,
                    ref,
                    ext,
                )
                self.assert_same_live(
                    ref, ebufs, seq, next_n, 0, world.width, tag + ".draft_extend"
                )
                ref = decode_bufs(world, bs)
                run_decode(
                    unbounded.fused_dsa_decode_metadata, world, seq, req, bs, ref
                )
                self.assert_same_live(
                    ref, dbufs, seq, 1, 0, world.width, tag + ".decode"
                )
            del graph


class _Backend(DSAMetadataManagementMixin):
    def __init__(self, kpool):
        self.dsa_index_kpool = kpool
        self.physical_page_size = PAGE
        self.dsa_index_topk = TOPK


class TestBoundedMetadataSelection(CustomTestCase):
    """The backend takes the bounded kernels only where they apply (kpool 1, no
    KPool metadata fusion) and SGLANG_DSA_BOUNDED_REPLAY_METADATA=0 restores the
    unbounded ones."""

    def _select(self, *, switch, kpool):
        with envs.SGLANG_DSA_BOUNDED_REPLAY_METADATA.override(switch):
            backend = _Backend(kpool)
            backend._init_kpool_metadata_fusion()
        return backend

    def test_switch_and_kpool(self):
        on = self._select(switch=True, kpool=1)
        self.assertTrue(on.dsa_meta_bounded)
        for attr, fn in (
            ("_fused_decode_metadata", bounded.fused_dsa_decode_metadata_bounded),
            (
                "_fused_verify_metadata",
                bounded.fused_dsa_target_verify_metadata_bounded,
            ),
            (
                "_fused_draft_extend_metadata",
                bounded.fused_dsa_draft_extend_metadata_bounded,
            ),
        ):
            selected = getattr(on, attr)
            self.assertIsInstance(selected, partial)
            self.assertIs(selected.func, fn)
            self.assertEqual(selected.keywords, PROD)
        off = self._select(switch=False, kpool=1)
        self.assertFalse(off.dsa_meta_bounded)
        self.assertIs(off._fused_decode_metadata, unbounded.fused_dsa_decode_metadata)
        pooled = self._select(switch=True, kpool=4)
        self.assertFalse(pooled.dsa_meta_bounded)


if __name__ == "__main__":
    unittest.main()
