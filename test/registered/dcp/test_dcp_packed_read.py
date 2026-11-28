"""CPU unit tests for the DCP packed read (C1 + D2v2 + admission + reserve).

1. Layout: ``remap_topk_to_packed`` (the tensor form the indexer remap runs)
   must be element-identical to ``packed_row_of`` (the per-position reference)
   for every prefix length / dcp size / chunk count combination probed,
   including the -1 pad sentinels and the extend tail.
2. Admission: ``dcp_packed_read_plan`` is the packed read's only switch, so
   every refusal must fall back to the permuting path, and the decision is
   cached per forward so all layers agree on one path.
3. Chunked gather protocol: ``dcp_issue_gather_chunks`` must order every
   sub-collective's completion before the caller consumes the prefix or
   records its ready event, and chunks == 1 must stay synchronous.
4. Reserve: ``runtime_gather_reserve_bytes`` sizes the headroom the KV pool
   leaves free for the packed read and the two prefetch slots.

Usage:
    python -m pytest test_dcp_packed_read.py -v
    python test_dcp_packed_read.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu import dcp_extend_gather
from sglang.srt.hardware_backend.npu.attention import dcp_memory
from sglang.srt.hardware_backend.npu.dcp_extend_gather import (
    dcp_issue_gather_chunks,
    dcp_packed_kv_lens,
    dcp_packed_read_plan,
)
from sglang.srt.layers.dcp import layout
from sglang.srt.layers.dcp.layout import (
    DcpPackedReadPlan,
    packed_row_of,
    plan_dcp_packed_read,
    remap_topk_to_packed,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestPackedRowOf(unittest.TestCase):
    def test_scalar_matches_layout_per_position(self):
        for prefix_len, extend_len, dcp_size in [
            (64, 16, 1),
            (65, 16, 2),
            (64, 16, 4),
            (67, 5, 8),
            (128, 1, 8),
            (1, 1, 4),
        ]:
            plan = plan_dcp_packed_read(prefix_len, extend_len, dcp_size)
            for chunks in {1, plan.chunks, 3}:
                if chunks != 1 and chunks > plan.send_rows:
                    chunks = max(1, plan.send_rows)
                mine = DcpPackedReadPlan(
                    prefix_len=plan.prefix_len,
                    extend_len=plan.extend_len,
                    dcp_size=plan.dcp_size,
                    send_rows=plan.send_rows,
                    gathered_rows=plan.gathered_rows,
                    rows=plan.rows,
                    chunks=chunks,
                )
                for pos in list(range(-2, prefix_len + extend_len + 2)) + [-1]:
                    self.assertEqual(
                        packed_row_of(pos, mine),
                        self._reference_row_of(pos, mine),
                        f"prefix={prefix_len} ext={extend_len} dcp={dcp_size} "
                        f"chunks={chunks} pos={pos}",
                    )

    @staticmethod
    def _reference_row_of(pos, plan):
        """Independent brute-force reference: replay the gather layout.

        Rank r sends local rows [0, local_r) where local_r = prefix//C + (r <
        prefix%C), zero-padded to send_rows; the rank-major gather lands rank
        r's local row i at r*send_rows + i. Chunked layouts fill chunk-major
        blocks instead. Extend rows follow the gathered prefix.
        """
        if pos < 0:
            return pos
        if pos >= plan.prefix_len:
            return plan.gathered_rows + (pos - plan.prefix_len)
        c = plan.dcp_size
        r = pos % c
        i = pos // c
        if plan.chunks == 1:
            return r * plan.send_rows + i
        q, rem = divmod(plan.send_rows, plan.chunks)
        n = plan.chunks
        k = min(i // q, n - 1) if q > 0 else n - 1
        j = i - k * q
        sk = q + rem if k == n - 1 else q
        return c * (k * q) + r * sk + j


class TestRemapTopkToPacked(unittest.TestCase):
    def test_tensor_matches_scalar(self):
        gen = torch.Generator().manual_seed(0)
        for prefix_len, extend_len, dcp_size in [(64, 16, 2), (67, 5, 8), (300, 40, 4)]:
            plan = plan_dcp_packed_read(prefix_len, extend_len, dcp_size)
            for chunks in {1, plan.chunks}:
                mine = plan._replace(chunks=chunks)
                tokens, k = 7, 5
                # Random positions in [-1, prefix+extend): -1 pads stay -1.
                pos = torch.randint(
                    -1, prefix_len + extend_len, (tokens, k), generator=gen
                )
                remapped = remap_topk_to_packed(pos, mine)
                ref = torch.tensor(
                    [[packed_row_of(int(p), mine) for p in row] for row in pos.tolist()]
                )
                self.assertTrue(torch.equal(remapped, ref))

    def test_negative_sentinels_pass_through(self):
        plan = plan_dcp_packed_read(64, 16, 4)
        pos = torch.tensor([[-1, -1, 0, 65, 79]])
        out = remap_topk_to_packed(pos, plan)
        self.assertEqual(out[0, :2].tolist(), [-1, -1])
        self.assertEqual(int(out[0, 2]), packed_row_of(0, plan))
        self.assertEqual(int(out[0, 3]), packed_row_of(65, plan))
        self.assertEqual(int(out[0, 4]), packed_row_of(79, plan))


class TestChunkCountSeam(unittest.TestCase):
    def test_chunk_constant_reaches_the_plan(self):
        # layout.py holds the chunk count in a module constant so tests can
        # probe other layouts; if the plan stopped reading it, every layout
        # probe above would silently exercise the shipped constant only. The
        # per-position layout itself is pinned by TestPackedRowOf.
        with patch.object(layout, "_DCP_GATHER_CHUNKS", 1):
            plan = plan_dcp_packed_read(64, 16, 4)
        self.assertEqual(plan.chunks, 1)


class TestPlanSizes(unittest.TestCase):
    def test_sizes(self):
        plan = plan_dcp_packed_read(100, 10, 4)
        self.assertEqual(plan.send_rows, 25)
        self.assertEqual(plan.gathered_rows, 100)
        self.assertEqual(plan.rows, 110)
        plan = plan_dcp_packed_read(101, 10, 4)
        self.assertEqual(plan.send_rows, 26)
        self.assertEqual(plan.gathered_rows, 104)
        self.assertEqual(plan.rows, 114)

    def test_chunks_clamped_to_send_rows(self):
        plan = plan_dcp_packed_read(2, 1, 4)  # send_rows=1 < any chunk count
        self.assertEqual(plan.chunks, 1)


_DCP4 = SimpleNamespace(dcp_size=4)


def _packed_forward_batch(prefix_lens, extend_lens, *, crop_free=True):
    md = SimpleNamespace(
        packed_read_plan=None,
        packed_read_plan_done=False,
        extend_crop_free=crop_free,
        packed_kv_lens=None,
    )
    return SimpleNamespace(
        attn_dcp_metadata=md,
        extend_prefix_lens_cpu=prefix_lens,
        extend_seq_lens_cpu=extend_lens,
    )


class TestPackedReadAdmission(unittest.TestCase):
    def test_single_request_crop_free_accepts_and_caches(self):
        fb = _packed_forward_batch([100], [10])
        with patch.object(dcp_extend_gather, "get_parallel", return_value=_DCP4):
            plan = dcp_packed_read_plan(fb, index_topk=2048)
        self.assertIsNotNone(plan)
        self.assertEqual(
            (plan.prefix_len, plan.extend_len, plan.dcp_size), (100, 10, 4)
        )
        # Resolve-once: a later call this forward returns the cached plan even
        # though the batch metadata moved -- every layer takes the same path.
        fb.extend_prefix_lens_cpu = [200]
        self.assertIs(dcp_packed_read_plan(fb, index_topk=2048), plan)

    def test_refusals_fall_back(self):
        cases = [
            ("multi_request", _packed_forward_batch([100, 50], [10, 5]), 2048),
            ("no_topk", _packed_forward_batch([100], [10]), None),
            ("no_cpu_lens", _packed_forward_batch(None, [10]), 2048),
            ("crop_needed", _packed_forward_batch([100], [10], crop_free=False), 2048),
        ]
        for name, fb, topk in cases:
            with self.subTest(name=name):
                with patch.object(
                    dcp_extend_gather, "get_parallel", return_value=_DCP4
                ):
                    self.assertIsNone(dcp_packed_read_plan(fb, topk))

    def test_fp8_store_refuses(self):
        # Patching the detection helper, not the pool: what needs pinning is
        # the wiring (FP8 store -> permuting fallback), the detection itself
        # is a getattr on a live pool.
        fb = _packed_forward_batch([100], [10])
        with patch.object(dcp_extend_gather, "_dsa_kv_store_is_fp8", return_value=True):
            with patch.object(dcp_extend_gather, "get_parallel", return_value=_DCP4):
                self.assertIsNone(dcp_packed_read_plan(fb, index_topk=2048))

    def test_refusal_is_cached_for_the_forward(self):
        # A forward that refused (here: no top-k yet) must stay refused when a
        # later layer holds one -- resolving late would leave earlier layers'
        # top-k unremapped while later layers read a packed buffer.
        fb = _packed_forward_batch([100], [10])
        with patch.object(dcp_extend_gather, "get_parallel", return_value=_DCP4):
            self.assertIsNone(dcp_packed_read_plan(fb, index_topk=None))
            self.assertIsNone(dcp_packed_read_plan(fb, index_topk=2048))

    def test_kv_lens_built_once(self):
        fb = _packed_forward_batch([100], [10])
        with patch.object(dcp_extend_gather, "get_parallel", return_value=_DCP4):
            plan = dcp_packed_read_plan(fb, index_topk=2048)
        lens = dcp_packed_kv_lens(fb, plan, torch.device("cpu"))
        self.assertEqual(lens.tolist(), [plan.rows])
        self.assertIs(dcp_packed_kv_lens(fb, plan, torch.device("cpu")), lens)


class _DelayedGroup:
    """Stands in for the dcp GroupCoordinator: two fake ranks whose gathered
    bytes land only on wait() -- the protocol the ready event relies on."""

    device_group = None

    def __init__(self):
        self.waited = 0

    def all_gather_into_tensor(self, out, send):
        out.copy_(torch.cat([send, send + 100]))


def _delayed_async(group):
    def impl(out, send, **kwargs):
        def wait():
            group.waited += 1
            group.all_gather_into_tensor(out, send)

        return SimpleNamespace(wait=wait)

    return impl


class TestPackedGatherCompletion(unittest.TestCase):
    def test_chunked_output_is_ready_for_inline_read_or_ready_event(self):
        group = _DelayedGroup()
        out = torch.full((10, 1), -777)
        send = torch.arange(5).view(-1, 1)
        plan = SimpleNamespace(dcp_size=2, send_rows=5, gathered_rows=10, chunks=3)
        with patch("torch.distributed.all_gather_into_tensor", _delayed_async(group)):
            dcp_issue_gather_chunks(group, out, send, plan)
        expected = torch.tensor([0, 100, 1, 101, 2, 3, 4, 102, 103, 104]).view(-1, 1)
        self.assertTrue(torch.equal(out, expected))
        self.assertEqual(group.waited, 3)

    def test_single_chunk_preserves_synchronous_path(self):
        group = _DelayedGroup()
        out = torch.empty((10, 1), dtype=torch.int64)
        send = torch.arange(5).view(-1, 1)
        plan = SimpleNamespace(dcp_size=2, send_rows=5, gathered_rows=10, chunks=1)
        dcp_issue_gather_chunks(group, out, send, plan)
        self.assertTrue(torch.equal(out, torch.cat([send, send + 100])))
        self.assertEqual(group.waited, 0)


class TestRuntimeReserve(unittest.TestCase):
    def reserve(self, *, ep=1, streamed=False, prefill_requests=1):
        projection = torch.empty(1, dtype=torch.bfloat16)
        layer = SimpleNamespace(
            use_dsa=True,
            o_proj=SimpleNamespace(output_size=7168),
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        setattr(layer, "w_vc_stream_shard" if streamed else "w_vc_qrep", projection)
        parallel = SimpleNamespace(attn_tp_size=16, moe_ep_size=ep)
        with patch.object(dcp_memory, "get_parallel", return_value=parallel):
            return dcp_memory.runtime_gather_reserve_bytes(
                SimpleNamespace(modules=lambda: [layer]),
                8192,
                context_tokens=1048576,
                prefill_requests=prefill_requests,
            )

    def test_streamed_projection_still_reserves_runtime_storage(self):
        self.assertGreater(self.reserve(streamed=True), 1 << 30)
        self.assertEqual(self.reserve(streamed=True), self.reserve())

    def test_ep_native_headroom(self):
        self.assertEqual(self.reserve(ep=16) - self.reserve(), 1 << 30)

    def test_multi_request_extends_hold_two_prefix_copies(self):
        # Single-request extends admit the packed read (one prefix copy); with
        # more requests the generic layout needs the contiguous output plus the
        # rank-major scratch, so the reserve more than doubles.
        self.assertGreater(self.reserve(prefill_requests=2), 2 * self.reserve())


if __name__ == "__main__":
    unittest.main()
