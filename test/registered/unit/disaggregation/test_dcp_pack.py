import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.dcp_pack import (
    dcp_pack_buffer_bytes,
    dcp_pack_slice_tokens,
    try_pack_dcp_src,
)
from sglang.srt.disaggregation.common.utils import (
    build_dcp_token_transfer_plan,
    group_concurrent_contiguous,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _plan(*, src, dst, page_size, dcp_size, dcp_rank, **kwargs):
    return build_dcp_token_transfer_plan(
        np.asarray(src, dtype=np.int32),
        np.asarray(dst, dtype=np.int32),
        physical_page_size=page_size,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        **kwargs,
    )


class TestDcpTokenTransferPlan(CustomTestCase):
    def test_one_virtual_page_explicit_rows(self):
        # P=2, N=4. Prefill pages 5,2,11,4; decode virtual page 7.
        # pos 0..7 src rows: 10,11, 4,5, 22,23, 8,9
        # draft follows the target's DCP sharding (owner mask + loc // dcp):
        # each rank stores only its own shard at division rows 14,15 (page P=2)
        expected_target_src = {
            0: [10, 22],
            1: [11, 23],
            2: [4, 8],
            3: [5, 9],
        }
        seen_src = []
        for rank, src in expected_target_src.items():
            plan = _plan(
                src=[5, 2, 11, 4],
                dst=[7],
                page_size=2,
                dcp_size=4,
                dcp_rank=rank,
                num_kv_tokens=8,
            )
            np.testing.assert_array_equal(plan.draft_src_token_indices, src)
            np.testing.assert_array_equal(plan.draft_dst_token_indices, [14, 15])
            np.testing.assert_array_equal(plan.target_src_token_indices, src)
            np.testing.assert_array_equal(plan.target_dst_token_indices, [14, 15])
            seen_src.extend(plan.target_src_token_indices.tolist())
        self.assertEqual(sorted(seen_src), sorted([10, 11, 4, 5, 22, 23, 8, 9]))

    def test_second_chunk_crosses_dest_pages(self):
        # Prefix already filled one virtual page (P*N=4). This chunk's 4 tokens
        # start at dest pos 4 and spill onto the next division-row pages.
        # draft follows the target's DCP sharding, so both plans coincide.
        plan = _plan(
            src=[9, 3],
            dst=[4, 6],
            page_size=2,
            dcp_size=2,
            dcp_rank=0,
            src_page_offset=2,
            decode_prefix_len=4,
            num_kv_tokens=4,
        )
        np.testing.assert_array_equal(plan.draft_src_token_indices, [18, 6])
        np.testing.assert_array_equal(plan.draft_dst_token_indices, [12, 13])
        np.testing.assert_array_equal(plan.target_src_token_indices, [18, 6])
        np.testing.assert_array_equal(plan.target_dst_token_indices, [12, 13])

        plan_r1 = _plan(
            src=[9, 3],
            dst=[4, 6],
            page_size=2,
            dcp_size=2,
            dcp_rank=1,
            src_page_offset=2,
            decode_prefix_len=4,
            num_kv_tokens=4,
        )
        np.testing.assert_array_equal(plan_r1.draft_src_token_indices, [19, 7])
        np.testing.assert_array_equal(plan_r1.draft_dst_token_indices, [12, 13])
        np.testing.assert_array_equal(plan_r1.target_src_token_indices, [19, 7])
        np.testing.assert_array_equal(plan_r1.target_dst_token_indices, [12, 13])

    def test_rejects_unaligned_prefix(self):
        with self.assertRaisesRegex(ValueError, "align"):
            _plan(
                src=[0],
                dst=[0],
                page_size=2,
                dcp_size=4,
                dcp_rank=0,
                decode_prefix_len=1,
                num_kv_tokens=2,
            )

    def test_empty_tokens(self):
        plan = _plan(
            src=[0], dst=[0], page_size=2, dcp_size=4, dcp_rank=0, num_kv_tokens=0
        )
        self.assertTrue(plan.empty())


class TestPackedDcpGrouping(CustomTestCase):
    def test_draft_packs_like_target(self):
        plan = _plan(
            src=[0, 1, 2, 3],
            dst=[0],
            page_size=2,
            dcp_size=4,
            dcp_rank=0,
            num_kv_tokens=8,
        )
        np.testing.assert_array_equal(plan.target_src_token_indices, [0, 4])
        np.testing.assert_array_equal(plan.target_dst_token_indices, [0, 1])
        target_src, _ = group_concurrent_contiguous(
            plan.target_src_token_indices, plan.target_dst_token_indices
        )
        self.assertEqual(target_src, [[0], [4]])

        packed_src, packed_dst = group_concurrent_contiguous(
            np.arange(2, dtype=np.int64), plan.target_dst_token_indices
        )
        self.assertEqual(packed_src, [[0, 1]])
        self.assertEqual(packed_dst, [[0, 1]])

        draft_src, draft_dst = group_concurrent_contiguous(
            plan.draft_src_token_indices, plan.draft_dst_token_indices
        )
        self.assertEqual(draft_src, [[0], [4]])
        self.assertEqual(draft_dst, [[0], [1]])


def _dcp_kv_manager_stub(*, page_size, kv_item_lens, num_draft_entries):
    return SimpleNamespace(
        kv_args=SimpleNamespace(
            page_size=page_size,
            kv_item_lens=kv_item_lens,
            num_draft_entries=num_draft_entries,
        )
    )


class TestPrepareDcpTokenItemLens(CustomTestCase):
    def test_draft_tail_scales_by_dst_dcp_size(self):
        mgr = _dcp_kv_manager_stub(
            page_size=64,
            kv_item_lens=[64 * 32, 64 * 32, 64 * 16],
            num_draft_entries=1,
        )
        token_lens = CommonKVManager.prepare_dcp_token_item_lens(
            mgr, [64 * 32, 64 * 32, 4 * 64 * 16], dst_dcp_size=4
        )
        self.assertEqual(token_lens, [32, 32, 16])

    def test_rejects_unscaled_draft_item_len(self):
        mgr = _dcp_kv_manager_stub(
            page_size=64,
            kv_item_lens=[64 * 32, 64 * 16],
            num_draft_entries=1,
        )
        with self.assertRaisesRegex(RuntimeError, "geometry differs at entry 1"):
            CommonKVManager.prepare_dcp_token_item_lens(
                mgr, [64 * 32, 64 * 16], dst_dcp_size=4
            )


class TestDcpPackBufferBytes(CustomTestCase):
    def test_sizes_fixed_regions_for_each_dcp_rank(self):
        self.assertEqual(
            dcp_pack_buffer_bytes(
                [64 * 16, 64 * 16],
                page_size=64,
                max_tokens=10,
                dcp_size=4,
            ),
            4 * 3 * (16 + 16),
        )

    def test_rejects_invalid_item_lens(self):
        with self.assertRaisesRegex(ValueError, "at least one page"):
            dcp_pack_buffer_bytes([0], page_size=64, max_tokens=8)
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            dcp_pack_buffer_bytes([100], page_size=64, max_tokens=8)


class TestTryDcpPack(CustomTestCase):
    def test_try_pack_uses_requested_region_and_dense_indices(self):
        dim = 4
        kv = torch.arange(16 * dim, dtype=torch.float32).view(16, 1, dim)
        item_len = int(kv[0].nbytes)
        pack = torch.zeros(8 * item_len, dtype=torch.uint8)
        gather_stream = Mock()
        buf = type(
            "Buf",
            (),
            {
                "buffer": pack,
                "fits": lambda self, n: n <= pack.numel(),
                "get_ptr": lambda self: 0x1000,
                "get_size": lambda self: pack.numel(),
                "get_gather_stream": lambda self: gather_stream,
            },
        )()
        src = np.array([1, 5, 9, 13], dtype=np.int64)
        pack_offset = 2 * item_len
        with (
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.torch.cuda.default_stream"
            ),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.torch.cuda.stream",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.disaggregation.common.dcp_pack.copy_mla_rows_into_pack"
            ) as copy_mock,
        ):
            packed = try_pack_dcp_src(
                pack_buffer=buf,
                kv_data_ptrs=[kv.data_ptr()],
                src_token_indices=src,
                token_item_lens=[item_len],
                pack_offset_bytes=pack_offset,
            )

        gather_stream.synchronize.assert_called_once_with()
        self.assertIsNotNone(packed)
        ptrs, indices = packed
        self.assertEqual(ptrs, [0x1000 + pack_offset])
        np.testing.assert_array_equal(indices, np.arange(4))
        pack_view = copy_mock.call_args.args[2]
        self.assertEqual(pack_view.storage_offset(), pack_offset)
        self.assertEqual(pack_view.numel(), src.size * item_len)


class TestDcpPackSliceTokens(CustomTestCase):
    def test_capacity_divides_buffer_by_per_token_bytes(self):
        # 24 layers x 576 B = 13824 B/token; 226,492,416 B buffer -> 16,384.
        self.assertEqual(dcp_pack_slice_tokens(24 * 576 * 16384, [576] * 24), 16384)
        # Leftover room smaller than one token is dropped.
        self.assertEqual(dcp_pack_slice_tokens(100, [60, 30]), 1)

    def test_empty_item_lens_means_no_packing(self):
        self.assertEqual(dcp_pack_slice_tokens(1024, []), 0)


def _fake_pack_buffer(size_bytes, *, fits=True):
    return SimpleNamespace(
        buffer=torch.zeros(size_bytes, dtype=torch.uint8),
        fits=lambda n: fits and n <= size_bytes,
        get_ptr=lambda: 0x1000,
        get_size=lambda: size_bytes,
        get_gather_stream=lambda: Mock(),
    )


@contextmanager
def _patched_dcp_pack_cuda():
    with (
        patch("sglang.srt.disaggregation.common.dcp_pack.torch.cuda.default_stream"),
        patch(
            "sglang.srt.disaggregation.common.dcp_pack.torch.cuda.stream",
            return_value=nullcontext(),
        ),
        patch(
            "sglang.srt.disaggregation.common.dcp_pack.copy_mla_rows_into_pack"
        ) as copy_mock,
    ):
        yield copy_mock


def _bare_sender():
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

    sender = MooncakeKVManager.__new__(MooncakeKVManager)
    sender.enable_custom_mem_pool = False
    return sender


class TestSendKvcacheDcpPackedSlices(CustomTestCase):
    """_send_kvcache_dcp_packed_slices: slice sizing, ordering, fallback."""

    ITEM_LEN = 16  # bytes per token per layer
    NUM_LAYERS = 2

    def _run(self, *, n_tokens, capacity_tokens, run_rets=None, fits=True):
        per_token = self.ITEM_LEN * self.NUM_LAYERS
        buf = _fake_pack_buffer(capacity_tokens * per_token, fits=fits)
        src_indices = 100 + 7 * np.arange(n_tokens, dtype=np.int64)
        dst_indices = np.arange(200, 200 + n_tokens, dtype=np.int64)
        src_ptrs = [0xAAAA0000 + i * 0x10000 for i in range(self.NUM_LAYERS)]
        dst_ptrs = [0xBBBB0000 + i * 0x10000 for i in range(self.NUM_LAYERS)]
        item_lens = [self.ITEM_LEN] * self.NUM_LAYERS

        events = []

        def run_layers(layers_params):
            events.append(("run", layers_params))
            if run_rets:
                return run_rets[len([e for e in events if e[0] == "run"]) - 1]
            return 0

        with _patched_dcp_pack_cuda() as copy_mock:

            def fake_gather(*args, **kwargs):
                events.append(("gather", np.asarray(args[1]).copy()))

            copy_mock.side_effect = fake_gather
            sender = _bare_sender()
            with patch(
                "sglang.srt.disaggregation.mooncake.conn.logger.warning"
            ) as warn_mock:
                done, ret = sender._send_kvcache_dcp_packed_slices(
                    "session",
                    pack_buffer=buf,
                    target_src_kv_ptrs=src_ptrs,
                    src_token_indices=src_indices,
                    dst_token_indices=dst_indices,
                    dst_kv_ptrs=dst_ptrs,
                    token_item_lens=item_lens,
                    run_layers=run_layers,
                )
        return (
            done,
            ret,
            events,
            warn_mock,
            (src_indices, dst_indices, src_ptrs, dst_ptrs),
        )

    def test_single_slice_matches_legacy_pack_path(self):
        done, ret, events, warn_mock, ctx = self._run(n_tokens=5, capacity_tokens=8)
        src_indices, dst_indices, src_ptrs, dst_ptrs = ctx
        self.assertTrue(done)
        self.assertEqual(ret, 0)
        warn_mock.assert_not_called()
        # One gather of all 5 tokens, then exactly one transfer round.
        self.assertEqual([e[0] for e in events], ["gather", "run"])
        np.testing.assert_array_equal(events[0][1], src_indices)
        layers_params = events[1][1]
        self.assertEqual(len(layers_params), self.NUM_LAYERS)
        for entry, (src_ptr, dst_ptr, item_len, groups) in enumerate(layers_params):
            # Packed src rows are dense [0..n) inside the per-entry region.
            self.assertEqual(src_ptr, 0x1000 + entry * 5 * self.ITEM_LEN)
            self.assertEqual(dst_ptr, dst_ptrs[entry])
            self.assertEqual(item_len, self.ITEM_LEN)
            src_groups, dst_groups = groups
            self.assertEqual(src_groups, [list(range(5))])
            np.testing.assert_array_equal(np.asarray(dst_groups[0]), dst_indices)

    def test_multi_slice_ordering_and_address_pairing(self):
        done, ret, events, warn_mock, ctx = self._run(n_tokens=10, capacity_tokens=4)
        src_indices, dst_indices, src_ptrs, dst_ptrs = ctx
        self.assertTrue(done)
        self.assertEqual(ret, 0)
        warn_mock.assert_not_called()
        # Slices of 4, 4, 2 tokens; each gather is followed by its transfer
        # before the next gather reuses the buffer.
        self.assertEqual(
            [e[0] for e in events],
            ["gather", "run", "gather", "run", "gather", "run"],
        )
        for slice_idx, (begin, end) in enumerate([(0, 4), (4, 8), (8, 10)]):
            np.testing.assert_array_equal(
                events[2 * slice_idx][1], src_indices[begin:end]
            )
            layers_params = events[2 * slice_idx + 1][1]
            n_slice = end - begin
            for entry, (src_ptr, dst_ptr, _, groups) in enumerate(layers_params):
                self.assertEqual(src_ptr, 0x1000 + entry * n_slice * self.ITEM_LEN)
                self.assertEqual(dst_ptr, dst_ptrs[entry])
                src_groups, dst_groups = groups
                self.assertEqual(src_groups, [list(range(n_slice))])
                np.testing.assert_array_equal(
                    np.asarray(dst_groups[0]), dst_indices[begin:end]
                )

    def test_transfer_failure_stops_remaining_slices(self):
        done, ret, events, warn_mock, _ = self._run(
            n_tokens=10, capacity_tokens=4, run_rets=[7]
        )
        self.assertTrue(done)
        self.assertEqual(ret, 7)
        self.assertEqual([e[0] for e in events], ["gather", "run"])

    def test_tiny_buffer_falls_back_without_packing(self):
        done, ret, events, warn_mock, _ = self._run(n_tokens=4, capacity_tokens=0)
        self.assertFalse(done)
        self.assertEqual(ret, 0)
        self.assertEqual(events, [])
        warn_mock.assert_not_called()

    def test_pack_misfit_falls_back_with_warning(self):
        done, ret, events, warn_mock, _ = self._run(
            n_tokens=4, capacity_tokens=4, fits=False
        )
        self.assertFalse(done)
        self.assertEqual(ret, 0)
        warn_mock.assert_called_once()
        self.assertIn("per-token RDMA", warn_mock.call_args.args[0])
        # The misfit is detected before any gather or transfer.
        self.assertEqual(events, [])


class TestSendKvcacheDcpSlicedEndToEnd(CustomTestCase):
    """send_kvcache_dcp with a real plan: sliced target + paired dst blocks."""

    def _send(self, *, n_pages, capacity_tokens, with_pack_buffer=True):
        page_size, dcp_size, dcp_rank = 2, 2, 0
        num_target, num_draft = 2, 1
        item_len = 16
        src_pages = np.arange(10, 10 + n_pages, dtype=np.int32)
        # Rank 0 owns offsets 0, 2, 4, ... -> n_pages tokens of dst shard.
        dst_pages = np.arange(50, 50 + n_pages, dtype=np.int32)

        src_ptrs = [0xAAAA0000 + i * 0x10000 for i in range(num_target + num_draft)]
        dst_ptrs = [0xBBBB0000 + i * 0x10000 for i in range(num_target + num_draft)]
        item_lens = [item_len] * (num_target + num_draft)

        sender = _bare_sender()
        sender.kv_args = SimpleNamespace(
            page_size=page_size,
            kv_data_ptrs=src_ptrs,
            kv_layer_ids=[],
            num_draft_entries=num_draft,
        )
        transfer_calls = []

        def fake_transfer(session_id, blocks):
            transfer_calls.append(blocks)
            return 0

        sender._transfer_data = fake_transfer
        pack_buffer = (
            _fake_pack_buffer(capacity_tokens * item_len * num_target)
            if with_pack_buffer
            else None
        )
        with _patched_dcp_pack_cuda():
            ret = sender.send_kvcache_dcp(
                "session",
                src_pages,
                dst_ptrs,
                dst_pages,
                dcp_token_item_lens=item_lens,
                dst_dcp_size=dcp_size,
                dst_dcp_rank=dcp_rank,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=n_pages * page_size,
                executor=Mock(),
                dst_layer_ids=[],
                pack_buffer=pack_buffer,
            )
        self.assertEqual(ret, 0)
        plan = build_dcp_token_transfer_plan(
            src_pages,
            dst_pages,
            physical_page_size=page_size,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=n_pages * page_size,
        )
        return transfer_calls, dst_ptrs, item_len, num_target, plan

    @staticmethod
    def _covered_dst_tokens(calls, dst_ptr):
        """Token positions written under one layer's dst base pointer."""
        positions = []
        for blocks in calls:
            for _, dst, length in blocks:
                if dst_ptr <= dst < dst_ptr + 0x10000:
                    off = dst - dst_ptr
                    positions.extend(range(off // 16, (off + length) // 16))
        return sorted(positions)

    def test_small_request_single_shot_unchanged(self):
        # 6 pages = 12 tokens -> rank shard 6 tokens <= capacity 8: one shot.
        calls, dst_ptrs, item_len, num_target, plan = self._send(
            n_pages=6, capacity_tokens=8
        )
        dst_tokens = plan.target_dst_token_indices.tolist()
        self.assertEqual(len(calls), 2)  # packed target, then draft
        target_blocks, draft_blocks = calls
        # Packed target: one dense block per layer, n_rank * item_len bytes.
        self.assertEqual(len(target_blocks), num_target)
        for i, (src, dst, length) in enumerate(target_blocks):
            self.assertEqual(dst, dst_ptrs[i] + dst_tokens[0] * item_len)
            self.assertEqual(length, len(dst_tokens) * item_len)
        # Draft still uses per-token groups (unchanged legacy behavior).
        self.assertEqual(len(draft_blocks), len(dst_tokens))
        for entry in range(num_target):
            self.assertEqual(
                self._covered_dst_tokens(calls[:1], dst_ptrs[entry]),
                dst_tokens,
            )

    def test_large_request_is_sliced_not_per_token(self):
        # 20 pages = 40 tokens -> rank shard 20 tokens > capacity 4:
        # 5 packed slices + 1 draft call, never a per-token target block.
        calls, dst_ptrs, item_len, num_target, plan = self._send(
            n_pages=20, capacity_tokens=4
        )
        dst_tokens = plan.target_dst_token_indices.tolist()
        self.assertEqual(len(dst_tokens), 20)
        self.assertEqual(len(calls), 6)
        for slice_call in calls[:5]:
            self.assertEqual(len(slice_call), num_target)  # dense per layer
            for _, _, length in slice_call:
                self.assertEqual(length, 4 * item_len)
        for entry in range(num_target):
            self.assertEqual(
                self._covered_dst_tokens(calls[:5], dst_ptrs[entry]),
                dst_tokens,
            )
        # Draft: 20 per-token blocks, unchanged.
        self.assertEqual(len(calls[5]), 20)

    def test_no_pack_buffer_keeps_per_token_fallback(self):
        calls, dst_ptrs, item_len, num_target, plan = self._send(
            n_pages=6, capacity_tokens=0, with_pack_buffer=False
        )
        self.assertEqual(len(calls), 1)  # target + draft in one batch
        # 6 per-token blocks x (2 target + 1 draft) layers.
        self.assertEqual(len(calls[0]), 6 * (num_target + 1))


if __name__ == "__main__":
    unittest.main()
