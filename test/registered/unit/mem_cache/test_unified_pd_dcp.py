"""PD page coordinates must stay separate from DCP kernel-facing row IDs."""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.mem_cache.allocator.unified_mamba import (
    UnifiedMambaTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMambaPool,
    UnifiedMLATokenToKVPool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def make_allocator(ps):
    raw = UnifiedKVPool(
        total_bytes=65536,
        page_size=ps,
        device="cpu",
        enable_memory_saver=False,
        sub_pool_specs=[
            MLASubPoolSpec(
                name="full",
                layer_num=3,
                grow_direction="down",
                kv_lora_rank=6,
                qk_rope_head_dim=2,
                store_dtype=torch.bfloat16,
            ),
            MambaSubPoolSpec(
                name="mamba",
                layer_num=2,
                grow_direction="up",
                conv_state_shapes=((3, 8),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(2, 2, 4),
                temporal_dtype=torch.float32,
            ),
        ],
    )
    full = UnifiedMLATokenToKVPool(
        unified_buffer=raw,
        sub_pool_name="full",
        kv_cache_dtype=torch.bfloat16,
        page_size=ps,
    )
    state = UnifiedMambaPool(
        unified_buffer=raw,
        sub_pool_name="mamba",
        spec_state_size=8,
        mamba_layer_ids=[1, 3],
    )
    allocator = UnifiedMambaTokenToKVPoolAllocator(
        unified_buffer=raw,
        kvcache=SimpleNamespace(full_kv_pool=full, mamba_pool=state),
        device="cpu",
        page_size=ps,
        lazy_compaction=True,
    )
    # The raw buffer starts zeroed; hand-out zeroing uses a GPU-only kernel.
    allocator.full_attn_allocator._zero_pages_on_alloc = False
    return allocator, full, state


class TestUnifiedPDDCP(unittest.TestCase):
    def test_rank_local_bytes_with_prefix_chunks_and_partial_page(self):
        for ps in (1, 4):
            for dcp in (1, 2, 4):
                for rank in range(dcp):
                    with (
                        self.subTest(ps=ps, dcp=dcp, rank=rank),
                        get_parallel().override(
                            attn_dcp_size=dcp, attn_dcp_rank=rank, dcp_enabled=dcp > 1
                        ),
                    ):
                        src, src_pool, _ = make_allocator(ps)
                        dst, dst_pool, _ = make_allocator(ps)
                        width = src.page_size
                        # Interleave other requests so this request's page list is fragmented.
                        src_parts, dst_parts = [], []
                        for _ in range(4):
                            src_parts.append(src.alloc(width))
                            src.alloc(2 * width)
                            dst.alloc(width)
                            dst_parts.append(dst.alloc(width))
                        src_ids, dst_ids = torch.cat(src_parts), torch.cat(dst_parts)
                        length = 4 * width - 1
                        for pos in range(length):
                            if pos % dcp != rank:
                                continue
                            local = src.full_attn_allocator.translate_kv_loc(
                                src_ids[pos : pos + 1] // dcp
                            ).item()
                            kernel = (local // ps) * ps * 3 + local % ps
                            for layer, view in enumerate(src_pool.kv_buffer):
                                view[kernel].fill_(100 * layer + pos)
                        src_layout = src_pool.get_transfer_layout([0, 2, 4])
                        dst_layout = dst_pool.get_transfer_layout([0, 2, 4])
                        # A cached prefix is omitted; one complete middle chunk, then a partial final page.
                        for start, end in ((width, 2 * width), (2 * width, length)):
                            src_pages = kv_to_page_indices(
                                src.translate_kv_indices_for_transfer(
                                    src_ids[start:end]
                                ),
                                width,
                            )
                            dst_pages = kv_to_page_indices(
                                dst.translate_kv_indices_for_transfer(
                                    dst_ids[start:end]
                                ),
                                width,
                            )
                            self.assertEqual(len(src_pages), len(dst_pages))
                            for sp, dp in zip(src_pages, dst_pages):
                                ctypes.memmove(
                                    dst_pool.kv_buffer[0].data_ptr()
                                    + int(dp) * dst_layout.block_bytes,
                                    src_pool.kv_buffer[0].data_ptr()
                                    + int(sp) * src_layout.block_bytes,
                                    src_layout.block_bytes,
                                )
                        for pos in range(width, length):
                            if pos % dcp != rank:
                                continue
                            local = dst.full_attn_allocator.translate_kv_loc(
                                dst_ids[pos : pos + 1] // dcp
                            ).item()
                            kernel = (local // ps) * ps * 3 + local % ps
                            for layer, view in enumerate(dst_pool.kv_buffer):
                                expected = torch.full_like(
                                    view[kernel], 100 * layer + pos
                                )
                                self.assertTrue(torch.equal(view[kernel], expected))

    def test_compaction_retranslates_pages_and_keeps_state_slots_independent(self):
        with get_parallel().override(
            attn_dcp_size=2, attn_dcp_rank=1, dcp_enabled=True
        ):
            alloc, pool, _ = make_allocator(4)
            width = alloc.page_size
            first = alloc.alloc(2 * width)
            hole = alloc.alloc(width)
            last = alloc.alloc(2 * width)
            ids = torch.cat([first, last])
            state_id = alloc.mamba_allocator.alloc(1)
            state_before = alloc.mamba_allocator.translate_kv_loc(state_id).clone()
            before = alloc.translate_kv_indices_for_transfer(ids).clone()
            pages_before = kv_to_page_indices(before, width)
            for index, page in enumerate(pages_before):
                for layer, view in enumerate(pool.kv_buffer):
                    view[int(page) * 12 : int(page) * 12 + 4].fill_(10 * layer + index)
            idle = [False]
            alloc.set_disagg_move_gate(lambda: idle[0])
            alloc.free(hole)
            self.assertEqual(alloc.full_attn_allocator.flush_for_allocation(), 0)
            self.assertTrue(
                torch.equal(before, alloc.translate_kv_indices_for_transfer(ids))
            )
            idle[0] = True
            self.assertGreater(alloc.full_attn_allocator.flush_for_allocation(), 0)
            after = alloc.translate_kv_indices_for_transfer(ids)
            self.assertFalse(torch.equal(before, after))
            for index, page in enumerate(kv_to_page_indices(after, width)):
                for layer, view in enumerate(pool.kv_buffer):
                    self.assertTrue(
                        torch.all(
                            view[int(page) * 12 : int(page) * 12 + 4]
                            == 10 * layer + index
                        )
                    )
            self.assertTrue(
                torch.equal(
                    state_before, alloc.mamba_allocator.translate_kv_loc(state_id)
                )
            )
            self.assertEqual(alloc.mamba_allocator.page_size, 1)

    def test_matching_peer_guard(self):
        with get_parallel().override(
            attn_dcp_size=2, attn_dcp_rank=1, dcp_enabled=True
        ):
            _, kv, state = make_allocator(4)
        manager = object.__new__(MooncakeKVManager)
        manager.attn_tp_size, manager.pp_size = 4, 1
        manager.dcp_size, manager.dcp_rank = 2, 1
        manager.is_hybrid_mla_backend = True
        manager.kv_args = SimpleNamespace(
            unified_kv_layout=kv.get_transfer_layout([0, 2, 4]),
            unified_state_layout=state.get_transfer_layout(),
        )
        peer = SimpleNamespace(
            dst_attn_tp_size=4,
            dst_dcp_size=2,
            dst_dcp_rank=1,
            dst_unified_kv_layout=manager.kv_args.unified_kv_layout,
            dst_unified_state_layout=manager.kv_args.unified_state_layout,
        )
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.get_memory",
            return_value=SimpleNamespace(enable_unified_memory=True),
        ):
            manager._validate_unified_peer_layout(peer)
            for field, value in (
                ("dst_dcp_size", 1),
                ("dst_dcp_size", 4),
                ("dst_dcp_rank", 0),
                ("dst_attn_tp_size", 8),
                ("dst_unified_kv_layout", None),
            ):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(RuntimeError, "matching DCP"),
                ):
                    manager._validate_unified_peer_layout(
                        SimpleNamespace(**{**vars(peer), field: value})
                    )


if __name__ == "__main__":
    unittest.main()
