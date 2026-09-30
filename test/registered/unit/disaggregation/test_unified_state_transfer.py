"""Copy real unified tensors through the Mooncake state path with a CPU wire."""

import ctypes
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import (
    build_unified_state_transfer_blocks,
    validate_unified_state_layouts,
)
from sglang.srt.mem_cache.layout.transfer import TransferLayout
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMambaPool,
    UnifiedMLATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def make_pool(tp, axis, groups):
    channels = sum(groups) if groups else 16
    conv_shape = (3, channels // tp) if axis == 1 else (channels // tp, 3)
    raw = UnifiedKVPool(
        total_bytes=65536,
        device="cpu",
        page_size=4,
        enable_memory_saver=False,
        sub_pool_specs=[
            MLASubPoolSpec(
                name="full",
                layer_num=2,
                grow_direction="down",
                kv_lora_rank=6,
                qk_rope_head_dim=2,
                store_dtype=torch.bfloat16,
            ),
            MambaSubPoolSpec(
                name="mamba",
                layer_num=2,
                grow_direction="up",
                conv_state_shapes=(conv_shape,),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(8 // tp, 2, 4),
                temporal_dtype=torch.float32,
                conv_slice_axis=axis,
                conv_shard_groups=groups,
            ),
        ],
    )
    state = UnifiedMambaPool(
        unified_buffer=raw,
        sub_pool_name="mamba",
        spec_state_size=8,
        mamba_layer_ids=[1, 3],
    )
    full = UnifiedMLATokenToKVPool(
        unified_buffer=raw,
        sub_pool_name="full",
        kv_cache_dtype=torch.bfloat16,
        page_size=4,
    )
    raw._raw.fill_(0xA5)
    return raw, state, full


def reference_shard(tensor, axis, tp, rank, groups=None):
    pieces = tensor.split(groups, dim=axis) if groups else [tensor]
    return torch.cat([piece.chunk(tp, dim=axis)[rank] for piece in pieces], dim=axis)


class TestUnifiedStateTransfer(unittest.TestCase):
    def _run(self, src_tp, dst_tp, axis, groups):
        sources = [make_pool(src_tp, axis, groups) for _ in range(src_tp)]
        destinations = [make_pool(dst_tp, axis, groups) for _ in range(dst_tp)]
        channels = sum(groups) if groups else 16
        conv_shape = (3, channels) if axis == 1 else (channels, 3)
        global_states = [
            torch.arange(3 * channels).reshape(conv_shape).to(torch.bfloat16),
            torch.arange(64).reshape(8, 2, 4).to(torch.float32),
        ]
        for src_rank, (_, pool, _) in enumerate(sources):
            for field, view in enumerate(
                [*pool.mamba_cache.conv, pool.mamba_cache.temporal]
            ):
                for layer in range(2):
                    expected = reference_shard(
                        global_states[field] + 100 * layer,
                        axis if field == 0 else 0,
                        src_tp,
                        src_rank,
                        groups if field == 0 else None,
                    )
                    view[layer, 2 + src_rank].copy_(expected)
        writes = [set() for _ in destinations]
        for dst_rank, (dst_raw, dst_state, dst_kv) in enumerate(destinations):
            dst_layout = dst_state.get_transfer_layout()
            peer = SimpleNamespace(
                dst_attn_tp_size=dst_tp,
                dst_tp_rank=dst_rank,
                dst_dcp_size=1,
                dst_unified_kv_layout=dst_kv.get_transfer_layout([0, 2]),
                dst_unified_state_layout=TransferLayout.from_dict(
                    json.loads(json.dumps(dst_layout.to_dict()))
                ),
                dst_state_data_ptrs=[[dst_raw._raw.data_ptr()]],
                dst_state_item_lens=[[dst_layout.block_bytes]],
                dst_state_dim_per_tensor=[[]],
                dst_state_layer_ids=[[]],
            )
            for src_rank, (src_raw, src_state, src_kv) in enumerate(sources):
                if src_rank // max(1, src_tp // dst_tp) != dst_rank // max(
                    1, dst_tp // src_tp
                ):
                    continue
                manager = object.__new__(MooncakeKVManager)
                manager.attn_tp_size, manager.pp_size, manager.dcp_size = src_tp, 1, 1
                manager.is_hybrid_mla_backend = True
                src_layout = src_state.get_transfer_layout()
                manager.kv_args = SimpleNamespace(
                    engine_rank=src_rank,
                    unified_kv_layout=src_kv.get_transfer_layout([0, 2]),
                    unified_state_layout=src_layout,
                    state_types=[StateType.MAMBA],
                    state_data_ptrs=[[src_raw._raw.data_ptr()]],
                    state_item_lens=[[src_layout.block_bytes]],
                    state_dim_per_tensor=[[]],
                    state_layer_ids=[[]],
                )

                def copy(_session, blocks):
                    for src, dst, length in blocks:
                        offset = dst - dst_raw._raw.data_ptr()
                        touched = set(range(offset, offset + length))
                        self.assertFalse(
                            writes[dst_rank] & touched, "contributors overlap"
                        )
                        writes[dst_rank].update(touched)
                        ctypes.memmove(dst, src, length)
                    return 0

                manager._transfer_data = copy
                with patch(
                    "sglang.srt.disaggregation.mooncake.conn.get_memory",
                    return_value=SimpleNamespace(enable_unified_memory=True),
                ):
                    manager._validate_unified_peer_layout(peer)
                    self.assertEqual(
                        manager.maybe_send_extra(
                            SimpleNamespace(
                                mooncake_session_id="cpu",
                                dst_state_indices=[[7 - dst_rank]],
                            ),
                            [[2 + src_rank]],
                            None,
                            peer,
                        ),
                        0,
                    )
            for field, view in enumerate(
                [*dst_state.mamba_cache.conv, dst_state.mamba_cache.temporal]
            ):
                for layer in range(2):
                    expected = reference_shard(
                        global_states[field] + 100 * layer,
                        axis if field == 0 else 0,
                        dst_tp,
                        dst_rank,
                        groups if field == 0 else None,
                    )
                    self.assertTrue(torch.equal(view[layer, 7 - dst_rank], expected))
            outside = torch.ones(dst_raw._raw.numel(), dtype=torch.bool)
            outside[list(writes[dst_rank])] = False
            self.assertTrue(torch.all(dst_raw._raw[outside] == 0xA5))

    def test_scatter_and_gather_tensor_bytes(self):
        for src_tp, dst_tp in ((2, 4), (4, 2)):
            for axis, groups in ((1, None), (0, (8, 8, 16)), (1, (8, 8, 16))):
                with self.subTest(
                    src_tp=src_tp, dst_tp=dst_tp, axis=axis, groups=groups
                ):
                    self._run(src_tp, dst_tp, axis, groups)

    def test_mla_kv_keeps_whole_pages_across_tp_sizes(self):
        for src_tp, dst_tp in ((2, 4), (4, 2)):
            src_raw, _, src_kv = make_pool(src_tp, 1, None)
            dst_raw, _, dst_kv = make_pool(dst_tp, 1, None)
            for layer, view in enumerate(src_kv.kv_buffer):
                for page in (2, 9):
                    for row in range(4):
                        view[page * 8 + row].fill_(layer * 100 + page * 10 + row)
            manager = object.__new__(MooncakeKVManager)
            manager.attn_tp_size, manager.pp_size = src_tp, 1
            manager.is_mla_backend, manager.is_hybrid_mla_backend = False, True
            manager.enable_custom_mem_pool = False
            manager.max_transfer_batch_indices = 0
            manager.kv_args = SimpleNamespace(
                kv_data_ptrs=src_kv.get_contiguous_buf_infos()[0],
                kv_item_lens=src_kv.get_contiguous_buf_infos()[2],
                kv_layer_ids=[],
            )

            def copy(_session, blocks):
                for src, dst, length in blocks:
                    ctypes.memmove(dst, src, length)
                return 0

            manager._transfer_data = copy
            with patch(
                "sglang.srt.disaggregation.mooncake.conn.get_memory",
                return_value=SimpleNamespace(enable_unified_memory=True),
            ):
                self.assertEqual(
                    manager.send_kvcache(
                        "cpu",
                        np.array([2, 9], dtype=np.int32),
                        dst_kv.get_contiguous_buf_infos()[0],
                        np.array([8, 3], dtype=np.int32),
                        executor=None,
                        dst_kv_item_len=dst_kv.get_contiguous_buf_infos()[2][0],
                        dst_attn_tp_size=dst_tp,
                    ),
                    0,
                )
            for src_page, dst_page in ((2, 8), (9, 3)):
                for layer in range(2):
                    self.assertTrue(
                        torch.equal(
                            src_kv.kv_buffer[layer][src_page * 8 : src_page * 8 + 4],
                            dst_kv.kv_buffer[layer][dst_page * 8 : dst_page * 8 + 4],
                        )
                    )

    def test_unequal_tp_requires_implemented_peer_layouts(self):
        _, state, kv = make_pool(2, 1, None)
        _, dst_state, dst_kv = make_pool(4, 1, None)
        manager = object.__new__(MooncakeKVManager)
        manager.attn_tp_size, manager.pp_size, manager.dcp_size = 2, 1, 1
        manager.is_hybrid_mla_backend = True
        manager.kv_args = SimpleNamespace(
            unified_kv_layout=kv.get_transfer_layout([0, 2]),
            unified_state_layout=state.get_transfer_layout(),
        )
        peer = SimpleNamespace(
            dst_attn_tp_size=4,
            dst_dcp_size=1,
            dst_unified_kv_layout=dst_kv.get_transfer_layout([0, 2]),
            dst_unified_state_layout=dst_state.get_transfer_layout(),
        )
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.get_memory",
            return_value=SimpleNamespace(enable_unified_memory=True),
        ):
            manager._validate_unified_peer_layout(peer)
            for field, value in (
                ("dst_dcp_size", 2),
                ("dst_unified_state_layout", None),
            ):
                bad = SimpleNamespace(**{**vars(peer), field: value})
                with self.assertRaisesRegex(RuntimeError, "requires hybrid MLA"):
                    manager._validate_unified_peer_layout(bad)
            manager.is_hybrid_mla_backend = False
            with self.assertRaisesRegex(RuntimeError, "requires hybrid MLA"):
                manager._validate_unified_peer_layout(peer)

    def test_reject_bad_shapes_and_groups_before_writing(self):
        src = make_pool(2, 1, None)[1].get_transfer_layout()
        dst = make_pool(4, 1, None)[1].get_transfer_layout()
        with self.assertRaisesRegex(ValueError, "shapes"):
            validate_unified_state_layouts(src, dst, 2, 2)
        bad = replace(
            src,
            tensors=(replace(src.tensors[0], shard_groups=(3, 13)), *src.tensors[1:]),
        )
        with self.assertRaises(ValueError):
            validate_unified_state_layouts(bad, dst, 2, 4)
        with self.assertRaisesRegex(ValueError, "different TP groups"):
            build_unified_state_transfer_blocks(
                src=src,
                dst=dst,
                src_base=0,
                dst_base=0,
                src_slot=2,
                dst_slot=5,
                src_tp=2,
                dst_tp=4,
                src_rank=0,
                dst_rank=3,
            )


if __name__ == "__main__":
    unittest.main()
