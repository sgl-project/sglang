"""Check PD byte addresses against actual page-major compute views."""

import unittest

import torch

from sglang.srt.mem_cache.layout.transfer import TransferLayout, TransferTensor
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMambaPool,
    UnifiedMLATokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_pools(page_size=4, conv_axis=1, groups=None):
    full_spec = MLASubPoolSpec(
        name="full",
        layer_num=3,
        grow_direction="down",
        kv_lora_rank=6,
        qk_rope_head_dim=2,
        store_dtype=torch.bfloat16,
    )
    state_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=2,
        grow_direction="up",
        conv_state_shapes=((3, 16), (2, 8)) if conv_axis == 1 else ((16, 3),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(4, 2, 2),
        temporal_dtype=torch.float32,
        conv_slice_axis=conv_axis,
        conv_shard_groups=groups,
    )
    raw = UnifiedKVPool(
        total_bytes=32768,
        sub_pool_specs=[full_spec, state_spec],
        device="cpu",
        enable_memory_saver=False,
        page_size=page_size,
    )
    full = UnifiedMLATokenToKVPool(
        unified_buffer=raw,
        sub_pool_name="full",
        kv_cache_dtype=torch.bfloat16,
        page_size=page_size,
    )
    state = UnifiedMambaPool(
        unified_buffer=raw,
        sub_pool_name="mamba",
        spec_state_size=8,
        mamba_layer_ids=[1, 4],
    )
    return raw, full, state


class TestUnifiedTransferLayout(unittest.TestCase):
    def test_mla_addresses_match_kernel_views_on_fragmented_pages(self):
        for ps in (1, 4, 16):
            raw, full, _ = make_pools(page_size=ps)
            layout = full.get_transfer_layout([0, 2, 5])
            for page in (7, 2, 9):
                for layer, view in enumerate(full.kv_buffer):
                    for row in range(ps):
                        kernel_id = page * ps * full.layer_num + row
                        expected = view[kernel_id].data_ptr()
                        addr = layout.address(raw._raw.data_ptr(), page, layer, row)
                        self.assertEqual(addr, expected)
                        view[kernel_id].fill_(100 * layer + 10 * page + row)
                        start = addr - raw._raw.data_ptr()
                        payload = raw._raw[
                            start : start + layout.tensors[layer].row_bytes
                        ]
                        self.assertTrue(
                            torch.equal(
                                payload.view(view.dtype), view[kernel_id].flatten()
                            )
                        )
            # Transfer pieces are descriptions, never extra registrations.
            self.assertEqual(
                full.get_contiguous_buf_infos(),
                ([raw._raw.data_ptr()], [raw._raw.numel()], [layout.block_bytes]),
            )
            self.assertTrue(all(t.slice_axis is None for t in layout.tensors))

    def test_state_addresses_match_strided_views(self):
        for axis, groups in ((1, None), (0, (8, 8, 16))):
            raw, _, state = make_pools(conv_axis=axis, groups=groups)
            layout = state.get_transfer_layout()
            views = list(state.mamba_cache.conv) + [state.mamba_cache.temporal]
            for slot in (5, 1, 8):
                for tensor_index, view in enumerate(views):
                    for layer in range(2):
                        index = tensor_index * 2 + layer
                        entry = layout.tensors[index]
                        addr = layout.address(raw._raw.data_ptr(), slot, index)
                        self.assertEqual(addr, view[layer, slot].data_ptr())
                        self.assertEqual(
                            entry.row_bytes,
                            view[layer, slot].numel() * view.element_size(),
                        )
                        self.assertEqual(
                            layout.block_bytes, view.stride(1) * view.element_size()
                        )
                        self.assertGreater(layout.block_bytes, entry.row_bytes)
                        view[layer, slot].fill_(100 * tensor_index + 10 * layer + slot)
                        start = addr - raw._raw.data_ptr()
                        payload = raw._raw[start : start + entry.row_bytes]
                        self.assertTrue(
                            torch.equal(
                                payload.view(view.dtype), view[layer, slot].flatten()
                            )
                        )
            conv = layout.tensors[0]
            self.assertEqual(conv.slice_dim, state.mamba_cache.conv[0].shape[2 + axis])
            self.assertEqual(conv.outer_count, 3 if axis == 1 else 1)
            self.assertEqual(conv.shard_groups, groups)
            self.assertEqual(
                state.get_contiguous_buf_infos(),
                ([raw._raw.data_ptr()], [raw._raw.numel()], [layout.block_bytes]),
            )
            self.assertEqual(state.get_state_dim_per_tensor(), [])

    def test_rejects_overlapping_or_out_of_block_pieces(self):
        tensor = TransferTensor("kv", 0, 0, (8,), 2)
        with self.assertRaisesRegex(ValueError, "overlap"):
            TransferLayout(64, 2, (tensor, tensor))
        with self.assertRaisesRegex(ValueError, "past its block"):
            TransferLayout(16, 2, (tensor,))
        with self.assertRaisesRegex(ValueError, "layer IDs"):
            make_pools()[1].get_transfer_layout([0])


if __name__ == "__main__":
    unittest.main()
