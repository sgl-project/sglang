"""CPU tests for the DSV4 prefill graph metadata contract.

The captured segment reads these tensors by address, so replay must refresh
their contents in place.  No NPU is required: the copy logic is pure torch.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.attention.dsv4_prefill_graph import (
    Dsv4PrefillGraphMetadata,
    compress_capacity,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_DUMMY_STATE = 999


def _kernel_metadata(*, bs, actual_seq_lengths_q_pa, actual_seq_lengths_kv):
    return {
        "c1a_metadata": torch.zeros(1024, dtype=torch.int32),
        "c4a_metadata": torch.ones(1024, dtype=torch.int32),
    }


def _state_table(
    *, ratio, req_pool_indices, start_pos, cu_seqlens, seqused, max_input_capacity
):
    history = (2 if ratio == 4 else 1) * ratio
    table = torch.full(
        (1, history + max_input_capacity), _DUMMY_STATE, dtype=torch.int32
    )
    table[:, :history] = ratio
    return table


def _backend():
    pool = SimpleNamespace(dummy_state_loc=_DUMMY_STATE)
    return SimpleNamespace(
        device="cpu",
        page_size=128,
        max_context_len=4096,
        _dsv4_has_c4=True,
        _dsv4_has_c128=True,
        _dsv4_unique_compress_ratios=[4, 128],
        _dsv4_state_pools_by_ratio={4: pool, 128: pool},
        req_to_token_pool=SimpleNamespace(c128_page_size=32),
        _kernel_metadata_from_parts=_kernel_metadata,
        _build_dsv4_prefill_graph_state_table=_state_table,
    )


def _seed(num_tokens):
    return SimpleNamespace(
        actual_seq_lengths_q=torch.tensor([num_tokens], dtype=torch.int32),
        actual_seq_lengths_q_pa=torch.tensor([0, num_tokens], dtype=torch.int32),
        actual_seq_lengths_kv=torch.tensor([num_tokens], dtype=torch.int32),
        seqused=torch.tensor([num_tokens], dtype=torch.int32),
        start_pos=torch.tensor([0], dtype=torch.int32),
        positions_cmp_padding_c4=torch.arange(min(num_tokens, num_tokens // 4 + 1)),
        positions_cmp_padding_c128=torch.arange(min(num_tokens, num_tokens // 128 + 1)),
        c4_loc=None,
        c128_loc=None,
        swa_page_table=torch.zeros(1, 2, dtype=torch.int64),
        c4_page_table=torch.zeros(1, 3, dtype=torch.int32),
        c128_page_table=torch.zeros(1, 1, dtype=torch.int32),
        kernel_metadata=_kernel_metadata(
            bs=1, actual_seq_lengths_q_pa=None, actual_seq_lengths_kv=None
        ),
    )


def _live_batch(batch_size, num_tokens, *, idle=False):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: idle),
        global_num_token_non_padded_cpu=0 if idle else num_tokens,
        batch_size=batch_size,
        req_pool_indices=torch.arange(batch_size, dtype=torch.int64),
    )


class TestDsv4PrefillGraphMetadata(CustomTestCase):
    def test_capture_allocates_fixed_isolated_buffers(self):
        seed = _seed(128)
        state = Dsv4PrefillGraphMetadata.allocate(
            backend=_backend(), seed=seed, num_tokens=128
        )

        self.assertIs(state.metadata, seed)
        self.assertIs(seed.dsv4_prefill_graph_state, state)

        # Capture must not write through to a real request slot.
        self.assertEqual(seed.c4_loc.shape, (compress_capacity(128, 4, 1),))
        self.assertEqual(seed.c4_loc.abs().sum().item(), 0)
        self.assertTrue((state.state_tables[4] == _DUMMY_STATE).all().item())

        # Page-table widths follow the eager builder, not the block count.
        self.assertEqual(tuple(seed.swa_page_table.shape), (1, 32))
        self.assertEqual(tuple(seed.c4_page_table.shape), (1, 32))
        self.assertEqual(tuple(seed.c128_page_table.shape), (1, 1))

    def test_replay_refreshes_in_place(self):
        seed = _seed(128)
        state = Dsv4PrefillGraphMetadata.allocate(
            backend=_backend(), seed=seed, num_tokens=128
        )
        captured = {
            name: id(getattr(seed, name))
            for name in (
                "actual_seq_lengths_q_pa",
                "actual_seq_lengths_kv",
                "c4_loc",
                "swa_page_table",
                "kernel_metadata",
            )
        }

        state.refresh(
            backend=_backend(),
            live=_seed(64),
            forward_batch=_live_batch(1, 64),
        )

        for name, object_id in captured.items():
            self.assertEqual(id(getattr(seed, name)), object_id, name)
        self.assertEqual(seed.actual_seq_lengths_q_pa.tolist(), [0, 64])
        self.assertEqual(seed.actual_seq_lengths_kv.tolist(), [64])
        self.assertEqual(seed.seqused.tolist(), [64])
        # Live columns carry the rebuilt table; the tail stays on the dummy.
        self.assertTrue((state.state_tables[4][0, :8] == 4).all().item())
        self.assertTrue((state.state_tables[4][0, 8:] == _DUMMY_STATE).all().item())

    def test_idle_replay_stays_legal(self):
        seed = _seed(128)
        state = Dsv4PrefillGraphMetadata.allocate(
            backend=_backend(), seed=seed, num_tokens=128
        )

        state.refresh(
            backend=_backend(),
            live=SimpleNamespace(),
            forward_batch=_live_batch(1, 0, idle=True),
        )

        self.assertEqual(seed.actual_seq_lengths_q_pa.tolist(), [0, 1])
        self.assertEqual(seed.actual_seq_lengths_kv.tolist(), [1])
        self.assertEqual(seed.seqused.tolist(), [0])
        self.assertTrue((state.state_tables[4] == _DUMMY_STATE).all().item())

    def test_topk_is_staged_per_layer(self):
        seed = _seed(128)
        state = Dsv4PrefillGraphMetadata.allocate(
            backend=_backend(), seed=seed, num_tokens=128
        )
        first = torch.zeros(128, 64, dtype=torch.int32)
        second = torch.ones(128, 64, dtype=torch.int32)

        self.assertIs(state.stage_topk(3, first), first)
        self.assertIs(state.stage_topk(5, second), second)
        self.assertIs(state.topk_by_layer[3], first)
        self.assertIs(state.topk_by_layer[5], second)
        # The caller still owns the shared attribute; the state only tracks
        # the per-layer buffers the break reads.
        self.assertIsNone(seed.c4_topk_indices)


if __name__ == "__main__":
    unittest.main()
