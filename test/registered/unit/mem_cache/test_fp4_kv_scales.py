"""FP4 KV block scales must travel with the KV they describe: PD transfer
registration and the cpu_tensor retraction backup."""

import unittest

import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.mem_cache.kv_cache_builder import _has_per_token_kv_scales
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SLOTS, PAGE, LAYERS, ROW, SCALE_ROW = 64, 4, 2, 8, 2


def _pool(scales: bool = True, chunk_size: int = 3) -> MHATokenToKVPool:
    """A plain MHA pool with FP4-style per-token scale rows; only the buffers
    get/load_cpu_copy and the PD registration read are set up."""
    pool = object.__new__(MHATokenToKVPool)
    pool.layer_num = LAYERS
    pool.page_size = PAGE
    pool.use_hnd = False
    pool.cpu_offloading_chunk_size = chunk_size
    pool.k_buffer = [torch.zeros(SLOTS, ROW, dtype=torch.uint8) for _ in range(LAYERS)]
    pool.v_buffer = [torch.zeros(SLOTS, ROW, dtype=torch.uint8) for _ in range(LAYERS)]
    pool.k_scale_buffer = pool.v_scale_buffer = None
    if scales:
        pool.k_scale_buffer = [
            torch.zeros(SLOTS, SCALE_ROW, dtype=torch.uint8) for _ in range(LAYERS)
        ]
        pool.v_scale_buffer = [
            torch.zeros(SLOTS, SCALE_ROW, dtype=torch.uint8) for _ in range(LAYERS)
        ]
    return pool


def _fill(pool, slots):
    """A different value for every (buffer kind, layer, token)."""
    tok = torch.arange(len(slots), dtype=torch.uint8)
    for layer in range(LAYERS):
        pool.k_buffer[layer][slots] = (tok + 10 * layer + 1)[:, None]
        pool.v_buffer[layer][slots] = (tok + 10 * layer + 101)[:, None]
        if pool.k_scale_buffer is not None:
            pool.k_scale_buffer[layer][slots] = (tok + 10 * layer + 51)[:, None]
            pool.v_scale_buffer[layer][slots] = (tok + 10 * layer + 151)[:, None]


class TestFP4KVScales(CustomTestCase):
    def test_pd_registers_block_scales_in_k_then_v_layer_order(self):
        pool = _pool()
        kv_args = KVArgs()
        setup_state_kv_args(kv_args, pool)

        self.assertEqual(kv_args.state_types.count(StateType.BLOCK_SCALE), 1)
        i = kv_args.state_types.index(StateType.BLOCK_SCALE)
        tensors = pool.k_scale_buffer + pool.v_scale_buffer
        self.assertEqual(kv_args.state_data_ptrs[i], [t.data_ptr() for t in tensors])
        self.assertEqual(kv_args.state_data_lens[i], [t.nbytes for t in tensors])
        self.assertEqual(kv_args.state_item_lens[i], [SCALE_ROW * PAGE] * len(tensors))

    def test_unquantized_pool_registers_no_block_scales(self):
        kv_args = KVArgs()
        setup_state_kv_args(kv_args, _pool(scales=False))
        self.assertNotIn(StateType.BLOCK_SCALE, kv_args.state_types)

    def test_retraction_restores_scales_across_chunks(self):
        # 7 tokens with chunk size 3: two full chunks and a 1-token tail.
        pool = _pool(chunk_size=3)
        old = torch.tensor([4, 5, 6, 7, 20, 21, 22])
        new = torch.tensor([40, 41, 42, 43, 12, 13, 14])
        _fill(pool, old)
        expected = _pool(chunk_size=3)
        _fill(expected, new)
        for buf in pool.k_scale_buffer + pool.v_scale_buffer:
            buf[new] = 255  # left by the previous owner of the new slots

        pool.load_cpu_copy(pool.get_cpu_copy(old), new)

        for name in ("k_buffer", "v_buffer", "k_scale_buffer", "v_scale_buffer"):
            for layer in range(LAYERS):
                got = getattr(pool, name)[layer][new]
                want = getattr(expected, name)[layer][new]
                self.assertTrue(torch.equal(got, want), f"{name}[{layer}]")

    def test_unquantized_round_trip_unchanged(self):
        pool = _pool(scales=False)
        old, new = torch.tensor([1, 2, 3, 4]), torch.tensor([30, 31, 32, 33])
        _fill(pool, old)
        backup = pool.get_cpu_copy(old)
        self.assertTrue(all(len(entry) == 2 for layer in backup for entry in layer))
        pool.load_cpu_copy(backup, new)
        for layer in range(LAYERS):
            self.assertTrue(
                torch.equal(pool.k_buffer[layer][new], pool.k_buffer[layer][old])
            )

    def test_default_retraction_backup_avoids_host_pool_for_fp4_scales(self):
        # The MHA host pool rejects per-token scales, so an FP4 pool must
        # resolve to the cpu_tensor backup.
        self.assertTrue(_has_per_token_kv_scales(_pool()))
        self.assertFalse(_has_per_token_kv_scales(_pool(scales=False)))


if __name__ == "__main__":
    unittest.main()
