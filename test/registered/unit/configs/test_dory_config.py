"""Unit tests for srt/configs/dory.py

The KV-slot layout is derived, not read off the checkpoint: attention layers inside the
recurrent block need one slot per loop while the input/output blocks need one. A regression
here silently under-allocates the pool or makes two layers share a slot.
"""

import unittest

from sglang.srt.configs.dory import DoryConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDoryConfig(CustomTestCase):
    def test_derived_layer_types(self):
        cfg = DoryConfig(num_hidden_layers=72)
        self.assertEqual(len(cfg.block_types), 72)
        self.assertEqual(cfg.block_types[0], "sliding_attention")
        self.assertEqual(cfg.block_types[1], "mlp")
        self.assertEqual(cfg.block_types[6], "full_attention")
        self.assertEqual(sum(t != "mlp" for t in cfg.block_types), 36)

    def test_slot_layout_per_loop(self):
        cfg = DoryConfig(
            n_input_layers=16,
            n_recurrent_layers=40,
            n_output_layers=16,
            n_recurrent_loops=2,
            recurrent_kv_cache_mode="per_loop",
        )
        # 16 static attention layers plus 20 recurrent ones per loop.
        self.assertEqual(cfg.num_kv_slots(), 56)
        attn_layers = [i for i in range(72) if cfg.block_types[i] != "mlp"]
        slots = {
            cfg.kv_slot_id(i, loop)
            for i in attn_layers
            for loop in range(cfg.kv_loops(i))
        }
        self.assertEqual(slots, set(range(56)))

    def test_recurrent_slots_differ_per_loop(self):
        cfg = DoryConfig(n_input_layers=16, n_recurrent_loops=2)
        rec_attn = [i for i in range(16, 56) if cfg.block_types[i] != "mlp"]
        self.assertEqual(len(rec_attn), 20)
        for layer in rec_attn:
            self.assertNotEqual(cfg.kv_slot_id(layer, 0), cfg.kv_slot_id(layer, 1))

    def test_slot_layout_last_loop(self):
        cfg = DoryConfig(
            n_input_layers=16,
            n_recurrent_loops=4,
            recurrent_kv_cache_mode="last_loop",
        )
        # One slot per attention layer regardless of loop count.
        self.assertEqual(cfg.num_kv_slots(), 36)
        rec_attn = [i for i in range(16, 56) if cfg.block_types[i] != "mlp"]
        self.assertEqual(len({cfg.kv_slot_id(i) for i in rec_attn}), 20)

    def test_partition_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            DoryConfig(n_input_layers=16, n_recurrent_layers=40, n_output_layers=8)


if __name__ == "__main__":
    unittest.main()
