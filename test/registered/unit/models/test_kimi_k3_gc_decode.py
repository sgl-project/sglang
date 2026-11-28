"""Group-uniform backend selection for Kimi K3 GC decode."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.moe.kimi_k3_gc import KimiK3GCDecode
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGCDecodeSelection(unittest.TestCase):
    def test_ep_ranks_choose_the_same_collective(self):
        adapter = KimiK3GCDecode.__new__(KimiK3GCDecode)
        adapter.attn_dp_size = 2
        adapter.decode_only = False
        for local in (ForwardMode.IDLE, ForwardMode.DECODE):
            for global_mode, expected in (
                (ForwardMode.DECODE, True),
                (ForwardMode.EXTEND, False),
                (ForwardMode.TARGET_VERIFY, False),
                (None, False),
            ):
                batch = SimpleNamespace(
                    forward_mode=local, global_forward_mode=global_mode
                )
                with self.subTest(local=local, global_mode=global_mode):
                    self.assertEqual(adapter.use_for(batch), expected)

    def test_decode_only_idle_peers_use_gc_and_unsupported_modes_fail(self):
        adapter = KimiK3GCDecode.__new__(KimiK3GCDecode)
        adapter.attn_dp_size = 2
        adapter.decode_only = True
        for local in (ForwardMode.IDLE, ForwardMode.DECODE):
            for global_mode in (None, ForwardMode.IDLE, ForwardMode.DECODE):
                with self.subTest(local=local, global_mode=global_mode):
                    self.assertTrue(adapter.use_for(SimpleNamespace(
                        forward_mode=local, global_forward_mode=global_mode
                    )))
        for mode in (ForwardMode.EXTEND, ForwardMode.TARGET_VERIFY, ForwardMode.MIXED):
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "forward mode"):
                adapter.use_for(SimpleNamespace(forward_mode=ForwardMode.IDLE, global_forward_mode=mode))

    def test_tp_without_a_global_mode_uses_the_local_mode(self):
        adapter = KimiK3GCDecode.__new__(KimiK3GCDecode)
        adapter.attn_dp_size = 1
        adapter.decode_only = False
        for local in (
            ForwardMode.DECODE,
            ForwardMode.EXTEND,
            ForwardMode.TARGET_VERIFY,
        ):
            batch = SimpleNamespace(forward_mode=local, global_forward_mode=None)
            self.assertEqual(adapter.use_for(batch), local == ForwardMode.DECODE)


if __name__ == "__main__":
    unittest.main()
