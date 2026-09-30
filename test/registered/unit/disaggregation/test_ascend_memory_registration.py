"""CPU regression tests for Ascend PD memory registration placeholders."""

import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAscendMemoryRegistration(unittest.TestCase):
    def make_manager(self, **overrides):
        args = dict(
            kv_data_ptrs=[0x1000, 0],
            kv_data_lens=[256, 0],
            aux_data_ptrs=[0x2000],
            aux_data_lens=[128],
            # Tail keys and scores retain a placeholder for skip-topk layers.
            state_data_ptrs=[[0x3000, 0], [0x4000, 0]],
            state_data_lens=[[512, 0], [512, 0]],
            state_item_lens=[[64, 0], [64, 0]],
        )
        args.update(overrides)
        manager = AscendKVManager.__new__(AscendKVManager)
        manager.kv_args = SimpleNamespace(**args)
        manager.engine = Mock()
        return manager

    def test_register_filters_placeholders_without_changing_wire_metadata(self):
        manager = self.make_manager(
            # Zero-length views may also have a nonzero pointer.
            aux_data_ptrs=[0x2000, 0x5000],
            aux_data_lens=[128, 0],
        )
        original = copy.deepcopy(vars(manager.kv_args))

        manager.register_buffer_to_engine()

        manager.engine.batch_register.assert_called_once_with(
            [0x1000, 0x2000, 0x3000, 0x4000], [256, 128, 512, 512]
        )
        self.assertEqual(vars(manager.kv_args), original)

    def test_deregister_uses_same_regions_and_clears_connections(self):
        manager = self.make_manager()
        manager.connection_pool = {"peer": object()}
        manager.connection_lock = threading.Lock()
        original = copy.deepcopy(vars(manager.kv_args))

        manager.register_buffer_to_engine()
        manager.deregister_buffer_to_engine()

        registered_ptrs = manager.engine.batch_register.call_args.args[0]
        manager.engine.batch_deregister.assert_called_once_with(registered_ptrs)
        self.assertNotIn(0, registered_ptrs)
        self.assertEqual(manager.connection_pool, {})
        self.assertEqual(vars(manager.kv_args), original)

    def test_all_empty_regions_do_not_call_engine(self):
        manager = self.make_manager(
            kv_data_ptrs=[0],
            kv_data_lens=[0],
            aux_data_ptrs=[0x2000],
            aux_data_lens=[0],
            state_data_ptrs=[[0], [0]],
            state_data_lens=[[0], [0]],
            state_item_lens=[[0], [0]],
        )

        manager.register_buffer_to_engine()
        manager.deregister_buffer_to_engine()

        manager.engine.batch_register.assert_not_called()
        manager.engine.batch_deregister.assert_not_called()

    def test_no_optional_components(self):
        manager = self.make_manager(
            kv_data_ptrs=[0x1000],
            kv_data_lens=[256],
            aux_data_ptrs=[],
            aux_data_lens=[],
            state_data_ptrs=None,
            state_data_lens=None,
            state_item_lens=None,
        )

        manager.register_buffer_to_engine()
        manager.deregister_buffer_to_engine()

        manager.engine.batch_register.assert_called_once_with([0x1000], [256])
        manager.engine.batch_deregister.assert_called_once_with([0x1000])

    def test_shared_regions_are_registered_once(self):
        manager = self.make_manager(
            state_data_ptrs=[[0x1000, 0], [0x3000, 0x3000]],
            state_data_lens=[[256, 0], [512, 512]],
        )

        manager.register_buffer_to_engine()
        manager.deregister_buffer_to_engine()

        manager.engine.batch_register.assert_called_once_with(
            [0x1000, 0x2000, 0x3000], [256, 128, 512]
        )
        manager.engine.batch_deregister.assert_called_once_with(
            [0x1000, 0x2000, 0x3000]
        )

    def test_invalid_nonempty_regions_fail_before_registration(self):
        for ptr, length in [(0, 512), (-1, 512), (0x3000, -1)]:
            with self.subTest(ptr=ptr, length=length):
                manager = self.make_manager(
                    state_data_ptrs=[[ptr]], state_data_lens=[[length]]
                )

                with self.assertRaisesRegex(ValueError, "Invalid Ascend PD"):
                    manager.register_buffer_to_engine()

                manager.engine.batch_register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
