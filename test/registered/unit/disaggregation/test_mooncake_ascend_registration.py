"""Allocation spans use referenced bytes and retain transfer view addresses."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.mooncake import npu_registration as reg
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAscendRegistration(CustomTestCase):
    def test_device_attributes_resolve_allocation_base(self):
        def attributes(pointer, out):
            out._obj.location.type = 1
            return 0

        def address_range(pointer, base, size):
            base._obj.value = 100
            size._obj.value = 200
            return 0

        native = SimpleNamespace(
            aclrtPointerGetAttributes=attributes,
            aclrtMemGetAddressRange=address_range,
        )
        with patch.object(reg, "api", return_value=native):
            self.assertEqual(reg.registration_region(120, 40), (100, 200))
            with self.assertRaisesRegex(RuntimeError, "exceeds allocation"):
                reg.registration_region(280, 40)

    def test_host_attributes_keep_the_original_extent(self):
        native = SimpleNamespace(aclrtPointerGetAttributes=Mock(return_value=0))
        with patch.object(reg, "api", return_value=native):
            self.assertEqual(reg.registration_region(120, 40), (120, 40))

    def test_shared_allocation_stops_at_last_used_byte(self):
        with patch.object(reg, "registration_region", return_value=(100, 1000)):
            self.assertEqual(
                reg.registration_regions([(100, 40), (180, 20), (120, 50)]),
                [(100, 100)],
            )

    def test_independent_host_regions_remain_separate(self):
        with patch.object(
            reg, "registration_region", side_effect=lambda ptr, length: (ptr, length)
        ):
            self.assertEqual(
                reg.registration_regions([(100, 40), (180, 20)]), [(100, 40), (180, 20)]
            )

    def test_out_of_allocation_span_is_rejected(self):
        with patch.object(reg, "registration_region", return_value=(100, 50)):
            with self.assertRaises(RuntimeError):
                reg.registration_regions([(140, 20)])

    def test_empty_input(self):
        self.assertEqual(reg.registration_regions([]), [])

    def test_native_api_failures_propagate(self):
        native = SimpleNamespace(aclrtPointerGetAttributes=Mock(return_value=7))
        with patch.object(reg, "api", return_value=native):
            with self.assertRaisesRegex(RuntimeError, "Attributes failed: 7"):
                reg.registration_region(100, 20)
        with self.assertRaises(ValueError):
            reg.registration_region(0, 20)

    def test_manager_deduplicates_registration_without_rewriting_views(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100, 180],
            kv_data_lens=[40, 20],
            aux_data_ptrs=[],
            aux_data_lens=[],
            state_data_ptrs=[],
            state_data_lens=[],
        )
        with (
            patch.object(
                envs.ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE, "get", return_value=True
            ),
            patch.object(
                envs.SGLANG_MOONCAKE_ASCEND_REGISTER_ALLOCATION,
                "get",
                return_value=True,
            ),
            patch.object(reg, "registration_region", return_value=(100, 1000)),
        ):
            self.assertEqual(manager._registerable_regions(), [(100, 100)])
        self.assertEqual(manager.kv_args.kv_data_ptrs, [100, 180])
        self.assertEqual(manager.kv_args.kv_data_lens, [40, 20])


if __name__ == "__main__":
    unittest.main()
