"""Fail startup before exposing KV or staging buffers with failed registration."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def manager(status=0):
    result = MooncakeKVManager.__new__(MooncakeKVManager)
    result.engine = MagicMock()
    result.engine.batch_register.return_value = status
    result.kv_args = SimpleNamespace(
        kv_data_ptrs=[1000, 2000, 1000, 9000],
        kv_data_lens=[128, 128, 128, 0],
        aux_data_ptrs=[3000],
        aux_data_lens=[64],
        state_data_ptrs=[[4000], [5000]],
        state_data_lens=[[32], [16]],
    )
    return result


class TestMooncakeRegistrationFailure(unittest.TestCase):
    def test_success_registers_main_aux_and_state_once_without_elided_entries(self):
        value = manager()
        value.register_buffer_to_engine()
        value.engine.batch_register.assert_called_once_with(
            [1000, 2000, 3000, 4000, 5000], [128, 128, 64, 32, 16]
        )

    def test_failed_registration_does_not_advertise_success_or_retry_individually(self):
        for status in (-202, -1, 1):
            with self.subTest(status=status):
                value = manager(status)
                with self.assertRaisesRegex(
                    RuntimeError, f"registration failed.*{status}"
                ):
                    value.register_buffer_to_engine()
                self.assertEqual(value.engine.batch_register.call_count, 1)
                value.engine.register.assert_not_called()
                value.engine.batch_deregister.assert_not_called()

    def test_provider_exception_propagates_without_an_implicit_retry(self):
        value = manager()
        error = RuntimeError("provider failure")
        value.engine.batch_register.side_effect = error
        with self.assertRaises(RuntimeError) as caught:
            value.register_buffer_to_engine()
        self.assertIs(caught.exception, error)
        self.assertEqual(value.engine.batch_register.call_count, 1)

    def test_empty_regions_need_no_registration(self):
        value = manager()
        value.kv_args = SimpleNamespace(
            kv_data_ptrs=[],
            kv_data_lens=[],
            aux_data_ptrs=[],
            aux_data_lens=[],
            state_data_ptrs=[],
            state_data_lens=[],
        )
        value.register_buffer_to_engine()
        value.engine.batch_register.assert_not_called()

    def test_both_roles_fail_before_staging_configuration_or_worker_start(self):
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                value = manager(-202)
                with (
                    patch.object(CommonKVManager, "__init__", return_value=None),
                    patch.object(MooncakeKVManager, "init_engine"),
                    patch(
                        "sglang.srt.disaggregation.mooncake.conn.check_mooncake_custom_mem_pool_enabled",
                        return_value=(False, None),
                    ),
                    patch(
                        "sglang.srt.disaggregation.mooncake.conn._validate_efa_allocator_compatibility"
                    ),
                    patch(
                        "sglang.srt.disaggregation.mooncake.conn.envs.SGLANG_DISAGG_STAGING_BUFFER.get",
                        side_effect=AssertionError(
                            "failed registration must stop startup"
                        ),
                    ) as staging,
                    patch.object(MooncakeKVManager, "start_prefill_thread") as prefill,
                    patch.object(MooncakeKVManager, "start_decode_thread") as decode,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "registration failed.*-202"
                    ):
                        MooncakeKVManager.__init__(
                            value, value.kv_args, mode, SimpleNamespace()
                        )
                    staging.assert_not_called()
                    prefill.assert_not_called()
                    decode.assert_not_called()

    def test_staging_registration_success_uses_exact_buffer(self):
        value = manager()
        value._register_staging_memory(6000, 256)
        value.engine.batch_register.assert_called_once_with([6000], [256])

    def test_failed_staging_registration_propagates_to_both_allocating_helpers(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_allocator,
            init_staging_buffers,
        )

        value = manager(-202)
        value.kv_args.page_size = 64
        value.kv_args.kv_item_lens = [128, 128]
        value.kv_args.gpu_id = 0
        buffer = MagicMock()
        buffer.get_ptr.return_value = 6000
        buffer.get_size.return_value = 256
        allocator = MagicMock()
        allocator.get_base_ptr.return_value = 7000
        allocator.get_total_size.return_value = 512
        with (
            patch(
                "sglang.srt.disaggregation.common.staging_handler._get_custom_mem_pool",
                return_value=(None, None),
            ),
            patch(
                "sglang.srt.disaggregation.common.staging_buffer.StagingBuffer",
                return_value=buffer,
            ) as make_buffer,
            patch(
                "sglang.srt.disaggregation.common.staging_buffer.StagingAllocator",
                return_value=allocator,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "staging.*registration failed.*-202"
            ):
                init_staging_buffers(
                    value._register_staging_memory,
                    value.kv_args,
                    2,
                    64,
                    device_type="cuda",
                )
            self.assertEqual(make_buffer.call_count, 1)
            with self.assertRaisesRegex(
                RuntimeError, "staging.*registration failed.*-202"
            ):
                init_staging_allocator(
                    value._register_staging_memory, value.kv_args, device_type="cuda"
                )


if __name__ == "__main__":
    unittest.main()
