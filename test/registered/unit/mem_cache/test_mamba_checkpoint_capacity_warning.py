"""Final checkpoint diagnostics must account for active slots and cache policy."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_cache_configurator import (
    KVCacheConfigurator,
    _warn_if_mamba_checkpoint_capacity_is_insufficient,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.mem_cache.kv_cache_configurator"


class TestMambaCheckpointCapacityWarning(CustomTestCase):
    @contextmanager
    def _config(self, *, host_only=False, **overrides):
        fields = dict(
            max_running_requests=33,
            chunked_prefill_size=2048,
            max_mamba_cache_size=165,
            disable_radix_cache=False,
            disable_overlap_schedule=False,
            mamba_radix_cache_strategy="extra_buffer",
            hybrid_cache_checkpoints_level=3,
            enable_int8_mamba_checkpoint=False,
            enable_hierarchical_cache=host_only,
            enable_unified_memory=False,
        )
        fields.update(overrides)
        with (
            envs.SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT.override(host_only),
            get_context().override_server_args(**fields),
        ):
            yield

    @staticmethod
    def _configurator():
        # Only per-runner inputs are doubled; configuration is published above.
        configurator = object.__new__(KVCacheConfigurator)
        configurator.model_config = SimpleNamespace(context_len=8192)
        configurator.attn_dp_size = 1
        configurator.mambaish_config = object()
        configurator.is_draft_worker = False
        configurator.is_hybrid_swa = False
        configurator._unified_sized = None
        return configurator

    @staticmethod
    def _warn(**overrides):
        inputs = dict(
            token_capacity=433_919,
            chunked_prefill_size=2048,
            max_mamba_cache_size=165,
            max_running_requests=33,
            tracking_slots_per_request=2,
        )
        inputs.update(overrides)
        _warn_if_mamba_checkpoint_capacity_is_insufficient(**inputs)

    def test_active_requests_leave_66_cached_slots(self):
        """The reported workload reserves 99 of 165 slots at full concurrency."""
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            self._warn()
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].args, (433_919, 212, 2048, 33, 66, 165, 99, 3))
        self.assertIn("filled with chunked prefixes", logs.output[0])
        self.assertIn("Short requests and branching", logs.output[0])

    def test_running_request_reservation_changes_warning_threshold(self):
        """100 states fit in the idle pool, but not its 66 cached slots at load."""
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            self._warn(token_capacity=204_800, max_running_requests=0)
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            self._warn(token_capacity=204_800)
        self.assertEqual(logs.records[0].args[1], 100)
        self.assertEqual(logs.records[0].args[4], 66)

    def test_larger_chunks_fit_in_loaded_cache(self):
        # ceil(433919 / 8192) = 53, below the 66 cached slots at full load.
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            self._warn(chunked_prefill_size=8192)

    def test_rounding_and_exact_capacity_boundary(self):
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            self._warn(
                token_capacity=8192,
                chunked_prefill_size=8192,
                max_mamba_cache_size=2,
                max_running_requests=1,
                tracking_slots_per_request=0,
            )
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            self._warn(
                token_capacity=8193,
                chunked_prefill_size=8192,
                max_mamba_cache_size=2,
                max_running_requests=1,
                tracking_slots_per_request=0,
            )
        self.assertEqual(logs.records[0].args[1], 2)
        self.assertEqual(logs.records[0].args[4], 1)

    def test_unconfigured_limits_do_not_warn(self):
        for overrides in (
            {"token_capacity": 0},
            {"chunked_prefill_size": None},
            {"chunked_prefill_size": 0},
            {"chunked_prefill_size": -1},
            {"max_mamba_cache_size": None},
            {"max_mamba_cache_size": 0},
        ):
            with self.subTest(overrides=overrides), self.assertNoLogs(_LOGGER):
                self._warn(**overrides)

    def test_tracking_slots_follow_each_cache_strategy(self):
        for strategy, overlap_disabled, cached_slots, slots_per_request in (
            ("extra_buffer", False, 66, 3),
            ("extra_buffer", True, 99, 2),
            ("extra_buffer_lazy", False, 99, 2),
            ("no_buffer", True, 132, 1),
        ):
            with (
                self.subTest(strategy=strategy, overlap_disabled=overlap_disabled),
                self._config(
                    mamba_radix_cache_strategy=strategy,
                    disable_overlap_schedule=overlap_disabled,
                ),
                self.assertLogs(_LOGGER, level="WARNING") as logs,
            ):
                self._configurator().warn_if_mamba_checkpoint_capacity_is_insufficient(
                    token_capacity=433_919, max_running_requests=33
                )
            self.assertEqual(logs.records[0].args[4], cached_slots)
            self.assertEqual(logs.records[0].args[7], slots_per_request)

    def test_inapplicable_cache_modes_do_not_warn(self):
        for overrides in (
            {"disable_radix_cache": True},
            {"enable_unified_memory": True},
            {"hybrid_cache_checkpoints_level": 2},
            {"mamba_max_states_per_path": 1},
            {"enable_int8_mamba_checkpoint": True},
            {"host_only": True},
        ):
            with (
                self.subTest(overrides=overrides),
                self._config(**overrides),
                self.assertNoLogs(_LOGGER, level="WARNING"),
            ):
                self._configurator().warn_if_mamba_checkpoint_capacity_is_insufficient(
                    token_capacity=433_919, max_running_requests=33
                )

    def test_sliding_window_models_can_checkpoint_every_chunk_at_level_2(self):
        with (
            self._config(
                hybrid_cache_checkpoints_level=2, mamba_max_states_per_path=-1
            ),
            self.assertLogs(_LOGGER, level="WARNING"),
        ):
            configurator = self._configurator()
            configurator.is_hybrid_swa = True
            configurator.warn_if_mamba_checkpoint_capacity_is_insufficient(
                token_capacity=433_919, max_running_requests=33
            )

    def test_nonhybrid_and_draft_workers_do_not_warn(self):
        for field, value in (("mambaish_config", None), ("is_draft_worker", True)):
            with self.subTest(field=field), self._config(), self.assertNoLogs(_LOGGER):
                configurator = self._configurator()
                setattr(configurator, field, value)
                configurator.warn_if_mamba_checkpoint_capacity_is_insufficient(
                    token_capacity=433_919, max_running_requests=33
                )

    def test_pre_capture_capacity_does_not_produce_a_warning(self):
        """A smaller final pool must not inherit warnings from sizing passes."""
        with self._config(), self.assertNoLogs(_LOGGER, level="WARNING"):
            configurator = self._configurator()
            self.assertEqual(configurator.resolve_max_num_reqs(433_919), 33)
            self.assertEqual(configurator.resolve_max_num_reqs(100_000), 33)
            configurator.warn_if_mamba_checkpoint_capacity_is_insufficient(
                token_capacity=100_000, max_running_requests=33
            )

    def test_final_capacity_is_reported_once(self):
        with self._config():
            configurator = self._configurator()
            with self.assertNoLogs(_LOGGER, level="WARNING"):
                configurator.resolve_max_num_reqs(1_000_000)
                configurator.resolve_max_num_reqs(433_919)
            with self.assertLogs(_LOGGER, level="WARNING") as logs:
                configurator.warn_if_mamba_checkpoint_capacity_is_insufficient(
                    token_capacity=433_919, max_running_requests=33
                )
            self.assertEqual(len(logs.records), 1)
            self.assertEqual(logs.records[0].args[0], 433_919)


if __name__ == "__main__":
    unittest.main()
