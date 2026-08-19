"""Unit tests for idle_sleeper — no server, no model loading."""

import unittest
from unittest.mock import MagicMock, call, patch

import zmq

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import idle_sleeper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestIdleSleeper(CustomTestCase):
    def create_sockets(self, count: int = 2) -> list:
        return [MagicMock() for _ in range(count)]

    def test_init_registers_every_socket_for_pollin(self):
        sockets = self.create_sockets(3)
        with (
            patch.object(idle_sleeper.zmq, "Poller") as mock_poller_cls,
            patch.object(idle_sleeper, "real_time", return_value=100.0),
            envs.SGLANG_EMPTY_CACHE_INTERVAL.override(-1),
        ):
            sleeper = idle_sleeper.IdleSleeper(sockets)

        mock_poller_cls.return_value.register.assert_has_calls(
            [call(s, zmq.POLLIN) for s in sockets]
        )
        self.assertEqual(mock_poller_cls.return_value.register.call_count, 3)
        self.assertEqual(sleeper.last_empty_time, 100.0)

    def test_maybe_sleep_polls_fixed_timeout_and_skips_eviction_when_disabled(self):
        # SGLANG_EMPTY_CACHE_INTERVAL default is -1 (disabled) — the production
        # default must never trigger eviction, only the 1s poll wakeup.
        with (
            patch.object(idle_sleeper.zmq, "Poller") as mock_poller_cls,
            patch.object(idle_sleeper, "real_time", return_value=100.0),
            patch.object(idle_sleeper.current_platform, "empty_cache") as empty_cache,
            envs.SGLANG_EMPTY_CACHE_INTERVAL.override(-1),
        ):
            sleeper = idle_sleeper.IdleSleeper(self.create_sockets())
            sleeper.maybe_sleep()

        mock_poller_cls.return_value.poll.assert_called_once_with(1000)
        empty_cache.assert_not_called()

    def test_empty_cache_not_called_at_or_below_interval(self):
        # Guards the strict `>` in maybe_sleep: an elapsed time exactly equal
        # to the interval must NOT evict (only strictly-greater does).
        cases = {
            "elapsed_below_interval": [10.0, 13.0],
            "elapsed_equal_interval": [10.0, 15.0],
        }
        for label, real_time_values in cases.items():
            with self.subTest(label=label):
                with (
                    patch.object(idle_sleeper.zmq, "Poller"),
                    patch.object(
                        idle_sleeper, "real_time", side_effect=list(real_time_values)
                    ),
                    patch.object(
                        idle_sleeper.current_platform, "empty_cache"
                    ) as empty_cache,
                    envs.SGLANG_EMPTY_CACHE_INTERVAL.override(5),
                ):
                    sleeper = idle_sleeper.IdleSleeper(self.create_sockets())
                    sleeper.maybe_sleep()

                empty_cache.assert_not_called()

    def test_empty_cache_called_once_and_last_empty_time_reset_past_interval(self):
        with (
            patch.object(idle_sleeper.zmq, "Poller"),
            patch.object(idle_sleeper, "real_time", side_effect=[10.0, 15.1, 15.2]),
            patch.object(idle_sleeper.current_platform, "empty_cache") as empty_cache,
            envs.SGLANG_EMPTY_CACHE_INTERVAL.override(5),
        ):
            sleeper = idle_sleeper.IdleSleeper(self.create_sockets())
            sleeper.maybe_sleep()

        empty_cache.assert_called_once_with()
        self.assertEqual(sleeper.last_empty_time, 15.2)


class TestRustServerIdleSleeper(CustomTestCase):
    def test_maybe_sleep_waits_on_ingress_with_configured_timeout(self):
        rust_server = MagicMock()
        with (
            patch.object(idle_sleeper, "real_time", return_value=100.0),
            patch.object(idle_sleeper.current_platform, "empty_cache") as empty_cache,
            envs.SGLANG_EMPTY_CACHE_INTERVAL.override(-1),
        ):
            default_sleeper = idle_sleeper.RustServerIdleSleeper(rust_server)
            default_sleeper.maybe_sleep()

            custom_sleeper = idle_sleeper.RustServerIdleSleeper(
                rust_server, timeout_ms=250
            )
            custom_sleeper.maybe_sleep()

        rust_server.wait_ingress.assert_has_calls([call(1000), call(250)])
        empty_cache.assert_not_called()

    def test_empty_cache_not_called_at_or_below_interval(self):
        # Rust equivalent of TestIdleSleeper's boundary test — the eviction
        # bookkeeping is duplicated, not shared, so the strict `>` must be
        # guarded independently in this class too.
        cases = {
            "elapsed_below_interval": [10.0, 13.0],
            "elapsed_equal_interval": [10.0, 15.0],
        }
        for label, real_time_values in cases.items():
            with self.subTest(label=label):
                rust_server = MagicMock()
                with (
                    patch.object(
                        idle_sleeper, "real_time", side_effect=list(real_time_values)
                    ),
                    patch.object(
                        idle_sleeper.current_platform, "empty_cache"
                    ) as empty_cache,
                    envs.SGLANG_EMPTY_CACHE_INTERVAL.override(5),
                ):
                    sleeper = idle_sleeper.RustServerIdleSleeper(rust_server)
                    sleeper.maybe_sleep()

                empty_cache.assert_not_called()

    def test_empty_cache_called_once_and_last_empty_time_reset_past_interval(self):
        # RustServerIdleSleeper duplicates IdleSleeper's eviction bookkeeping
        # rather than sharing it — guard against the two copies drifting apart.
        rust_server = MagicMock()
        with (
            patch.object(idle_sleeper, "real_time", side_effect=[10.0, 15.1, 15.2]),
            patch.object(idle_sleeper.current_platform, "empty_cache") as empty_cache,
            envs.SGLANG_EMPTY_CACHE_INTERVAL.override(5),
        ):
            sleeper = idle_sleeper.RustServerIdleSleeper(rust_server)
            sleeper.maybe_sleep()

        empty_cache.assert_called_once_with()
        self.assertEqual(sleeper.last_empty_time, 15.2)


if __name__ == "__main__":
    unittest.main()
