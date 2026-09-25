"""Unit tests for the MLX buffer-cache limit applied at runner construction.

MLX's default cache limit equals its memory limit (roughly all physical RAM), and
``MlxModelRunner`` pins MLX allocations with ``mx.set_wired_limit``, so an uncapped
buffer cache wires up the machine under sustained load.  These tests pin the cap's
resolution rules:

- ``SGLANG_MLX_CACHE_LIMIT_GB`` unset: 10% of Metal's recommended working set, and
  never below 1 GiB;
- set to a non-negative value: exactly that many GiB (``0`` disables the cache);
- set to a negative value: rejected.
"""

from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)
_SKIP_REASON = "requires mlx + mlx_lm"

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx import model_runner
    from sglang.srt.hardware_backend.mlx.model_runner import (
        DEFAULT_CACHE_LIMIT_FRACTION,
        MIN_DEFAULT_CACHE_LIMIT_BYTES,
        MlxModelRunner,
        default_cache_limit_bytes,
    )

GIB = 1024**3


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestDefaultCacheLimit(CustomTestCase):
    def test_fraction_of_recommended_working_set(self):
        working_set = 96 * GIB
        self.assertEqual(
            default_cache_limit_bytes(
                {"max_recommended_working_set_size": working_set}
            ),
            int(working_set * DEFAULT_CACHE_LIMIT_FRACTION),
        )

    def test_floor_on_small_devices(self):
        # 10% of an 8 GiB working set is below the floor.
        self.assertEqual(
            default_cache_limit_bytes({"max_recommended_working_set_size": 8 * GIB}),
            MIN_DEFAULT_CACHE_LIMIT_BYTES,
        )

    def test_missing_device_info_falls_back_to_floor(self):
        self.assertEqual(default_cache_limit_bytes({}), MIN_DEFAULT_CACHE_LIMIT_BYTES)


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestApplyCacheLimit(CustomTestCase):
    def _apply(self, env_value):
        runner = MlxModelRunner.__new__(MlxModelRunner)
        device_info = {"max_recommended_working_set_size": 40 * GIB}
        with (
            mock.patch.object(model_runner.mx, "set_cache_limit") as set_limit,
            mock.patch.object(model_runner.mx, "device_info", return_value=device_info),
        ):
            if env_value is None:
                envs.SGLANG_MLX_CACHE_LIMIT_GB.clear()
                runner._apply_cache_limit()
            else:
                with envs.SGLANG_MLX_CACHE_LIMIT_GB.override(env_value):
                    runner._apply_cache_limit()
        return set_limit

    def test_unset_applies_default(self):
        set_limit = self._apply(None)
        set_limit.assert_called_once_with(int(40 * GIB * DEFAULT_CACHE_LIMIT_FRACTION))

    def test_explicit_value_is_used_verbatim(self):
        set_limit = self._apply(2.5)
        set_limit.assert_called_once_with(int(2.5 * GIB))

    def test_zero_disables_the_cache(self):
        set_limit = self._apply(0)
        set_limit.assert_called_once_with(0)

    def test_negative_is_rejected(self):
        with self.assertRaises(ValueError):
            self._apply(-1)


if __name__ == "__main__":
    unittest.main()
