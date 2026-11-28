"""MLX buffer-cache limit: default formula and SGLANG_MLX_CACHE_LIMIT_GB overrides."""

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

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx import model_runner
    from sglang.srt.hardware_backend.mlx.model_runner import (
        DEFAULT_CACHE_LIMIT_FRACTION,
        MIN_DEFAULT_CACHE_LIMIT_BYTES,
        MlxModelRunner,
        default_cache_limit_bytes,
    )

GIB = 1024**3


@unittest.skipUnless(_HAS_MLX, "requires mlx + mlx_lm")
class TestCacheLimit(CustomTestCase):
    def test_default_formula(self):
        big = {"max_recommended_working_set_size": 96 * GIB}
        self.assertEqual(
            default_cache_limit_bytes(big), int(96 * GIB * DEFAULT_CACHE_LIMIT_FRACTION)
        )
        small = {"max_recommended_working_set_size": 8 * GIB}
        self.assertEqual(
            default_cache_limit_bytes(small), MIN_DEFAULT_CACHE_LIMIT_BYTES
        )
        self.assertEqual(default_cache_limit_bytes({}), MIN_DEFAULT_CACHE_LIMIT_BYTES)

    def _apply(self, env_value):
        runner = MlxModelRunner.__new__(MlxModelRunner)
        info = {"max_recommended_working_set_size": 40 * GIB}
        with (
            mock.patch.object(model_runner.mx, "set_cache_limit") as set_limit,
            mock.patch.object(model_runner.mx, "device_info", return_value=info),
        ):
            if env_value is None:
                envs.SGLANG_MLX_CACHE_LIMIT_GB.clear()
                runner._apply_cache_limit()
            else:
                with envs.SGLANG_MLX_CACHE_LIMIT_GB.override(env_value):
                    runner._apply_cache_limit()
        return set_limit

    def test_env_resolution(self):
        self._apply(None).assert_called_once_with(
            int(40 * GIB * DEFAULT_CACHE_LIMIT_FRACTION)
        )
        self._apply(2.5).assert_called_once_with(int(2.5 * GIB))
        self._apply(0).assert_called_once_with(0)
        with self.assertRaises(ValueError):
            self._apply(-1)


if __name__ == "__main__":
    unittest.main()
