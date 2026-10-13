"""The Kimi-K3 AITER BF16 profile is opt-in, ROCm-only, and set at model build.

AITER reads ``AITER_CONFIG_GEMM_BF16`` when it first loads its BF16 GEMM table,
so what matters is that the variable is set before that happens, that an
operator's own value wins, and that a late call says it cannot apply.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.models import kimi_k3_rocm_bootstrap as bootstrap
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VAR = "AITER_CONFIG_GEMM_BF16"
MODULE = "sglang.srt.models.kimi_k3_rocm_bootstrap"


class _LoadedTable:
    @staticmethod
    def cache_info():
        return SimpleNamespace(currsize=1)


class TestKimiK3RocmBootstrap(CustomTestCase):
    def setUp(self):
        bootstrap.maybe_set_aiter_m16384_profile.cache_clear()
        self._tmp = tempfile.TemporaryDirectory()
        aiter_root = Path(self._tmp.name) / "aiter"
        (aiter_root / "configs").mkdir(parents=True)
        (aiter_root / "configs" / "bf16_tuned_gemm.csv").write_text("")
        self._spec = SimpleNamespace(origin=str(aiter_root / "__init__.py"))
        self._base = str(aiter_root / "configs" / "bf16_tuned_gemm.csv")

    def tearDown(self):
        bootstrap.maybe_set_aiter_m16384_profile.cache_clear()
        self._tmp.cleanup()

    def _run(self, *, hip: bool, enabled: bool, modules=None):
        with (
            patch.dict(os.environ),
            patch.dict(sys.modules, modules or {}),
            patch(f"{MODULE}.torch.version.hip", "6.4" if hip else None),
            patch(f"{MODULE}.importlib.util.find_spec", return_value=self._spec),
            envs.SGLANG_ROCM_K3_AITER_M16384_PROFILE.override(enabled),
        ):
            os.environ.pop(VAR, None)
            bootstrap.maybe_set_aiter_m16384_profile()
            return os.environ.get(VAR)

    def test_off_by_default(self):
        self.assertIsNone(self._run(hip=True, enabled=False))

    def test_absent_off_hip(self):
        self.assertIsNone(self._run(hip=False, enabled=True))

    def test_appends_the_k3_profile_on_hip(self):
        paths = self._run(hip=True, enabled=True).split(os.pathsep)
        self.assertEqual(paths[0], self._base)
        self.assertTrue(paths[-1].endswith("kimik3_m16384_profile.csv"))

    def test_an_operators_value_survives(self):
        with patch.dict(os.environ, {VAR: "/tuned.csv"}):
            with (
                patch(f"{MODULE}.torch.version.hip", "6.4"),
                envs.SGLANG_ROCM_K3_AITER_M16384_PROFILE.override(True),
            ):
                bootstrap.maybe_set_aiter_m16384_profile()
            self.assertEqual(os.environ[VAR], "/tuned.csv")

    def test_warns_when_aiter_already_loaded_the_table(self):
        tuned_gemm = SimpleNamespace(get_GEMM_A16W16_config_=_LoadedTable())
        with self.assertLogs(MODULE, level="WARNING"):
            value = self._run(
                hip=True, enabled=True, modules={"aiter.tuned_gemm": tuned_gemm}
            )
        self.assertIsNotNone(value)


if __name__ == "__main__":
    unittest.main()
