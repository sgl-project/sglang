"""Muse Spark on gfx1250 adds its grouped-MoE rows to aiter's tables instead of replacing them.

``AITER_CONFIG_GROUPED_FMOE``, once set, is the only table aiter's grouped MoE reads: its own tuned
table and the per-model ones are dropped. The Muse Spark expert module used to set the variable to
its own CSV when it was imported, which every server on gfx1250 does, so every other model lost its
tuned tiles. Guards: importing the module leaves the variable alone, and the rows are added only when
a Muse Spark model prepares its experts, beside the tables aiter would have read (or the ones the
variable already names), once.
"""

import importlib
import os
import sys
import types
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="base-b-test-1-gpu-small-amd")

ENV = "AITER_CONFIG_GROUPED_FMOE"


def _fake_aiter_core(default_file):
    """``aiter.jit.core`` reduced to the table aiter resolves when the variable is unset."""
    core = types.ModuleType("aiter.jit.core")
    core.AITER_CONFIGS = types.SimpleNamespace(
        AITER_CONFIG_GROUPED_FMOE_FILE=default_file
    )
    return mock.patch.dict(sys.modules, {"aiter.jit.core": core})


class TestMuseSparkGfx1250GroupedFmoeConfig(CustomTestCase):
    def setUp(self):
        from sglang.srt.rocm_muse_spark import mxfp4_moe

        self.moe = mxfp4_moe

    def test_import_leaves_the_variable_alone(self):
        gfx1250 = types.SimpleNamespace(gcnArchName="AMD gfx1250")
        try:
            with (
                mock.patch.dict(os.environ),
                mock.patch("torch.cuda.get_device_properties", return_value=gfx1250),
            ):
                os.environ.pop(ENV, None)
                importlib.reload(self.moe)
                self.assertTrue(self.moe.GFX1250_A8W4)
                self.assertNotIn(ENV, os.environ)
        finally:
            importlib.reload(self.moe)  # the constants of the device the tests run on

    def test_rows_join_the_tables_aiter_would_read(self):
        with mock.patch.dict(os.environ), _fake_aiter_core("/aiter/merged.csv"):
            os.environ.pop(ENV, None)
            self.moe._add_gfx1250_a8w4_fmoe_rows()
            self.assertEqual(
                os.environ[ENV].split(os.pathsep),
                ["/aiter/merged.csv", self.moe._LOCAL_IDS_CSV],
            )
            self.moe._add_gfx1250_a8w4_fmoe_rows()  # a second MoE layer adds nothing
            self.assertEqual(
                os.environ[ENV].split(os.pathsep),
                ["/aiter/merged.csv", self.moe._LOCAL_IDS_CSV],
            )

    def test_rows_join_the_tables_the_variable_names(self):
        named = os.pathsep.join(["/user/a.csv", "/user/b.csv"])
        # aiter's default is not consulted: the user's tables already replace it
        with mock.patch.dict(os.environ), _fake_aiter_core("/aiter/merged.csv"):
            os.environ[ENV] = named
            self.moe._add_gfx1250_a8w4_fmoe_rows()
            self.assertEqual(
                os.environ[ENV].split(os.pathsep),
                ["/user/a.csv", "/user/b.csv", self.moe._LOCAL_IDS_CSV],
            )

    def test_the_rows_ship_with_the_package(self):
        self.assertTrue(os.path.isfile(self.moe._LOCAL_IDS_CSV))
        with open(self.moe._LOCAL_IDS_CSV) as f:
            header = f.readline().strip().split(",")
        for column in ("gfx", "token", "model_dim", "inter_dim", "expert", "topk"):
            self.assertIn(column, header)


if __name__ == "__main__":
    unittest.main()
