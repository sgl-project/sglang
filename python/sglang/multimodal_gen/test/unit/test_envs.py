import os
import sys
import types
import unittest
from importlib import util
from unittest.mock import patch


def _get_bool_env_var(key, default="false"):
    return os.environ.get(key, str(default)).lower() == "true"


class TestCacheDitEnvVars(unittest.TestCase):
    def setUp(self):
        self._saved_attn = os.environ.pop("SGLANG_CACHE_DIT_ATTN_BACKEND", None)
        self._saved_compile = os.environ.pop("SGLANG_CACHE_DIT_MINDIESD_COMPILE", None)

    def tearDown(self):
        os.environ.pop("SGLANG_CACHE_DIT_ATTN_BACKEND", None)
        os.environ.pop("SGLANG_CACHE_DIT_MINDIESD_COMPILE", None)
        if self._saved_attn is not None:
            os.environ["SGLANG_CACHE_DIT_ATTN_BACKEND"] = self._saved_attn
        if self._saved_compile is not None:
            os.environ["SGLANG_CACHE_DIT_MINDIESD_COMPILE"] = self._saved_compile

    def _import_envs(self):
        stub_utils = types.ModuleType("sglang.multimodal_gen.runtime.utils.common")
        stub_utils.get_bool_env_var = _get_bool_env_var

        stubs = {
            "sglang": types.ModuleType("sglang"),
            "sglang.multimodal_gen": types.ModuleType("sglang.multimodal_gen"),
            "sglang.multimodal_gen.runtime": types.ModuleType(
                "sglang.multimodal_gen.runtime"
            ),
            "sglang.multimodal_gen.runtime.utils": types.ModuleType(
                "sglang.multimodal_gen.runtime.utils"
            ),
            "sglang.multimodal_gen.runtime.utils.common": stub_utils,
        }

        module_path = os.path.join(os.path.dirname(__file__), "..", "..", "envs.py")
        with patch.dict(sys.modules, stubs):
            spec = util.spec_from_file_location("envs_test_target", module_path)
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module

    def test_attn_backend_set_reads_correctly(self):
        os.environ["SGLANG_CACHE_DIT_ATTN_BACKEND"] = "_mindiesd_laser"
        module = self._import_envs()
        self.assertEqual(module.SGLANG_CACHE_DIT_ATTN_BACKEND, "_mindiesd_laser")

    def test_attn_backend_unset_defaults_none(self):
        module = self._import_envs()
        val = module.SGLANG_CACHE_DIT_ATTN_BACKEND
        self.assertTrue(val is None or val == "")

    def test_mindiesd_compile_default_false(self):
        module = self._import_envs()
        self.assertFalse(module.SGLANG_CACHE_DIT_MINDIESD_COMPILE)

    def test_mindiesd_compile_set_true(self):
        os.environ["SGLANG_CACHE_DIT_MINDIESD_COMPILE"] = "true"
        module = self._import_envs()
        self.assertTrue(module.SGLANG_CACHE_DIT_MINDIESD_COMPILE)


if __name__ == "__main__":
    unittest.main()
