import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _server_args():
    return SimpleNamespace(
        custom_sigquit_handler=None,
        dcp_size=1,
        enable_metrics=False,
        enable_nccl_nvls=False,
        enable_symm_mem=False,
        gc_threshold=None,
        get_attention_backends=lambda: [],
        nnodes=1,
    )


class TestEngineEnvironment(unittest.TestCase):
    @patch("sglang.srt.entrypoints.engine._log_legacy_kernel_cache_dirs")
    @patch("sglang.srt.entrypoints.engine.mp.set_start_method")
    @patch("sglang.srt.entrypoints.engine.signal.signal")
    @patch("sglang.srt.entrypoints.engine.set_ulimit")
    def test_does_not_override_cuda_module_loading(
        self, _set_ulimit, _signal, _set_start_method, _log_cache_dirs
    ):
        for value in ("DEFAULT", "EAGER", "LAZY"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {
                    "CUDA_MODULE_LOADING": value,
                    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
                },
                clear=True,
            ):
                _set_envs_and_config(_server_args())
                self.assertEqual(os.environ["CUDA_MODULE_LOADING"], value)

    @patch("sglang.srt.entrypoints.engine._log_legacy_kernel_cache_dirs")
    @patch("sglang.srt.entrypoints.engine.mp.set_start_method")
    @patch("sglang.srt.entrypoints.engine.signal.signal")
    @patch("sglang.srt.entrypoints.engine.set_ulimit")
    def test_does_not_set_cuda_module_loading_default(
        self, _set_ulimit, _signal, _set_start_method, _log_cache_dirs
    ):
        with patch.dict(
            os.environ,
            {"SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1"},
            clear=True,
        ):
            _set_envs_and_config(_server_args())
            self.assertNotIn("CUDA_MODULE_LOADING", os.environ)


if __name__ == "__main__":
    unittest.main()
