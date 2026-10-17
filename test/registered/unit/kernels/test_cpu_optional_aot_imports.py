"""CPU torch_native startup must not require optional AOT-only features."""

import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestCPUOptionalAOTImports(CustomTestCase):
    def test_native_startup_without_cpu_extension(self):
        # A fresh process prevents already-imported AOT modules from hiding
        # unconditional CPU imports on an AMX/GPU-equipped CI host.
        code = textwrap.dedent(
            """
            import importlib.abc
            import sys
            import torch

            class BlockAOT(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "sgl_kernel" or fullname.startswith("sgl_kernel."):
                        raise ModuleNotFoundError("optional CPU AOT extension absent", name=fullname)

            sys.meta_path.insert(0, BlockAOT())
            torch.cpu._is_amx_tile_supported = lambda: False
            from sglang.srt.server_args import ServerArgs
            from sglang.srt.managers.scheduler import Scheduler
            from sglang.srt.speculative.ngram_worker import NGRAMWorker
            from sglang.kernels.ops.kvcache.cache_move import copy_all_layer_kv_cache_cpu
            assert "sgl_kernel" not in sys.modules
            try:
                copy_all_layer_kv_cache_cpu(None, None, None, None)
            except ModuleNotFoundError as error:
                assert error.name == "sgl_kernel", error
            else:
                raise AssertionError("AOT-only operations must still require the extension")

            # Once available, the original AOT binding still receives the same
            # arguments and returns its result. The extension is the boundary.
            from types import ModuleType
            from unittest.mock import patch
            aot = ModuleType("sgl_kernel")
            aot.copy_all_layer_kv_cache_cpu = lambda *args: args
            with patch.dict(sys.modules, {"sgl_kernel": aot}):
                assert copy_all_layer_kv_cache_cpu(1, 2, 3, 4) == (1, 2, 3, 4)
            print("CPU_NATIVE_IMPORTS_OK")
            """
        )
        env = os.environ.copy()
        env.update(SGLANG_USE_CPU_ENGINE="1", CUDA_VISIBLE_DEVICES="")
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CPU_NATIVE_IMPORTS_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
