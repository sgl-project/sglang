"""Backend configuration contracts; no NPU device is required."""

import os
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu import compile_backend as backend_module
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="base-a-test-npu")


class TestNpugraphExBackend(unittest.TestCase):
    def setUp(self):
        self.env_patch = mock.patch.dict(os.environ, {}, clear=False)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        os.environ.pop("SGLANG_NPU_ENABLE_STATIC_KERNEL", None)
        self.config = types.SimpleNamespace()
        self.backend = object()
        self.package = types.ModuleType("npugraph_ex")
        self.package.__path__ = []
        self.package.CompilerConfig = mock.Mock(return_value=self.config)
        self.package.get_npu_backend = mock.Mock(return_value=self.backend)
        self.configs = types.ModuleType("npugraph_ex.configs")
        self.configs.__path__ = []
        self.config_module = types.ModuleType("npugraph_ex.configs.compiler_config")
        self.config_module._process_kwargs_options = mock.Mock()
        self.modules = {
            "npugraph_ex": self.package,
            "npugraph_ex.configs": self.configs,
            "npugraph_ex.configs.compiler_config": self.config_module,
            "npugraph_ex.configs.npugraphex_config": None,
        }
        self.npu = types.SimpleNamespace(set_compile_mode=mock.Mock())

    def call_backend(self, **kwargs):
        with (
            mock.patch.dict(sys.modules, self.modules),
            mock.patch.object(torch, "npu", self.npu, create=True),
        ):
            return backend_module.get_npugraph_ex_backend(**kwargs)

    def options(self):
        return self.config_module._process_kwargs_options.call_args.args[1]

    def test_backend_options(self):
        """The option set handed to npugraph_ex, with and without statickernel."""
        base = {
            "force_eager": True,
            "inplace_pass": False,
            "clone_input": False,
            "clone_output": False,
        }
        self.assertIs(self.call_backend(), self.backend)
        self.assertIs(
            self.config_module._process_kwargs_options.call_args.args[0], self.config
        )
        self.assertEqual(
            self.options(), {"options": {**base, "static_kernel_compile": False}}
        )
        self.package.get_npu_backend.assert_called_once_with(
            compiler_config=self.config
        )
        self.npu.set_compile_mode.assert_called_once_with(jit_compile=False)

        os.environ["SGLANG_NPU_ENABLE_STATIC_KERNEL"] = "1"
        self.call_backend()
        self.assertEqual(
            self.options(), {"options": {**base, "static_kernel_compile": True}}
        )

    def test_legacy_config_layout(self):
        self.modules["npugraph_ex.configs.compiler_config"] = None
        self.modules["npugraph_ex.configs.npugraphex_config"] = self.config_module
        self.assertIs(self.call_backend(), self.backend)
        self.config_module._process_kwargs_options.assert_called_once()

    def test_missing_transitive_dependency_is_not_a_legacy_layout(self):
        failure = ModuleNotFoundError("missing runtime library", name="runtime_lib")
        with mock.patch.object(backend_module, "import_module", side_effect=failure):
            with self.assertRaises(ModuleNotFoundError) as caught:
                self.call_backend()
        self.assertIs(caught.exception, failure)

    def test_statickernel_prerequisites(self):
        os.environ["SGLANG_NPU_ENABLE_STATIC_KERNEL"] = "1"
        with self.assertRaisesRegex(ValueError, "enable-torch-compile"):
            backend_module.validate_npugraph_ex_config(
                enable_compile=False, graph_enabled=True, world_size=1
            )
        with self.assertRaisesRegex(ValueError, "capture"):
            backend_module.validate_npugraph_ex_config(
                enable_compile=True, graph_enabled=False, world_size=1
            )

    def test_check_server_args_rejects_disabled_graph(self):
        """The validator is wired into the canonical argument check."""
        from sglang.srt.arg_groups import validation_hook
        from sglang.srt.model_executor.cuda_graph_config import Backend

        os.environ["SGLANG_NPU_ENABLE_STATIC_KERNEL"] = "1"
        cfg = types.SimpleNamespace(
            ep_join_mode=None,
            dp_size=1,
            tp_size=1,
            pp_size=1,
            nnodes=1,
            pp_max_micro_batch_size=None,
            device="npu",
            enable_torch_compile=True,
            disable_cuda_graph=False,
            cuda_graph_config=types.SimpleNamespace(
                decode=types.SimpleNamespace(backend=Backend.DISABLED)
            ),
        )
        with (
            mock.patch.object(validation_hook, "resolving_view", return_value=cfg),
            mock.patch.dict(
                sys.modules,
                {"sglang.srt.hardware_backend.npu.compile_backend": backend_module},
            ),
        ):
            with self.assertRaisesRegex(ValueError, "decode NPU graph"):
                validation_hook.check_server_args(object())

    def test_check_server_args_reads_an_unresolved_graph_config(self):
        """cuda_graph_config is None until the resolution pipeline fills it."""
        from sglang.srt.arg_groups import validation_hook

        os.environ["SGLANG_NPU_ENABLE_STATIC_KERNEL"] = "1"
        cfg = types.SimpleNamespace(
            ep_join_mode=None,
            dp_size=1,
            tp_size=1,
            pp_size=1,
            nnodes=1,
            pp_max_micro_batch_size=None,
            device="npu",
            enable_torch_compile=True,
            disable_cuda_graph=False,
            cuda_graph_config=None,
        )
        with (
            mock.patch.object(validation_hook, "resolving_view", return_value=cfg),
            mock.patch.dict(
                sys.modules,
                {"sglang.srt.hardware_backend.npu.compile_backend": backend_module},
            ),
        ):
            with self.assertRaisesRegex(ValueError, "decode NPU graph"):
                validation_hook.check_server_args(object())


if __name__ == "__main__":
    unittest.main()
