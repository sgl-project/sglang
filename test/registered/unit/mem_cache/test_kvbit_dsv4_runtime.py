import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.srt.mem_cache.kvbit_dsv4_runtime import (
    DSV4KVBitRuntimeCapability,
    require_dsv4_kvbit_runtime_capability,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


class TestDSV4INT4Runtime(unittest.TestCase):
    def test_injected_capability_combinations(self):
        require_dsv4_kvbit_runtime_capability(DSV4KVBitRuntimeCapability(True, True))
        for write, decode in ((False, False), (True, False), (False, True)):
            with self.subTest(write=write, decode=decode):
                with self.assertRaisesRegex(
                    RuntimeError, "scratch fallback is disabled"
                ):
                    require_dsv4_kvbit_runtime_capability(
                        DSV4KVBitRuntimeCapability(write, decode)
                    )

    def test_default_gate_imports_and_probes_aot(self):
        probe = Mock(spec=lambda: None)
        wrapper = SimpleNamespace(require_kvbit_int4_extension=probe)
        with patch(
            "sglang.srt.mem_cache.kvbit_dsv4_runtime.import_module",
            autospec=True,
            side_effect=[ModuleType("triton"), wrapper],
        ) as loader:
            require_dsv4_kvbit_runtime_capability()
            self.assertEqual(
                loader.call_args_list,
                [call("triton"), call("sgl_kernel.kvbit_flash_mla")],
            )
            probe.assert_called_once_with()

    def test_missing_dependency_and_broken_extension_fail_before_allocation(self):
        missing = ImportError("extension missing")
        for dependencies in (
            [missing],
            [ModuleType("triton"), missing],
            [
                ModuleType("triton"),
                SimpleNamespace(
                    require_kvbit_int4_extension=Mock(
                        spec=lambda: None, side_effect=missing
                    )
                ),
            ],
        ):
            with (
                self.subTest(dependencies=len(dependencies)),
                patch(
                    "sglang.srt.mem_cache.kvbit_dsv4_runtime.import_module",
                    autospec=True,
                    side_effect=dependencies,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "AOT extension") as caught:
                    require_dsv4_kvbit_runtime_capability()
                self.assertIs(caught.exception.__cause__, missing)


class TestDSV4INT4ExtensionProbe(unittest.TestCase):
    def _wrapper(self):
        # Only the external native package is substituted. Execute the real wrapper.
        package = ModuleType("sgl_kernel")
        package.kvbit_flashmla_ops = ModuleType("sgl_kernel.kvbit_flashmla_ops")
        flash_mla = ModuleType("sgl_kernel.flash_mla")
        flash_mla.FlashMLASchedMeta = object
        path = ROOT / "python/sglang/kernels/aot/python/sgl_kernel/kvbit_flash_mla.py"
        spec = importlib.util.spec_from_file_location("_kvbit_wrapper_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules, {"sgl_kernel": package, "sgl_kernel.flash_mla": flash_mla}
        ):
            spec.loader.exec_module(module)
        return module

    def test_missing_stale_and_cpu_only_extensions_are_rejected(self):
        wrapper = self._wrapper()
        decode = Mock(spec=lambda: None)
        cases = (
            (SimpleNamespace(), False, "missing"),
            (SimpleNamespace(kvbit_int4_sparse_decode_fwd=decode), False, "missing"),
            (
                SimpleNamespace(
                    kvbit_int4_sparse_decode_fwd=decode,
                    kvbit_int4_abi_version=lambda: 1,
                ),
                True,
                "ABI mismatch",
            ),
            (
                SimpleNamespace(
                    kvbit_int4_sparse_decode_fwd=decode,
                    kvbit_int4_abi_version=lambda: 2,
                ),
                False,
                "no CUDA",
            ),
        )
        for ops, cuda, error in cases:
            with (
                self.subTest(error=error),
                patch.object(wrapper.torch.ops, "sgl_kernel", ops),
                patch.object(
                    wrapper.torch._C,
                    "_dispatch_has_kernel_for_dispatch_key",
                    return_value=cuda,
                ),
            ):
                with self.assertRaisesRegex(ImportError, error):
                    wrapper.require_kvbit_int4_extension()

    def test_registered_cuda_extension_and_import_error_cause(self):
        wrapper = self._wrapper()
        ops = SimpleNamespace(
            kvbit_int4_sparse_decode_fwd=Mock(spec=lambda: None),
            kvbit_int4_abi_version=lambda: 2,
        )
        with (
            patch.object(wrapper.torch.ops, "sgl_kernel", ops),
            patch.object(
                wrapper.torch._C,
                "_dispatch_has_kernel_for_dispatch_key",
                return_value=True,
            ) as dispatch,
        ):
            wrapper.require_kvbit_int4_extension()
            dispatch.assert_called_once_with(
                "sgl_kernel::kvbit_int4_sparse_decode_fwd", "CUDA"
            )
        original = OSError("shared object not found")
        wrapper._kvbit_flashmla_import_error = original
        with self.assertRaises(ImportError) as caught:
            wrapper.require_kvbit_int4_extension()
        self.assertIs(caught.exception.__cause__, original)


if __name__ == "__main__":
    unittest.main()
