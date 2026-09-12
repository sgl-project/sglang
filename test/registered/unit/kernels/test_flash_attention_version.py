"""Validate the FA3 wrapper contract without loading CUDA extensions."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestFlashAttentionVersion(CustomTestCase):
    def setUp(self):
        package = types.ModuleType("sgl_kernel")
        source_dir = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/kernels/aot/python/sgl_kernel"
        )
        package.__path__ = [str(source_dir)]
        native = types.ModuleType("sgl_kernel.flash_ops")
        imports = patch.dict(
            sys.modules, {"sgl_kernel": package, "sgl_kernel.flash_ops": native}
        )
        imports.start()
        self.addCleanup(imports.stop)
        spec = importlib.util.spec_from_file_location(
            "_flash_attention_version_test", source_dir / "flash_attn.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_unsupported_versions_fail_before_tensor_or_device_access(self):
        for version in (0, 2, 4):
            for name, args in (
                ("flash_attn_with_kvcache", (None, None, None)),
                ("flash_attn_varlen_func", (None, None, None, None, None)),
            ):
                with self.subTest(version=version, function=name):
                    with self.assertRaisesRegex(ValueError, "only supports ver=3"):
                        getattr(self.module, name)(*args, ver=version)

    def test_default_and_explicit_fa3_dispatch(self):
        q = torch.zeros(1, 1, 2, 64)
        kv = torch.zeros(1, 8, 2, 64)
        expected = torch.ones_like(q)
        for kwargs in ({}, {"ver": 3}):
            for name, args, extra in (
                ("flash_attn_with_kvcache", (q, kv, kv), {}),
                (
                    "flash_attn_varlen_func",
                    (q[0], kv[0], kv[0], None, None),
                    {"max_seqlen_q": 1, "max_seqlen_k": 8},
                ),
            ):
                with self.subTest(function=name, kwargs=kwargs):
                    with (
                        patch.object(
                            self.module, "is_fa3_supported", return_value=True
                        ),
                        patch.object(torch.ops.sgl_kernel, "fwd", create=True) as fwd,
                    ):
                        fwd.default.return_value = (expected, None)
                        result = getattr(self.module, name)(*args, **extra, **kwargs)
                        self.assertIs(result, expected)
                        fwd.default.assert_called_once()


if __name__ == "__main__":
    unittest.main()
