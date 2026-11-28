import ast
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import sglang
from sglang.srt.layers.attention import flashattention_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_BACKEND_SOURCE = (
    Path(sglang.__file__).parent / "srt/layers/attention/flashattention_backend.py"
)
_ENTRY_POINTS = ("flash_attn_with_kvcache", "flash_attn_varlen_func")


class TestFlashAttentionKernelDispatch(CustomTestCase):
    def test_kernel_calls_use_selected_entry_points(self):
        # A call through any name other than self.flash_attn_* bypasses the build
        # __init__ selected, e.g. generic FA4 on SM12x, or FA3 when `ver` is omitted.
        tree = ast.parse(_BACKEND_SOURCE.read_text())
        bypasses = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in _ENTRY_POINTS:
                bypasses.append(f"line {node.lineno}: {func.id}(...)")
            elif (
                isinstance(func, ast.Attribute)
                and func.attr in _ENTRY_POINTS
                and not (isinstance(func.value, ast.Name) and func.value.id == "self")
            ):
                bypasses.append(f"line {node.lineno}: {ast.unparse(func)}(...)")
        self.assertEqual(bypasses, [])

    def test_fa3_backend_loads_fa4_when_a_call_needs_it(self):
        fa3 = (lambda *a, **k: "fa3_varlen", lambda *a, **k: "fa3_kvcache")
        fa4 = (lambda *a, **k: "fa4_varlen", lambda *a, **k: "fa4_kvcache", None)
        backend = flashattention_backend.FlashAttentionBackend.__new__(
            flashattention_backend.FlashAttentionBackend
        )
        backend.fa_impl_ver = 3
        backend._fa_kernels = {3: fa3}
        with (
            patch.object(
                flashattention_backend, "_load_fa4_kernels", return_value=fa4
            ) as load,
            patch.object(
                flashattention_backend, "get_device_capability", return_value=(9, 0)
            ),
        ):
            self.assertEqual(backend.flash_attn_with_kvcache(ver=4), "fa4_kvcache")
            self.assertEqual(backend.flash_attn_varlen_func(ver=4), "fa4_varlen")
            self.assertEqual(backend.flash_attn_with_kvcache(ver=3), "fa3_kvcache")
        load.assert_called_once()

    def test_image_masks_need_fa4_capable_paged_mha(self):
        class OwnBuild(flashattention_backend.FlashAttentionBackend):
            pass

        bf16, fp8 = torch.bfloat16, torch.float8_e4m3fn
        cases = (
            (
                flashattention_backend.FlashAttentionBackend,
                3,
                False,
                False,
                bf16,
                (9, 0),
                True,
            ),
            (
                flashattention_backend.FlashAttentionBackend,
                3,
                False,
                False,
                bf16,
                (8, 0),
                False,
            ),
            (
                flashattention_backend.FlashAttentionBackend,
                3,
                False,
                False,
                fp8,
                (9, 0),
                False,
            ),
            (
                flashattention_backend.FlashAttentionBackend,
                3,
                False,
                True,
                bf16,
                (9, 0),
                False,
            ),
            (OwnBuild, 3, False, False, bf16, (9, 0), False),
            (
                flashattention_backend.FlashAttentionBackend,
                3,
                True,
                False,
                bf16,
                (9, 0),
                True,
            ),
            (
                flashattention_backend.FlashAttentionBackend,
                4,
                False,
                False,
                fp8,
                (10, 0),
                True,
            ),
        )
        for cls, ver, upgrades, mla, kv_dtype, capability, expected in cases:
            with self.subTest(
                cls=cls.__name__,
                ver=ver,
                upgrades=upgrades,
                mla=mla,
                kv_dtype=kv_dtype,
                capability=capability,
            ):
                backend = cls.__new__(cls)
                backend.fa_impl_ver = ver
                backend.fa3_upgrades_to_fa4 = upgrades
                backend.use_mla = mla
                backend.kv_cache_dtype = kv_dtype
                with patch.object(
                    flashattention_backend,
                    "get_device_capability",
                    return_value=capability,
                ):
                    self.assertEqual(backend.supports_image_masks(), expected)


if __name__ == "__main__":
    unittest.main()
