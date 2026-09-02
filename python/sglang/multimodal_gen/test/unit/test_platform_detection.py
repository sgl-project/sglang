# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime import platforms


class NVMLUnavailableError(Exception):
    pass


def _entry_point(name: str, result: str | None):
    entry_point = MagicMock(name=f"entry_point_{name}")
    entry_point.name = name
    entry_point.value = f"test_plugin:{name}"
    entry_point.load.return_value = MagicMock(return_value=result)
    return entry_point


class TestCudaPlatformDetection(unittest.TestCase):
    def test_torch_fallback_excludes_hip(self):
        cases = (
            ("6.0", None),
            (
                None,
                "sglang.multimodal_gen.runtime.platforms.cuda.CudaPlatform",
            ),
        )

        for hip_version, expected in cases:
            with (
                self.subTest(hip_version=hip_version),
                patch(
                    "sglang.multimodal_gen.utils.import_pynvml",
                    side_effect=NVMLUnavailableError,
                ),
                patch.object(os.path, "isfile", return_value=False),
                patch.object(os.path, "exists", return_value=False),
                patch.object(torch.version, "hip", hip_version, create=True),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.cuda, "device_count", return_value=1),
            ):
                self.assertEqual(platforms.cuda_platform_plugin(), expected)


class TestDiffusionPlatformPlugins(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": ""},
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    @patch.object(platforms, "entry_points")
    def test_explicit_selection_loads_only_selected_plugin(self, entry_points):
        selected = _entry_point("selected", "vendor.platform.Platform")
        ignored = [
            _entry_point("duplicate", None),
            _entry_point("duplicate", None),
            _entry_point("cuda", None),
        ]
        entry_points.return_value = [selected, *ignored]
        os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "selected"

        self.assertEqual(
            platforms.resolve_current_platform_cls_qualname(),
            "vendor.platform.Platform",
        )
        selected.load.assert_called_once_with()
        for entry_point in ignored:
            entry_point.load.assert_not_called()

    @patch.object(platforms, "entry_points")
    def test_auto_detection_requires_one_active_plugin(self, entry_points):
        entry_points.return_value = [
            _entry_point("inactive", None),
            _entry_point("active", "vendor.platform.Platform"),
        ]
        self.assertEqual(
            platforms.resolve_current_platform_cls_qualname(),
            "vendor.platform.Platform",
        )

        entry_points.return_value = [
            _entry_point("first", "first.Platform"),
            _entry_point("second", "second.Platform"),
        ]
        with self.assertRaisesRegex(RuntimeError, "Multiple platform plugins"):
            platforms.resolve_current_platform_cls_qualname()

    @patch.object(platforms, "entry_points")
    def test_invalid_entry_point_names_fail_before_import(self, entry_points):
        cases = (
            ([_entry_point("same", None), _entry_point("same", None)], "", "unique"),
            (
                [_entry_point("same", None), _entry_point("same", None)],
                "same",
                "unique",
            ),
            ([_entry_point("XPU", None)], "", "built-in"),
        )
        for entries, selected, message in cases:
            with self.subTest(message=message):
                entry_points.return_value = entries
                os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = selected
                with self.assertRaisesRegex(RuntimeError, message):
                    platforms.resolve_current_platform_cls_qualname()
                for entry_point in entries:
                    entry_point.load.assert_not_called()

    @patch.object(platforms, "entry_points")
    def test_explicit_selection_requires_active_match(self, entry_points):
        cases = (
            ([], ValueError, "not found"),
            ([_entry_point("selected", None)], RuntimeError, "returned None"),
        )
        os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "selected"
        for entries, error_type, message in cases:
            with self.subTest(message=message):
                entry_points.return_value = entries
                with self.assertRaisesRegex(error_type, message):
                    platforms.resolve_current_platform_cls_qualname()

    def test_builtin_override_bypasses_plugin_selection(self):
        expected = {
            "cpu": "sglang.multimodal_gen.runtime.platforms.cpu.CpuPlatform",
            "cuda": "sglang.multimodal_gen.runtime.platforms.cuda.CudaPlatform",
            "rocm": "sglang.multimodal_gen.runtime.platforms.rocm.RocmPlatform",
            "mps": "sglang.multimodal_gen.runtime.platforms.mps.MpsPlatform",
            "npu": "sglang.multimodal_gen.runtime.platforms.npu.NPUPlatformBase",
            "musa": "sglang.multimodal_gen.runtime.platforms.musa.MusaPlatform",
        }

        for name, qualname in expected.items():
            with (
                self.subTest(name=name),
                patch.object(platforms, "entry_points") as entry_points,
            ):
                os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = name
                self.assertEqual(
                    platforms.resolve_current_platform_cls_qualname(),
                    qualname,
                )
                entry_points.assert_not_called()

    @patch.object(platforms, "entry_points")
    def test_xpu_override_remains_unsupported(self, entry_points):
        os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "xpu"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            platforms.resolve_current_platform_cls_qualname()
        entry_points.assert_not_called()

    @patch.object(platforms, "entry_points", return_value=[])
    def test_xpu_keeps_automatic_detection_priority(self, _entry_points):
        xpu_qualname = "sglang.multimodal_gen.runtime.platforms.xpu.XpuPlatform"
        detectors = {
            "mps": MagicMock(return_value=None),
            "xpu": MagicMock(return_value=xpu_qualname),
            "rocm": MagicMock(return_value=None),
            "cuda": MagicMock(return_value=None),
            "npu": MagicMock(return_value=None),
            "musa": MagicMock(return_value=None),
            "cpu": MagicMock(return_value=None),
        }
        with patch.object(platforms, "builtin_platform_plugins", detectors):
            self.assertEqual(
                platforms.resolve_current_platform_cls_qualname(), xpu_qualname
            )
        detectors["mps"].assert_called_once_with()
        detectors["xpu"].assert_called_once_with()
        for name in ("rocm", "cuda", "npu", "musa", "cpu"):
            detectors[name].assert_not_called()

    def test_builtin_name_set_matches_qualnames_and_detectors(self):
        self.assertEqual(
            platforms.BUILTIN_PLATFORM_NAMES,
            set(platforms._BUILTIN_PLATFORM_QUALNAMES),
        )
        self.assertEqual(
            platforms.BUILTIN_PLATFORM_NAMES,
            set(platforms.builtin_platform_plugins),
        )

    def test_external_platform_must_use_oot_identity(self):
        class WrongPlatform(platforms.Platform):
            _enum = platforms.PlatformEnum.CPU

        with (
            patch.object(
                platforms, "resolve_obj_by_qualname", return_value=WrongPlatform
            ),
            self.assertRaisesRegex(TypeError, "PlatformEnum.OOT"),
        ):
            platforms._load_platform_class("vendor.WrongPlatform")


if __name__ == "__main__":
    unittest.main()
