# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime import platforms


class NVMLUnavailableError(Exception):
    pass


def _entry_point(name: str, result: str | None, dist: str | None = None):
    entry_point = MagicMock(name=f"entry_point_{name}")
    entry_point.name = name
    entry_point.value = f"test_plugin:{name}"
    entry_point.dist = SimpleNamespace(name=dist) if dist else None
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

        current_platform = platforms._current_platform
        current_selection = platforms._current_platform_selection
        self.addCleanup(setattr, platforms, "_current_platform", current_platform)
        self.addCleanup(
            setattr, platforms, "_current_platform_selection", current_selection
        )
        self._reset_current_platform()

    @patch.object(platforms, "entry_points")
    def test_selected_platform_records_its_distribution(self, entry_points):
        class _FakeOot(platforms.Platform):
            _enum = platforms.PlatformEnum.OOT
            device_name = "fake"
            device_type = "fake"

        cases = (("selected", "explicit override"), ("", "automatic discovery"))
        for override, description in cases:
            with (
                self.subTest(description=description),
                patch.object(
                    platforms, "resolve_obj_by_qualname", return_value=_FakeOot
                ),
            ):
                self._reset_current_platform()
                entry_points.return_value = [
                    _entry_point("selected", "vendor.platform.Platform", "vendor-pkg"),
                    _entry_point("inactive", None, "other-pkg"),
                ]
                os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = override

                self.assertEqual(platforms.get_selected_platform_dist(), "vendor-pkg")
                self.assertIsInstance(platforms._current_platform, _FakeOot)

    def test_accessor_reports_no_distribution_for_a_builtin(self):
        os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "cpu"
        self._reset_current_platform()

        self.assertIsNone(platforms.get_selected_platform_dist())

    def _reset_current_platform(self):
        platforms._current_platform = None
        platforms._current_platform_selection = None

    @patch.object(platforms, "entry_points")
    def test_explicit_selection_loads_only_selected_plugin(self, entry_points):
        selected = _entry_point("selected", "vendor.platform.Platform")
        ignored = [
            _entry_point("duplicate", None),
            _entry_point("duplicate", None),
            _entry_point("cuda", "squatter.Platform"),
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

    @patch.object(platforms, "entry_points")
    def test_external_plugin_cannot_return_a_builtin_platform(self, entry_points):
        entry_points.return_value = [
            _entry_point(
                "selected",
                "sglang.multimodal_gen.runtime.platforms.cpu.CpuPlatform",
                "vendor-pkg",
            )
        ]
        os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "selected"

        with self.assertRaisesRegex(TypeError, "PlatformEnum.OOT"):
            platforms.get_selected_platform_dist()

        self.assertIsNone(platforms._current_platform)
        self.assertIsNone(platforms._current_platform_selection)

    def test_external_platform_identity_is_validated_before_publication(self):
        class BadPlatform(platforms.Platform):
            _enum = platforms.PlatformEnum.OOT
            device_name = "fake"
            device_type = "fake"

        selection = platforms._PlatformSelection(
            "vendor.BadPlatform", "selected", "vendor-pkg"
        )
        for attribute in ("device_name", "device_type"):
            with (
                self.subTest(attribute=attribute),
                patch.object(BadPlatform, attribute, " "),
                patch.object(
                    platforms, "_select_current_platform", return_value=selection
                ),
                patch.object(
                    platforms, "_load_platform_class", return_value=BadPlatform
                ),
                self.assertRaisesRegex(TypeError, attribute),
            ):
                platforms._resolve_current_platform()

            self.assertIsNone(platforms._current_platform)
            self.assertIsNone(platforms._current_platform_selection)

    @patch.object(platforms, "entry_points")
    def test_explicit_plugin_must_return_a_class_qualname(self, entry_points):
        entry_points.return_value = [_entry_point("selected", 42, "vendor-pkg")]
        os.environ["SGLANG_DIFFUSION_PLATFORM_OVERRIDE"] = "selected"

        with self.assertRaisesRegex(TypeError, "non-empty class qualname"):
            platforms.resolve_current_platform_cls_qualname()

    @patch.object(platforms, "entry_points")
    def test_a_failing_activation_is_not_downgraded_to_a_builtin(self, entry_points):
        # Skipping the plugin here would run the whole job on the wrong hardware.
        broken = _entry_point("broken", None)
        broken.load.return_value = MagicMock(
            side_effect=RuntimeError("vendor runtime is broken")
        )
        entry_points.return_value = [broken]

        with self.assertRaisesRegex(RuntimeError, "vendor runtime is broken"):
            platforms.resolve_current_platform_cls_qualname()


if __name__ == "__main__":
    unittest.main()
