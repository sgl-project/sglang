# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan3d.paint import (
    Hunyuan3DPaintPreprocessStage,
)


class TestHunyuan3DPaintRemesh(unittest.TestCase):
    def test_remesh_precedes_uv_unwrap(self):
        stage = SimpleNamespace(
            config=SimpleNamespace(paint_use_remesh=True, paint_max_faces=4)
        )
        mesh = Mock(faces=[None] * 8)
        reduced = mesh.simplify_quadric_decimation.return_value
        with patch(
            "sglang.multimodal_gen.runtime.utils.mesh3d_utils.mesh_uv_wrap"
        ) as unwrap:
            result = Hunyuan3DPaintPreprocessStage._unwrap_mesh(stage, mesh)
        mesh.simplify_quadric_decimation.assert_called_once_with(face_count=4)
        unwrap.assert_called_once_with(reduced)
        self.assertIs(result, unwrap.return_value)

    def test_disabled_remesh_preserves_input(self):
        self._check_no_remesh(enabled=False, faces=8)

    def test_mesh_at_or_below_limit_is_preserved(self):
        for faces in (3, 4):
            with self.subTest(faces=faces):
                self._check_no_remesh(enabled=True, faces=faces)

    def _check_no_remesh(self, enabled, faces):
        stage = SimpleNamespace(
            config=SimpleNamespace(paint_use_remesh=enabled, paint_max_faces=4)
        )
        mesh = Mock(faces=[None] * faces)
        with patch(
            "sglang.multimodal_gen.runtime.utils.mesh3d_utils.mesh_uv_wrap"
        ) as unwrap:
            result = Hunyuan3DPaintPreprocessStage._unwrap_mesh(stage, mesh)
        mesh.simplify_quadric_decimation.assert_not_called()
        unwrap.assert_called_once_with(mesh)
        self.assertIs(result, unwrap.return_value)


if __name__ == "__main__":
    unittest.main()
