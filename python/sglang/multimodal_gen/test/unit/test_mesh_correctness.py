"""Regressions for scale-invariant mesh acceptance and isolated sampling RNGs."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import trimesh

from sglang.multimodal_gen.test.server import test_server_utils as mesh_utils
from sglang.test.test_utils import CustomTestCase


class TestMeshCorrectness(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.generated = self.root / "generated.ply"
        self.reference = self.root / "reference.ply"
        downloader = patch.object(
            mesh_utils, "_download_reference_mesh", return_value=self.reference
        )
        downloader.start()
        self.addCleanup(downloader.stop)

    @staticmethod
    def plane(scale=1.0, offset=0.0):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        vertices[:, 2] += offset * np.sqrt(2)
        return trimesh.Trimesh(
            vertices=vertices * scale, faces=[[0, 1, 2], [0, 2, 3]], process=False
        )

    def validate(self, **kwargs):
        return mesh_utils.validate_mesh_correctness(str(self.generated), **kwargs)

    def save_pair(self, scale=1.0, offset=0.0):
        self.plane(scale).export(self.reference)
        self.plane(scale, offset).export(self.generated)

    def test_common_scale_preserves_acceptance(self):
        # The old linear threshold rejects the large, slightly shifted plane,
        # and accepts the small, badly shifted plane.
        for scale in (0.01, 1, 1000):
            for offset in (0.02, 0.25, 0.08):
                with self.subTest(scale=scale, offset=offset):
                    self.save_pair(scale, offset)
                    if offset == 0.02:
                        self.assertIsNone(self.validate())
                    else:
                        with self.assertRaises(AssertionError):
                            self.validate()

    def test_generated_size_and_position_errors(self):
        self.save_pair()
        for mesh in (self.plane(10), self.plane(offset=10)):
            mesh.export(self.generated)
            with self.assertRaises(AssertionError):
                self.validate()

    def test_corrupt_glb_is_rejected(self):
        self.save_pair()
        self.generated = self.root / "generated.glb"
        self.generated.write_bytes(b"glTF\x02\x00\x00\x00broken")
        with self.assertRaisesRegex(AssertionError, "Invalid generated mesh"):
            self.validate()

    def test_missing_performance_baseline_does_not_hide_mesh_result(self):
        """A missing perf reference used to abort before validating real geometry."""
        from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
        from sglang.multimodal_gen.test.runner.pytest_runner import (
            _is_retryable_failure,
        )
        from sglang.multimodal_gen.test.server import test_server_common as common
        from sglang.multimodal_gen.test.server.testcase_configs import (
            DiffusionSamplingParams,
            DiffusionServerArgs,
            DiffusionTestCase,
        )

        case = DiffusionTestCase(
            "mesh_missing_baseline",
            DiffusionServerArgs(
                model_path="test", modality="3d", custom_validator="mesh"
            ),
            DiffusionSamplingParams(prompt="test"),
            run_consistency_check=False,
            run_models_api_check=False,
            run_t2v_input_reference_check=False,
            perf_repeat_requests=1,
        )
        record = RequestPerfRecord(
            request_id="mesh",
            commit_hash="test",
            tag="mesh",
            stages=[],
            steps=[100],
            total_duration_ms=1000,
        )
        runner = common.DiffusionServerBase()
        runner._perf_results = []
        baseline_path = self.root / "candidate.json"
        with (
            patch.dict(
                "os.environ", {"SGLANG_GEN_GT": "0", "SGLANG_GEN_BASELINE": "0"}
            ),
            patch.object(common, "get_generate_fn", return_value=None),
            patch.object(runner, "run_and_collect", return_value=(record, b"mesh")),
            patch.object(common.BASELINE_CONFIG, "scenarios", {}),
            patch.object(
                common, "get_perf_baseline_update_path", return_value=baseline_path
            ),
            patch.dict(mesh_utils.MESH_OUTPUT_PATHS, {}, clear=True),
        ):
            for bad_geometry in (False, True):
                with self.subTest(bad_geometry=bad_geometry):
                    self.save_pair(offset=10 if bad_geometry else 0)
                    mesh_utils.MESH_OUTPUT_PATHS[case.id] = str(self.generated)
                    with self.assertRaises(
                        common.PerformanceValidationError
                    ) as failure:
                        runner._test_diffusion_request(
                            case, SimpleNamespace(load_time_ms=100), 1
                        )
                    message = str(failure.exception)
                    self.assertIn("[performance]", message)
                    self.assertNotIn(case.id, mesh_utils.MESH_OUTPUT_PATHS)
                    self.assertEqual("[mesh correctness]" in message, bad_geometry)
                    self.assertEqual(_is_retryable_failure(message), not bad_geometry)

    def test_missing_mesh_artifact_fails_correctness(self):
        """An absent generated mesh must not silently pass the mesh branch."""
        from sglang.multimodal_gen.test.server import test_server_common as common
        from sglang.multimodal_gen.test.server.testcase_configs import (
            DiffusionSamplingParams,
            DiffusionServerArgs,
            DiffusionTestCase,
        )

        case = DiffusionTestCase(
            "missing_mesh",
            DiffusionServerArgs(
                model_path="test", modality="3d", custom_validator="mesh"
            ),
            DiffusionSamplingParams(prompt="test"),
            run_consistency_check=False,
            run_models_api_check=False,
            run_t2v_input_reference_check=False,
        )
        runner = common.DiffusionServerBase()
        with (
            patch.dict("os.environ", {"SGLANG_GEN_GT": "0"}),
            patch.object(common, "get_generate_fn", return_value=None),
            patch.object(runner, "run_and_collect", return_value=(None, b"")),
            patch.object(runner, "_validate_and_record"),
            patch.dict(mesh_utils.MESH_OUTPUT_PATHS, {}, clear=True),
        ):
            with self.assertRaisesRegex(
                pytest.fail.Exception, "Missing generated mesh"
            ):
                runner._test_diffusion_request(
                    case, SimpleNamespace(load_time_ms=100), 1
                )

    def test_rng_isolation_and_repeatability(self):
        original_state = np.random.get_state()
        self.addCleanup(np.random.set_state, original_state)
        for offset in (0.02, 0.25):
            self.save_pair(offset=offset)
            diagnostics = []
            for external_seed in (1, 99, 1):
                np.random.seed(external_seed)
                before = np.random.get_state()
                with self.assertLogs(mesh_utils.logger, level="INFO") as logs:
                    if offset == 0.02:
                        self.assertIsNone(self.validate(random_seed=42))
                    else:
                        with self.assertRaises(AssertionError) as failure:
                            self.validate(random_seed=42)
                        self.assertIn("normalized_cd=", str(failure.exception))
                diagnostics.append(logs.output)
                after = np.random.get_state()
                self.assertEqual(before[0], after[0])
                np.testing.assert_array_equal(before[1], after[1])
                self.assertEqual(before[2:], after[2:])
            self.assertEqual(diagnostics[0], diagnostics[1])
            self.assertEqual(diagnostics[0], diagnostics[2])

    def test_transformed_scene_and_planar_self_check(self):
        self.save_pair()
        self.assertIsNone(self.validate())
        scene = trimesh.Scene()
        scene.add_geometry(self.plane(), node_name="first")
        transform = trimesh.transformations.rotation_matrix(np.pi / 3, [0, 1, 0])
        transform[:3, 3] = [3, 2, 1]
        scene.add_geometry(self.plane(), node_name="second", transform=transform)
        transformed = self.plane()
        transformed.apply_transform(transform)
        world_mesh = trimesh.util.concatenate([self.plane(), transformed])
        self.generated = self.root / "generated.glb"
        world_mesh.export(self.reference)
        scene.export(self.generated)
        self.assertIsNone(self.validate())
        scene_reference = self.root / "reference.glb"
        scene.export(scene_reference)
        world_mesh.export(self.generated)
        with patch.object(
            mesh_utils, "_download_reference_mesh", return_value=scene_reference
        ):
            self.assertIsNone(self.validate())

    def test_threshold_is_inclusive(self):
        self.save_pair(offset=0.02)
        with self.assertLogs(mesh_utils.logger, level="INFO") as logs:
            self.validate()
        score = float(logs.output[-1].split("normalized_cd=")[1].split(",")[0])
        self.assertIsNone(self.validate(cd_threshold_ratio=score))
        with self.assertRaises(AssertionError):
            self.validate(cd_threshold_ratio=np.nextafter(score, 0))

    def test_nonfinite_distances_fail_explicitly(self):
        self.save_pair()
        # Finite, positive-area geometry whose squared separation overflows.
        self.write_ply(
            self.generated,
            [[0, 0, 1e200], [1, 0, 1e200], [1, 1, 1e200], [0, 1, 1e200]],
            [[0, 1, 2], [0, 2, 3]],
        )
        with self.assertRaisesRegex(AssertionError, "Non-finite generated/reference"):
            self.validate()

    def test_nonfinite_reference_diagonal(self):
        self.save_pair()
        self.write_ply(
            self.reference,
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1e200, 0, 0],
                [1e200, 1, 0],
                [1e200, 0, 1],
            ],
            [[0, 1, 2], [3, 4, 5]],
        )
        with self.assertRaisesRegex(AssertionError, "reference.*diagonal"):
            self.validate()

    def test_invalid_parameters_precede_io(self):
        for name, values in (
            ("num_sample_points", (0, -1, True, np.bool_(False), 1.5, "3", None)),
            ("random_seed", (-1, True, np.bool_(True), 1.5, "42", None)),
            (
                "cd_threshold_ratio",
                (-1, np.nan, np.inf, -np.inf, 10**400, "0.01", None),
            ),
        ):
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        self.validate(**{name: value})
        self.save_pair()
        self.assertIsNone(
            self.validate(num_sample_points=np.int64(4096), random_seed=np.uint64(42))
        )
        with self.assertRaises(AssertionError):
            self.validate(cd_threshold_ratio=0)

    def test_invalid_meshes_identify_input(self):
        # ASCII PLY preserves NaN/Inf and degenerate faces on disk; no loader mock.
        cases = {
            "empty": ([], []),
            "point_cloud": ([[0, 0, 0]], []),
            "invalid_face": ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 99]]),
            "zero_area": ([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]]),
            "collapsed": ([[0, 0, 0]] * 3, [[0, 1, 2]]),
            "nan": ([[0, 0, 0], [1, 0, 0], [0, float("nan"), 0]], [[0, 1, 2]]),
            "inf": ([[0, 0, 0], [1, 0, 0], [0, float("inf"), 0]], [[0, 1, 2]]),
            "overflow_area": ([[0, 0, 0], [1e200, 0, 0], [0, 1e200, 0]], [[0, 1, 2]]),
        }
        for label, path in (
            ("generated", self.generated),
            ("reference", self.reference),
        ):
            for case, (vertices, faces) in cases.items():
                with self.subTest(input=label, case=case):
                    self.save_pair()
                    self.write_ply(path, vertices, faces)
                    with self.assertRaisesRegex(AssertionError, label):
                        self.validate()
            self.save_pair()
            path.write_text("not a mesh")
            with self.assertRaisesRegex(AssertionError, label):
                self.validate()

    @staticmethod
    def write_ply(path, vertices, faces):
        header = (
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property double x\nproperty double y\nproperty double z\n"
            f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
        )
        path.write_text(
            header
            + "".join(" ".join(map(str, v)) + "\n" for v in vertices)
            + "".join("3 " + " ".join(map(str, f)) + "\n" for f in faces)
        )


if __name__ == "__main__":
    unittest.main()
