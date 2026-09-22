"""Tests for slash-command test selection: declarative groups and `--changed`."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import yaml

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HANDLER_PATH = _REPO_ROOT / "scripts/ci/utils/slash_command_handler.py"


def _load_handler():
    github = ModuleType("github")
    github.Auth = object()
    github.Github = object()
    spec = importlib.util.spec_from_file_location(
        "slash_command_handler", _HANDLER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"github": github}):
        spec.loader.exec_module(module)
    return module


class TestMultimodalLaunchers(CustomTestCase):
    def test_component_accuracy_uses_two_workers_and_preserves_selector(self):
        handler = _load_handler()
        spec = (
            f"{handler.MULTIMODAL_TEST_DIR}/single_test_file/component_accuracy/"
            "test_component_accuracy_2_gpu.py::TestComponentAccuracy2GPU::"
            "test_encoder_accuracy[ltx_2_two_stage_t2v]"
        )
        self.assertEqual(
            handler.build_pytest_command(spec),
            [
                "python3",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                "-m",
                "pytest",
                spec,
                "-x",
            ],
        )
        with patch.object(
            handler,
            "resolve_test_file",
            return_value=(spec.split("::", 1)[0], True, None),
        ):
            resolved = handler._resolve_test_spec(spec)[0]
        self.assertEqual(resolved["runs_on"], "2-gpu-h100")
        self.assertEqual(resolved["test_command"], spec)

    def test_qwen_encoder_gets_two_gpu_runner_and_workers(self):
        handler = _load_handler()
        path = f"{handler.MULTIMODAL_TEST_DIR}/unit/test_qwen_image21_distributed.py"
        self.assertEqual(handler.detect_multimodal_suite(path), ("2-gpu-h100", None))
        self.assertEqual(handler.torchrun_processes(path), 2)

    def test_single_gpu_and_self_spawning_tests_keep_plain_pytest(self):
        handler = _load_handler()
        for relative_path in (
            "single_test_file/component_accuracy/test_component_accuracy_1_gpu.py",
            "server/test_server_2_gpu.py",
            "single_test_file/test_encoder_fold_srt_2_gpu.py",
            "unit/test_component_accuracy_parallel_runtime.py",
        ):
            with self.subTest(path=relative_path):
                spec = f"{handler.MULTIMODAL_TEST_DIR}/{relative_path}"
                self.assertEqual(
                    handler.build_pytest_command(spec),
                    ["python3", "-m", "pytest", spec, "-x"],
                )

    def test_workflow_launches_each_spec_and_stops_on_failure(self):
        workflow = yaml.safe_load(
            (_REPO_ROOT / ".github/workflows/rerun-test.yml").read_text()
        )
        step = next(
            step
            for step in workflow["jobs"]["rerun-test-multimodal-gen"]["steps"]
            if step.get("name") == "Run test"
        )
        self.assertEqual(step["env"]["TEST_COMMAND"], "${{ inputs.test_command }}")
        prefix = "python/sglang/multimodal_gen/test/"
        distributed_spec = (
            prefix + "single_test_file/component_accuracy/"
            "test_component_accuracy_2_gpu.py::TestComponentAccuracy2GPU::"
            "test_encoder_accuracy[ltx_2_two_stage_t2v]"
        )
        server_spec = prefix + "server/test_server_2_gpu.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Replace only the launched pytest/torchrun modules; execute the
            # real workflow shell and launcher without GPUs or model weights.
            for package in (root / "torch", root / "torch/distributed"):
                package.mkdir(exist_ok=True)
                (package / "__init__.py").touch()
            recorder = (
                "import json, os, sys\n"
                "with open(os.environ['LAUNCH_LOG'], 'a') as f:\n"
                "    f.write(json.dumps(sys.argv) + '\\n')\n"
                "sys.exit(int(os.environ['LAUNCH_EXIT_CODE']))\n"
            )
            (root / "torch/distributed/run.py").write_text(recorder)
            (root / "pytest.py").write_text(recorder)
            (root / "bin").mkdir()
            (root / "bin/python3").symlink_to(sys.executable)
            log = root / "launches.jsonl"
            for exit_code in (0, 7):
                with self.subTest(exit_code=exit_code):
                    log.write_text("")
                    result = subprocess.run(
                        ["bash", "-e", "-c", step["run"]],
                        cwd=_REPO_ROOT,
                        env=dict(
                            os.environ,
                            PATH=f"{root / 'bin'}:{os.environ['PATH']}",
                            PYTHONPATH=str(root),
                            TEST_COMMAND=f"{distributed_spec}\n{server_spec}",
                            LAUNCH_LOG=str(log),
                            LAUNCH_EXIT_CODE=str(exit_code),
                        ),
                        capture_output=True,
                        text=True,
                    )
                    calls = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertEqual(
                        result.returncode, 1 if exit_code else 0, result.stderr
                    )
                    self.assertEqual(len(calls), 1 if exit_code else 2)
                    self.assertEqual(
                        calls[0][1:],
                        [
                            "--standalone",
                            "--nproc_per_node=2",
                            "-m",
                            "pytest",
                            distributed_spec,
                            "-x",
                        ],
                    )
                    if not exit_code:
                        self.assertEqual(calls[1][1:], [server_spec, "-x"])


class TestConfiguredTestGroups(CustomTestCase):
    def test_additional_group_requires_only_manifest_data(self):
        handler = _load_handler()
        previous_cwd = os.getcwd()
        try:
            os.chdir(_REPO_ROOT)
            with tempfile.TemporaryDirectory() as temp_dir:
                manifest = Path(temp_dir) / "groups.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "mixed": [
                                "registered/rust/test_run_rust_tests.py",
                                "registered/core/test_srt_endpoint.py",
                            ]
                        }
                    )
                )
                with patch.object(handler, "TEST_GROUPS_FILE_PATH", str(manifest)):
                    specs, error = handler.resolve_test_group_specs("mixed")

            self.assertIsNone(error)
            self.assertEqual(
                specs,
                [
                    "registered/rust/test_run_rust_tests.py",
                    "registered/core/test_srt_endpoint.py",
                ],
            )
        finally:
            os.chdir(previous_cwd)

    def test_rust_server_group(self):
        handler = _load_handler()
        previous_cwd = os.getcwd()
        try:
            os.chdir(_REPO_ROOT)
            specs, error = handler.resolve_test_group_specs("rust-server")
            self.assertIsNone(error)
            self.assertEqual(
                specs,
                [
                    "registered/rust/test_run_rust_tests.py",
                    "registered/core/test_srt_endpoint.py",
                    "registered/vlm/test_rust_native_mm_e2e.py",
                    "registered/vlm/test_rust_native_mm_mmmu.py",
                ],
            )

            resolved = [
                item
                for test_spec in specs
                for item in handler._resolve_test_spec(test_spec)
            ]
            self.assertTrue(all(item["error"] is None for item in resolved), resolved)
            self.assertEqual(
                [item["mode"] for item in resolved],
                ["cpu", "cuda", "cuda", "cuda"],
            )
        finally:
            os.chdir(previous_cwd)


class TestChangedTestFiles(CustomTestCase):
    def test_only_dispatchable_changed_test_files_are_selected(self):
        """`--changed` feeds the dispatcher directly, so every path it returns must
        be runnable: no deleted tests, helpers, source, `manual/` files, or
        multimodal `test_*.py` that collect nothing. A pure move is dropped only
        when it leaves dispatch alone."""
        handler = _load_handler()
        mm = handler.MULTIMODAL_TEST_DIR
        pr = SimpleNamespace(
            get_files=lambda: [
                SimpleNamespace(
                    filename=name,
                    status=status,
                    changes=changes,
                    previous_filename=previous,
                )
                for name, status, changes, previous in [
                    (
                        "test/registered/unit/mem_cache/test_radix_cache_unit.py",
                        "modified",
                        4,
                        None,
                    ),
                    ("test/registered/core/test_srt_endpoint.py", "removed", 12, None),
                    ("test/registered/unit/mem_cache/helpers.py", "modified", 2, None),
                    ("python/sglang/srt/mem_cache/radix_cache.py", "modified", 7, None),
                    ("test/manual/test_not_registered.py", "added", 20, None),
                    (
                        "test/registered/spec/test_moved_untouched.py",
                        "renamed",
                        0,
                        "test/registered/core/test_moved_untouched.py",
                    ),
                    (
                        "test/registered/spec/test_moved_into_ci.py",
                        "renamed",
                        0,
                        "test/manual/test_moved_into_ci.py",
                    ),
                    (
                        f"{mm}/2_gpu/test_moved_pool.py",
                        "renamed",
                        0,
                        f"{mm}/unit/test_moved_pool.py",
                    ),
                    (
                        "test/registered/spec/test_moved_and_edited.py",
                        "renamed",
                        9,
                        "test/registered/core/test_moved_and_edited.py",
                    ),
                    (f"{mm}/server/test_server_common.py", "modified", 3, None),
                    (f"{mm}/server/test_server_utils.py", "modified", 3, None),
                    (f"{mm}/unit/manual/test_fp4_linear.py", "modified", 3, None),
                    # Absent from the checkout, as a fork-added file is; kept so
                    # resolve_test_file() reports `File not found` for it.
                    (
                        f"{mm}/server/test_server_added_by_fork.py",
                        "added",
                        30,
                        None,
                    ),
                ]
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / mm
            (root / "server").mkdir(parents=True)
            (root / "unit" / "manual").mkdir(parents=True)
            (root / "2_gpu").mkdir(parents=True)
            (root / "server" / "test_server_common.py").write_text(
                "def test_diffusion_generation():\n    pass\n"
            )
            (root / "server" / "test_server_utils.py").write_text(
                "def build_server():\n    return None\n"
            )
            (root / "unit" / "manual" / "test_fp4_linear.py").write_text(
                "class TestFp4Linear:\n    def test_it(self):\n        pass\n"
            )
            (root / "2_gpu" / "test_moved_pool.py").write_text(
                "def test_two_gpu():\n    pass\n"
            )
            previous_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                self.assertEqual(
                    handler.changed_test_files(pr),
                    [
                        f"{mm}/2_gpu/test_moved_pool.py",
                        f"{mm}/server/test_server_added_by_fork.py",
                        f"{mm}/server/test_server_common.py",
                        "test/registered/spec/test_moved_and_edited.py",
                        "test/registered/spec/test_moved_into_ci.py",
                        "test/registered/unit/mem_cache/test_radix_cache_unit.py",
                    ],
                )
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
