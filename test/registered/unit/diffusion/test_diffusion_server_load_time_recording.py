"""Unit tests for the load duration recorded by subclassed diffusion server tests.

``DiffusionServerBase._record_performance_result`` rejects a summary whose load
duration is missing, and only the server context carries that measurement. The
AMD and XPU nightlies override ``test_diffusion_generation`` rather than reusing
the base implementation, so any load time the base class alone forwards never
reaches their summaries: every case then fails with "Load duration missing or
invalid: None" while the server did measure one.
"""

import ast
import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    ScenarioConfig,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# needs no GPU, but base-a-test-cpu installs python[dev], which omits the
# diffusion extra this imports; the diffusion lane is the only one that has it
register_cuda_ci(est_time=20, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")

SERVER_LOAD_TIME_MS = 44977.73

# both trees that can subclass DiffusionServerBase
SEARCH_ROOTS = ("test/registered", "python/sglang/multimodal_gen/test")

# DiffusionServerBase itself: its call site defines the contract, not a drift from it
BASE_CLASS_FILE = "python/sglang/multimodal_gen/test/server/test_server_common.py"

# every diffusion test that overrides test_diffusion_generation
OVERRIDING_TESTS = (
    (
        "test/registered/amd/test_zimage_turbo.py",
        "TestZImageTurboAMD",
        "AMD_ZIMAGE_CASES",
    ),
    (
        "test/registered/xpu/test_xpu_zimage_turbo.py",
        "TestZImageTurboXPU",
        "XPU_ZIMAGE_CASES",
    ),
    (
        "test/registered/xpu/test_xpu_flux2_dev.py",
        "TestFlux2DevXPU",
        "XPU_FLUX2_CASES",
    ),
)


def _repo_root() -> Path:
    """Anchor on the trees themselves so relocating this file stays correct."""
    for parent in Path(__file__).resolve().parents:
        if all((parent / root).is_dir() for root in SEARCH_ROOTS):
            return parent
    raise RuntimeError(f"no ancestor of {__file__} contains {SEARCH_ROOTS}")


REPO_ROOT = _repo_root()


def _load_module(relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"load_time_guard_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calls_validate_and_record(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "_validate_and_record"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        for node in ast.walk(tree)
    )


def _perf_record() -> RequestPerfRecord:
    return RequestPerfRecord(
        request_id="load-time-guard",
        commit_hash="test",
        tag="load-time-guard",
        stages=[{"name": "DenoisingStage", "execution_time_ms": 100.0}],
        steps=[100.0],
        total_duration_ms=1000.0,
    )


class TestDiffusionServerLoadTimeRecording(CustomTestCase):
    def test_every_override_is_covered(self):
        """A new override must join OVERRIDING_TESTS instead of going unguarded."""
        found = {
            path.relative_to(REPO_ROOT).as_posix()
            for root in SEARCH_ROOTS
            for path in (REPO_ROOT / root).rglob("test_*.py")
            if _calls_validate_and_record(path)
        }
        self.assertEqual(found - {BASE_CLASS_FILE}, {e[0] for e in OVERRIDING_TESTS})
        self.assertIn(BASE_CLASS_FILE, found, "base class call site went missing")

    def test_overriding_tests_record_the_measured_load_time(self):
        """Each override must reach _record_performance_result with the server's load time."""
        # thresholds are not the subject here, so keep them off the runner's speed
        scenario = ScenarioConfig({}, {}, 1e6, 1e6, 1e6, expected_load_ms=1e6)

        for relative_path, class_name, cases_name in OVERRIDING_TESTS:
            with self.subTest(test_file=relative_path):
                module = _load_module(relative_path)
                case = replace(getattr(module, cases_name)[0], run_perf_check=False)
                server = getattr(module, class_name)()
                server._perf_results = []
                server.run_and_collect = lambda *args, **kwargs: (_perf_record(), b"")
                server._test_v1_models_endpoint = lambda *args, **kwargs: None
                context = SimpleNamespace(load_time_ms=SERVER_LOAD_TIME_MS)

                with (
                    patch.dict(BASELINE_CONFIG.scenarios, {case.id: scenario}),
                    patch.object(module, "_compute_clip_score", return_value=None),
                    patch.object(module, "_save_image_and_write_summary"),
                ):
                    server.test_diffusion_generation(case, context)

                self.assertEqual(len(server._perf_results), 1)
                result = server._perf_results[0]
                self.assertEqual(result["load_time_ms"], SERVER_LOAD_TIME_MS)
                self.assertEqual(
                    result["load_inclusive_e2e_ms"],
                    SERVER_LOAD_TIME_MS + result["e2e_ms"],
                )


if __name__ == "__main__":
    unittest.main()
