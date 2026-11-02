"""Optional predictor dependencies must not prevent Replay startup."""

import subprocess
import sys
import textwrap

import pytest

BLOCK_AICONFIGURATOR = """
import importlib.abc
import sys

class BlockAIConfigurator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "aiconfigurator" or fullname.startswith("aiconfigurator."):
            raise ModuleNotFoundError("No module named 'aiconfigurator'", name="aiconfigurator")

sys.meta_path.insert(0, BlockAIConfigurator())
"""


def run_isolated(code):
    result = subprocess.run(
        [sys.executable, "-c", BLOCK_AICONFIGURATOR + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_replay_without_aiconfigurator(tmp_path):
    table = tmp_path / "replay.json"
    table.write_text('{"[[1, 0]]": 0.125}')
    run_isolated(
        f"""
        from sglang_simulator.simulation import utils
        from sglang_simulator.simulation.manager.config import ConfigManager
        from sglang_simulator.time_predictor import (
            InferTimePredictor, ReplayTimePredictor, ScheduleBatch, ScheduleRequest,
        )
        ConfigManager._raw_config = {{
            "predictor": {{"name": "replay", "database_path": {str(table)!r}}}
        }}
        predictor = ConfigManager.get_inference_time_predictor(None, None, None)
        assert isinstance(predictor, ReplayTimePredictor)
        assert isinstance(predictor, InferTimePredictor)
        batch = ScheduleBatch(reqs=[ScheduleRequest(extend_length=1)])
        assert predictor.predict_infer_time(batch) == 0.125
        assert "sglang_simulator.time_predictor.aiconfigurator" not in sys.modules
        assert "aiconfigurator" not in sys.modules
        """
    )


@pytest.mark.parametrize(
    "operation",
    [
        "from sglang_simulator.time_predictor import AIConfiguratorTimePredictor",
        "from sglang_simulator.time_predictor.aiconfigurator import get_perf_model",
        "ConfigManager.get_inference_time_predictor(None, None, None)",
        "profile_device_available_bytes(None, SimpleNamespace(hbm_capacity_gb=80), None)",
    ],
)
def test_aiconfigurator_missing_dependency_error(operation):
    run_isolated(
        f"""
        from types import SimpleNamespace
        from sglang_simulator.simulation.manager.config import ConfigManager
        from sglang_simulator.simulation.utils import profile_device_available_bytes
        ConfigManager._raw_config = {{"predictor": {{"name": "aiconfigurator"}}}}
        try:
            {operation}
        except ImportError as exc:
            assert "sglang-simulator[aic]" in str(exc)
            assert "--max-total-tokens" in str(exc)
            assert isinstance(exc.__cause__, ModuleNotFoundError)
            assert exc.__cause__.name == "aiconfigurator"
        else:
            raise AssertionError("Expected a helpful optional dependency error")
        """
    )


def test_unrelated_missing_dependency_is_not_masked():
    run_isolated(
        """
        class BrokenAIConfigurator(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "aiconfigurator":
                    raise ModuleNotFoundError("No module named 'other_dependency'", name="other_dependency")
        sys.meta_path.insert(0, BrokenAIConfigurator())
        try:
            from sglang_simulator.time_predictor import AIConfiguratorTimePredictor
        except ModuleNotFoundError as exc:
            assert exc.name == "other_dependency"
        else:
            raise AssertionError("Expected the original dependency error")
        """
    )
