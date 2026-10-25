"""Replay startup must not import an unselected predictor's dependencies."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLOCK_OPTIONAL = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'aiconfigurator', 'joblib', 'sklearn'}:
            raise ModuleNotFoundError('blocked unselected dependency: ' + fullname, name=fullname)

sys.meta_path.insert(0, BlockOptional())
"""


def run_isolated(script, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")])
    )
    completed = subprocess.run(
        [sys.executable, "-c", BLOCK_OPTIONAL + script, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        pytest.fail(completed.stdout + completed.stderr, pytrace=False)


@pytest.mark.parametrize("strategy", ["default", "knn"])
def test_replay_configuration_and_hooks_import_without_optional_predictors(strategy):
    run_isolated(
        """
import json
from pathlib import Path
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.sglang.hook_bootstrap import install_simulator_hooks
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.time_predictor import ReplayTimePredictor, ScheduleBatch, ScheduleRequest

table = Path(sys.argv[1])
predictor_config = {'name': 'replay', 'database_path': str(table)}
if sys.argv[2] != 'default':
    predictor_config.update(miss_strategy=sys.argv[2], miss_knn_k=1)
ConfigManager._raw_config = {'predictor': predictor_config}
predictor = ConfigManager.get_inference_time_predictor(None, None, SchedulerConfig())
assert isinstance(predictor, ReplayTimePredictor)
entries = json.loads(table.read_text())
for key, value in entries.items():
    batch = ScheduleBatch([ScheduleRequest(*pair) for pair in json.loads(key)])
    assert predictor.predict_infer_time(batch) == value
missing = ScheduleBatch([ScheduleRequest(100, 0)])
value = predictor.predict_infer_time(missing)
if sys.argv[2] == 'default':
    assert value == 0
else:
    assert value in entries.values()
assert predictor.get_metrics()['replay_miss_steps'] == 1
assert not any(name.split('.')[0] in {'aiconfigurator', 'joblib', 'sklearn'} for name in sys.modules)
""",
        str(ROOT / "examples/assets/replay_table.json"),
        strategy,
    )


def test_public_optional_predictors_load_only_when_requested():
    run_isolated(
        """
import sglang_simulator.time_predictor as predictors

assert all(isinstance(name, str) for name in predictors.__all__)
for name, dependency in [('AIConfiguratorTimePredictor', 'aiconfigurator'), ('MLTimePredictor', 'joblib')]:
    try:
        getattr(predictors, name)
    except ModuleNotFoundError as error:
        assert error.name == dependency
    else:
        raise AssertionError('requested predictor must require its dependency')
try:
    predictors.unknown_predictor
except AttributeError:
    pass
else:
    raise AssertionError('unknown public attribute must fail')
"""
    )


def test_automatic_memory_profile_still_requires_aic_when_called():
    run_isolated(
        """
from types import SimpleNamespace
from sglang_simulator.simulation.utils import profile_device_available_bytes

try:
    profile_device_available_bytes(None, SimpleNamespace(name='missing-capacity', hbm_capacity_gb=None), None)
except ValueError as error:
    assert 'hbm_capacity_gb' in str(error)
else:
    raise AssertionError('missing declared capacity must still fail')
try:
    profile_device_available_bytes(None, SimpleNamespace(hbm_capacity_gb=80), None)
except ModuleNotFoundError as error:
    assert error.name == 'aiconfigurator'
else:
    raise AssertionError('automatic memory sizing must not invent a fallback')
"""
    )
