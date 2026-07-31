import importlib.util
import json
import tomllib
from pathlib import Path

from sglang_simulator.dataset.autobench import sample_autobench_requests

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"


def test_example_json_files_parse():
    for path in EXAMPLES.rglob("*.json"):
        with path.open(encoding="utf-8") as stream:
            json.load(stream)


def test_replay_example_trace_contract():
    rows = sample_autobench_requests(
        dataset_path=str(EXAMPLES / "replay/trace.jsonl"),
        num_requests=3,
        tokenizer=None,
    )
    assert len(rows) == 3
    assert [row.timestamp for row in rows] == [0, 100, 200]
    assert all(row.prompt_len == len(row.prompt) for row in rows)
    assert rows[1].prompt[:3] == rows[0].prompt[:3]


def test_replay_example_uses_a_local_database():
    config_path = EXAMPLES / "replay/simulator.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    database = config_path.parent / config["predictor"]["database_path"]
    assert database.is_file()


def test_example_script_can_be_imported():
    path = EXAMPLES / "run_inprocess.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_package_declares_runtime_dependencies():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert set(metadata["project"]["dependencies"]) == {
        "numpy",
        "scikit-learn",
        "joblib",
    }
