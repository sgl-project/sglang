import importlib.util
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
if "transformers" not in sys.modules:
    transformers = types.ModuleType("transformers")
    transformers.PreTrainedTokenizerBase = object
    transformers.PreTrainedTokenizer = object
    sys.modules["transformers"] = transformers
SCRIPT = Path(__file__).parents[1] / "scripts" / "run_infercast_open_loop.py"
SPEC = importlib.util.spec_from_file_location("run_infercast_open_loop", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_trace_preserves_timestamp_and_shared_prefix(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "timestamp_ms": 12.5,
                "input_length": 5,
                "output_length": 3,
                "hash_ids": [7, 8],
                "block_size": 2,
                "case_id": "prefix-hit",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = MODULE.load_trace(trace)

    assert len(dataset) == 1
    assert dataset[0].token_ids == [1007, 1007, 1008, 1008, 120000]
    assert dataset[0].custom_params == {
        "created_time": 0.0125,
        "trace": {
            "hash_ids": [7, 8],
            "block_size": 2,
            "case_id": "prefix-hit",
        },
    }


def test_file_identity_and_phase_snapshot_are_immutable(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")
    identity = MODULE.file_identity(trace)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text('{"value": 1}\n', encoding="utf-8")
    artifacts = MODULE.snapshot_outputs(output_dir, "warmup")
    (output_dir / "metrics.json").write_text('{"value": 2}\n', encoding="utf-8")

    assert identity["path"] == str(trace)
    assert len(identity["sha256"]) == 64
    assert Path(artifacts["metrics.json"]).read_text(encoding="utf-8") == (
        '{"value": 1}\n'
    )
