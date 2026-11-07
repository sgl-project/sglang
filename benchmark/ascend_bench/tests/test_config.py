import pytest
from asc_bench.config import ConfigError, load_config

VALID = """
name: mini
model:
  path: Qwen/Qwen3-8B
server:
  args:
    --device: npu
    --tp-size: 2
  axes:
    --mem-fraction-static: [0.8, 0.85]
workload:
  dataset_name: random
  args:
    --random-input-len: 1024
  axes:
    --max-concurrency: [1, 8]
  num_prompts_mult: 8
sla:
  thresholds:
    p99_ttft_ms: 2000
run:
  repeats: 2
"""


def write(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_config_loads(tmp_path):
    cfg = load_config(write(tmp_path, VALID))
    assert cfg.name == "mini"
    assert cfg.model.path == "Qwen/Qwen3-8B"
    assert cfg.server.axes["--mem-fraction-static"] == [0.8, 0.85]
    assert cfg.workload.num_prompts_mult == 8
    assert cfg.run.repeats == 2


def test_errors_are_aggregated(tmp_path):
    bad = VALID.replace("name: mini", "name: ''").replace(
        "    p99_ttft_ms: 2000", "    bad_key: 2000"
    )
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, bad))
    message = str(excinfo.value)
    assert "name must be a non-empty string" in message
    assert "bad_key" in message
    assert "2 config error" in message


def test_flags_must_start_with_double_dash(tmp_path):
    bad = VALID.replace("    --device: npu", "    device: npu")
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, bad))
    assert "must start with '--'" in str(excinfo.value)


def test_args_and_axes_overlap_rejected(tmp_path):
    bad = VALID.replace(
        "  axes:\n    --mem-fraction-static: [0.8, 0.85]",
        "  axes:\n    --device: [npu, cpu]",
    )
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, bad))
    assert "both args and axes" in str(excinfo.value)


def test_cuda_graph_bs_limit_enforced(tmp_path):
    bad = VALID.replace(
        "    --tp-size: 2",
        "    --tp-size: 2\n    --cuda-graph-bs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]",
    )
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, bad))
    assert "cuda-graph-bs" in str(excinfo.value)
    assert "507000" in str(excinfo.value)


def test_unknown_dataset_rejected(tmp_path):
    bad = VALID.replace("dataset_name: random", "dataset_name: nope")
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, bad))
    assert "dataset_name" in str(excinfo.value)


def test_mult_requires_concurrency_axis(tmp_path):
    bad = VALID.replace("    --max-concurrency: [1, 8]", "    --request-rate: [1]")
    with pytest.raises(ConfigError) as excinfo:
        load_config(write(tmp_path, bad))
    assert "num_prompts_mult requires" in str(excinfo.value)


def test_top_level_must_be_mapping(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, "- a\n- b\n"))
