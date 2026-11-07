from asc_bench.config import load_config
from asc_bench.expand import expand_cells

CONFIG_TEXT = """
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
  seed: 42
"""


def make_cfg(tmp_path, text=CONFIG_TEXT):
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_cell_count_is_full_product(tmp_path):
    cells = expand_cells(make_cfg(tmp_path))
    assert len(cells) == 2 * 2 * 2  # server axes x workload axes x repeats


def test_order_and_ids_are_deterministic(tmp_path):
    cells_a = expand_cells(make_cfg(tmp_path))
    cells_b = expand_cells(make_cfg(tmp_path))
    assert [c.cell_id for c in cells_a] == [c.cell_id for c in cells_b]
    assert len({c.cell_id for c in cells_a}) == len(cells_a)
    assert cells_a[0].cell_id == "qwen3-8b__s00__w00__r0"
    assert cells_a[-1].cell_id == "qwen3-8b__s01__w01__r1"


def test_num_prompts_is_mult_times_concurrency(tmp_path):
    cells = expand_cells(make_cfg(tmp_path))
    by_id = {c.cell_id: c for c in cells}
    assert by_id["qwen3-8b__s00__w00__r0"].num_prompts == 8 * 1
    assert by_id["qwen3-8b__s00__w01__r0"].num_prompts == 8 * 8


def test_axes_override_base_args(tmp_path):
    cells = expand_cells(make_cfg(tmp_path))
    mem_values = {c.server_args["--mem-fraction-static"] for c in cells}
    assert mem_values == {0.8, 0.85}
    assert all(c.server_args["--device"] == "npu" for c in cells)


def test_cell_hash_excludes_sla_and_run(tmp_path):
    cells = expand_cells(make_cfg(tmp_path))
    changed = CONFIG_TEXT.replace("p99_ttft_ms: 2000", "p99_ttft_ms: 9999").replace(
        "repeats: 2", "repeats: 2 "
    )
    cells_other = expand_cells(make_cfg(tmp_path, changed))
    assert cells[0].cell_hash == cells_other[0].cell_hash
    mutated = expand_cells(
        make_cfg(
            tmp_path,
            CONFIG_TEXT.replace("--random-input-len: 1024", "--random-input-len: 2048"),
        )
    )
    assert cells[0].cell_hash != mutated[0].cell_hash


def test_repeats_share_cell_hash_and_differ_in_id(tmp_path):
    cells = expand_cells(make_cfg(tmp_path))
    r0 = next(c for c in cells if c.cell_id.endswith("r0"))
    r1 = next(c for c in cells if c.cell_id.endswith("r1"))
    assert r0.cell_hash == r1.cell_hash
    assert r0.cell_id != r1.cell_id
