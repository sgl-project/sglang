import importlib.util
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[5]
_MODULE_PATH = _ROOT / "benchmark/kernels/attention/dsa_selective_kv_benchmark_utils.py"
_SPEC = importlib.util.spec_from_file_location(
    "dsa_selective_kv_benchmark_utils", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

BENCHMARK_RESULT_COLUMNS = _MODULE.BENCHMARK_RESULT_COLUMNS
SyntheticDSACase = _MODULE.SyntheticDSACase
build_benchmark_record = _MODULE.build_benchmark_record
make_synthetic_dsa_indices = _MODULE.make_synthetic_dsa_indices


def test_synthetic_case_is_deterministic_and_has_requested_shape():
    case = SyntheticDSACase(
        prefix_rows=4096,
        query_tokens=8,
        topk=128,
        unique_ratio=0.25,
        seed=17,
    )

    first = make_synthetic_dsa_indices(case)
    second = make_synthetic_dsa_indices(case)

    assert first.flat_topk_indices.shape == (8, 128)
    assert first.page_table_1_flattened.shape == (4096,)
    assert first.flat_topk_indices.dtype == torch.int32
    assert first.page_table_1_flattened.dtype == torch.int32
    torch.testing.assert_close(first.flat_topk_indices, second.flat_topk_indices)
    torch.testing.assert_close(
        first.page_table_1_flattened, second.page_table_1_flattened
    )


@pytest.mark.parametrize("unique_ratio", [0.125, 0.5, 1.0])
def test_synthetic_case_controls_logical_unique_ratio(unique_ratio):
    case = SyntheticDSACase(
        prefix_rows=8192,
        query_tokens=8,
        topk=128,
        unique_ratio=unique_ratio,
        seed=23,
    )

    inputs = make_synthetic_dsa_indices(case)

    valid_entries = case.query_tokens * case.topk
    expected_unique = round(valid_entries * unique_ratio)
    assert torch.unique(inputs.flat_topk_indices).numel() == expected_unique
    assert inputs.logical_unique_rows == expected_unique


def test_synthetic_case_can_model_cross_request_physical_aliasing():
    case = SyntheticDSACase(
        prefix_rows=1024,
        query_tokens=4,
        topk=128,
        unique_ratio=1.0,
        physical_unique_ratio=0.5,
        seed=29,
    )

    inputs = make_synthetic_dsa_indices(case)
    logical = torch.unique(inputs.flat_topk_indices)
    physical = torch.unique(inputs.page_table_1_flattened[logical.long()])

    assert logical.numel() == 512
    assert physical.numel() == 256
    assert inputs.physical_unique_rows == 256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prefix_rows", 0),
        ("query_tokens", 0),
        ("topk", 0),
        ("unique_ratio", 0.0),
        ("unique_ratio", 1.1),
        ("physical_unique_ratio", 0.0),
        ("physical_unique_ratio", 1.1),
    ],
)
def test_synthetic_case_rejects_invalid_geometry(field, value):
    kwargs = dict(
        prefix_rows=1024,
        query_tokens=4,
        topk=128,
        unique_ratio=0.5,
        physical_unique_ratio=1.0,
    )
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        SyntheticDSACase(**kwargs)


def test_benchmark_record_has_stable_json_csv_schema():
    case = SyntheticDSACase(
        prefix_rows=4096,
        query_tokens=8,
        topk=128,
        unique_ratio=0.25,
        seed=31,
    )
    inputs = make_synthetic_dsa_indices(case)
    timings = {
        "full_dequant_us": 100.0,
        "full_dequant_cached_us": 90.0,
        "selective_no_dedup_us": 80.0,
        "oracle_dedup_remap_us": 10.0,
        "dense_dedup_remap_us": 8.0,
        "oracle_selected_dequant_us": 20.0,
        "dense_selected_dequant_us": 22.0,
        "oracle_selective_total_us": 30.0,
        "dense_selective_total_us": 32.0,
    }

    record = build_benchmark_record(
        case=case,
        inputs=inputs,
        timings_us=timings,
        device_name="NVIDIA H20",
        git_sha="deadbeef",
        pool_rows=8192,
    )

    assert tuple(record) == BENCHMARK_RESULT_COLUMNS
    assert record["dense_speedup_vs_current"] == pytest.approx(100.0 / 32.0)
    assert record["dense_speedup_vs_cached_full"] == pytest.approx(90.0 / 32.0)
    assert record["logical_unique_ratio"] == pytest.approx(0.25)
    assert record["pool_rows"] == 8192
    assert record["selection_capacity_rows"] == 1024
    assert record["full_kv_traffic_bytes"] == 4096 * (656 + 576 * 2)
    assert record["active_selective_kv_traffic_bytes"] == 256 * (656 + 576 * 2)
    assert record["full_bf16_workspace_bytes"] == 4096 * 576 * 2
    assert record["fixed_selective_bf16_workspace_bytes"] == 1024 * 576 * 2
    assert record["dense_metadata_workspace_bytes"] > 8192 * (8 + 4)
    assert record["estimated_dense_metadata_traffic_bytes"] > 8192 * 2 * 8
    assert record["device_name"] == "NVIDIA H20"
    assert record["git_sha"] == "deadbeef"
