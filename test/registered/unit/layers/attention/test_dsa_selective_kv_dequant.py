import importlib.util
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# Keep this CPU contract test independent of SGLang's package-level CUDA/Triton
# imports.  That lets contributors validate the index semantics before renting
# a GPU machine or installing the complete runtime environment.
_ROOT = Path(__file__).resolve().parents[5]
_MODULE_PATH = _ROOT / "python/sglang/srt/layers/attention/dsa/selective_kv_dequant.py"
_SPEC = importlib.util.spec_from_file_location("selective_kv_dequant", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_ENV_PATH = _ROOT / "python/sglang/srt/environ.py"
_ENV_SPEC = importlib.util.spec_from_file_location("sglang_environ", _ENV_PATH)
assert _ENV_SPEC is not None and _ENV_SPEC.loader is not None
_ENV_MODULE = importlib.util.module_from_spec(_ENV_SPEC)
_ENV_SPEC.loader.exec_module(_ENV_MODULE)

KV_BYTES_PER_ROW = _MODULE.KV_BYTES_PER_ROW
KV_DEQUANTIZED_BYTES_PER_ROW = _MODULE.KV_DEQUANTIZED_BYTES_PER_ROW
build_selective_kv_remap_reference = _MODULE.build_selective_kv_remap_reference
estimate_selective_kv_traffic = _MODULE.estimate_selective_kv_traffic
should_use_selective_kv_dequant = _MODULE.should_use_selective_kv_dequant


def should_use_dense_epoch_kv_dequant(*args, **kwargs):
    implementation = getattr(_MODULE, "should_use_dense_epoch_kv_dequant", None)
    assert (
        implementation is not None
    ), "should_use_dense_epoch_kv_dequant is not implemented"
    return implementation(*args, **kwargs)


def build_selective_kv_no_dedup(*args, **kwargs):
    implementation = getattr(_MODULE, "build_selective_kv_no_dedup", None)
    assert implementation is not None, "build_selective_kv_no_dedup is not implemented"
    return implementation(*args, **kwargs)


def maybe_build_selective_kv_no_dedup(*args, **kwargs):
    implementation = getattr(_MODULE, "maybe_build_selective_kv_no_dedup", None)
    assert (
        implementation is not None
    ), "maybe_build_selective_kv_no_dedup is not implemented"
    return implementation(*args, **kwargs)


def dequantize_dsa_prefix_kv_selective(*args, **kwargs):
    implementation = getattr(_MODULE, "dequantize_dsa_prefix_kv_selective", None)
    assert (
        implementation is not None
    ), "dequantize_dsa_prefix_kv_selective is not implemented"
    return implementation(*args, **kwargs)


def resolve_selective_kv_mode(*args, **kwargs):
    implementation = getattr(_MODULE, "resolve_selective_kv_mode", None)
    assert implementation is not None, "resolve_selective_kv_mode is not implemented"
    return implementation(*args, **kwargs)


def _i32(values):
    return torch.tensor(values, dtype=torch.int32)


def test_reference_remap_preserves_first_physical_occurrence_order():
    page_table = _i32([40, 41, 42, 43])
    topk = _i32([[2, 0], [3, 1]])

    result = build_selective_kv_remap_reference(page_table, topk)

    torch.testing.assert_close(result.physical_slots, _i32([42, 40, 43, 41]))
    torch.testing.assert_close(result.remapped_topk, _i32([[0, 1], [2, 3]]))
    assert result.num_unique == 4


def test_reference_remap_deduplicates_within_and_across_queries():
    page_table = _i32([10, 11, 12, 13])
    topk = _i32([[1, 1, 3], [3, 1, 3]])

    result = build_selective_kv_remap_reference(page_table, topk)

    torch.testing.assert_close(result.physical_slots, _i32([11, 13]))
    torch.testing.assert_close(result.remapped_topk, _i32([[0, 0, 1], [1, 0, 1]]))
    assert result.num_valid == 6
    assert result.num_unique == 2


def test_reference_remap_deduplicates_physical_aliases_across_requests():
    # Flat logical rows 0 and 3 belong to different requests, but radix-prefix
    # sharing makes them refer to the same physical KV-cache slot.
    page_table = _i32([70, 71, 80, 70, 81])
    topk = _i32([[0, 1], [3, 4]])

    result = build_selective_kv_remap_reference(page_table, topk)

    torch.testing.assert_close(result.physical_slots, _i32([70, 71, 81]))
    torch.testing.assert_close(result.remapped_topk, _i32([[0, 1], [0, 2]]))


def test_no_dedup_remap_keeps_one_physical_row_per_valid_occurrence():
    page_table = _i32([70, 71, 80, 70, 81])
    topk = _i32([[0, 1], [3, 4]])

    result = build_selective_kv_no_dedup(page_table, topk)

    # Physical slot 70 is deliberately repeated.  This prototype measures the
    # selective upper bound without paying any dedup metadata cost.
    torch.testing.assert_close(result.physical_slots, _i32([70, 71, 70, 81]))
    torch.testing.assert_close(result.remapped_topk, _i32([[0, 1], [2, 3]]))


def test_no_dedup_remap_preserves_padding_without_compacting_shape():
    page_table = _i32([20, 21, 22])
    topk = _i32([[2, -1], [0, -1]])

    result = build_selective_kv_no_dedup(page_table, topk)

    # Padding occurrences use physical row 0 as a safe, ignored landing row;
    # FlashMLA still receives -1 for those positions.
    torch.testing.assert_close(result.physical_slots, _i32([22, 20, 20, 20]))
    torch.testing.assert_close(result.remapped_topk, _i32([[0, -1], [2, -1]]))


def test_no_dedup_remap_handles_all_padding_without_a_page_table():
    result = build_selective_kv_no_dedup(_i32([]), _i32([[-1, -1]]))

    assert result.physical_slots.shape == (0,)
    torch.testing.assert_close(result.remapped_topk, _i32([[-1, -1]]))


def test_no_dedup_probe_planner_requires_gate_and_conservative_profitability():
    page_table = torch.arange(4096, dtype=torch.int32)
    profitable_topk = torch.arange(128, dtype=torch.int32).reshape(1, 128)

    assert (
        maybe_build_selective_kv_no_dedup(page_table, profitable_topk, enabled=False)
        is None
    )
    selected = maybe_build_selective_kv_no_dedup(
        page_table, profitable_topk, enabled=True
    )
    assert selected is not None
    torch.testing.assert_close(selected.physical_slots, profitable_topk.reshape(-1))

    # Q*K reaches the full prefix, so the probe must retain the current path
    # instead of betting on overlap it does not deduplicate yet.
    unprofitable_topk = torch.arange(4096, dtype=torch.int32).reshape(32, 128)
    assert (
        maybe_build_selective_kv_no_dedup(page_table, unprofitable_topk, enabled=True)
        is None
    )


def test_selective_workspace_reuses_addresses_and_grows_by_capacity_bucket():
    workspace_type = getattr(_MODULE, "SelectiveKVWorkspace", None)
    assert workspace_type is not None, "SelectiveKVWorkspace is not implemented"
    workspace = workspace_type(torch.device("cpu"))

    first = workspace.get_bf16(7)
    first_ptr = first.data_ptr()
    smaller = workspace.get_bf16(3)

    assert first.shape == (7, 1, 576)
    assert first.dtype == torch.bfloat16
    assert smaller.shape == (3, 1, 576)
    assert smaller.data_ptr() == first_ptr
    assert workspace.bf16_capacity == 8

    grown = workspace.get_bf16(9)
    assert grown.shape == (9, 1, 576)
    assert workspace.bf16_capacity == 16


def test_selective_workspace_reuses_fixed_shape_occurrence_metadata():
    workspace_type = getattr(_MODULE, "SelectiveKVWorkspace", None)
    assert workspace_type is not None, "SelectiveKVWorkspace is not implemented"
    workspace = workspace_type(torch.device("cpu"))

    physical, remapped = workspace.get_occurrence_metadata(5)
    physical_ptr = physical.data_ptr()
    remapped_ptr = remapped.data_ptr()
    physical_small, remapped_small = workspace.get_occurrence_metadata(2)

    assert physical.dtype == torch.int32
    assert remapped.dtype == torch.int32
    assert physical.shape == remapped.shape == (5,)
    assert physical_small.data_ptr() == physical_ptr
    assert remapped_small.data_ptr() == remapped_ptr
    assert workspace.occurrence_capacity == 8


def test_selective_workspace_allocates_generation_safe_dense_dedup_state():
    workspace_type = getattr(_MODULE, "SelectiveKVWorkspace", None)
    assert workspace_type is not None, "SelectiveKVWorkspace is not implemented"
    workspace = workspace_type(torch.device("cpu"))

    buffers = workspace.get_dense_dedup_buffers(
        num_pool_rows=9,
        selection_capacity=5,
        num_occurrences=7,
    )

    assert buffers.slot_epoch.shape == (16,)
    assert buffers.slot_to_compact.shape == (16,)
    assert buffers.selected_physical_slots.shape == (8,)
    assert buffers.remapped_topk.shape == (8,)
    assert buffers.block_offsets.shape == (1,)
    assert buffers.epoch.shape == (1,)
    assert buffers.num_unique.shape == (1,)
    # A 64-bit generation prevents a long-lived CUDA Graph service from ever
    # aliasing a stale int32 epoch after wraparound.
    assert buffers.slot_epoch.dtype == torch.int64
    assert buffers.slot_to_compact.dtype == torch.int32
    assert buffers.selected_physical_slots.dtype == torch.int32
    assert buffers.remapped_topk.dtype == torch.int32
    assert buffers.block_offsets.dtype == torch.int32
    assert buffers.epoch.dtype == torch.int64
    assert buffers.epoch.item() == 0
    assert buffers.num_unique.item() == 0
    assert torch.all(buffers.slot_epoch == -1)


def test_selective_workspace_reuses_dense_dedup_state_for_smaller_shapes():
    workspace_type = getattr(_MODULE, "SelectiveKVWorkspace", None)
    assert workspace_type is not None, "SelectiveKVWorkspace is not implemented"
    workspace = workspace_type(torch.device("cpu"))
    large = workspace.get_dense_dedup_buffers(16, 8, 8)
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            large.slot_epoch,
            large.slot_to_compact,
            large.selected_physical_slots,
            large.remapped_topk,
            large.block_offsets,
            large.epoch,
            large.num_unique,
        )
    )

    small = workspace.get_dense_dedup_buffers(7, 3, 4)

    assert small.slot_epoch.shape == (16,)
    assert small.slot_to_compact.shape == (16,)
    assert small.selected_physical_slots.shape == (8,)
    assert small.remapped_topk.shape == (8,)
    assert pointers == tuple(
        tensor.data_ptr()
        for tensor in (
            small.slot_epoch,
            small.slot_to_compact,
            small.selected_physical_slots,
            small.remapped_topk,
            small.block_offsets,
            small.epoch,
            small.num_unique,
        )
    )


@pytest.mark.parametrize("num_rows", [-1, 1.5, True])
def test_selective_workspace_rejects_invalid_extents(num_rows):
    workspace_type = getattr(_MODULE, "SelectiveKVWorkspace", None)
    assert workspace_type is not None, "SelectiveKVWorkspace is not implemented"
    workspace = workspace_type(torch.device("cpu"))

    with pytest.raises(ValueError, match="non-negative integer"):
        workspace.get_bf16(num_rows)


def test_reference_remap_preserves_minus_one_and_handles_empty_selection():
    page_table = _i32([20, 21])

    mixed = build_selective_kv_remap_reference(page_table, _i32([[1, -1], [-1, 0]]))
    torch.testing.assert_close(mixed.physical_slots, _i32([21, 20]))
    torch.testing.assert_close(mixed.remapped_topk, _i32([[0, -1], [-1, 1]]))

    empty = build_selective_kv_remap_reference(page_table, _i32([[-1, -1]]))
    assert empty.physical_slots.shape == (0,)
    torch.testing.assert_close(empty.remapped_topk, _i32([[-1, -1]]))
    assert empty.num_valid == 0
    assert empty.num_unique == 0


@pytest.mark.parametrize(
    ("topk", "match"),
    [
        ([[-2]], "smaller than -1"),
        ([[2]], "outside page_table_1_flattened"),
    ],
)
def test_reference_remap_rejects_invalid_logical_indices(topk, match):
    with pytest.raises(ValueError, match=match):
        build_selective_kv_remap_reference(_i32([5, 6]), _i32(topk))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int16])
def test_reference_remap_rejects_non_index_dtypes(dtype):
    with pytest.raises(TypeError, match="integer index tensor"):
        build_selective_kv_remap_reference(
            torch.tensor([5, 6], dtype=dtype), _i32([[0]])
        )


def test_traffic_model_accounts_for_full_selective_and_peak_workspace_bytes():
    estimate = estimate_selective_kv_traffic(
        prefix_rows=10_000,
        valid_topk_entries=4_096,
        unique_rows=1_024,
    )

    assert estimate.full_kv_bytes == 10_000 * (
        KV_BYTES_PER_ROW + KV_DEQUANTIZED_BYTES_PER_ROW
    )
    assert estimate.selective_kv_bytes == 1_024 * (
        KV_BYTES_PER_ROW + KV_DEQUANTIZED_BYTES_PER_ROW
    )
    assert estimate.full_workspace_bytes == 10_000 * KV_DEQUANTIZED_BYTES_PER_ROW
    assert estimate.selective_workspace_bytes == (1_024 * KV_DEQUANTIZED_BYTES_PER_ROW)
    assert estimate.metadata_bytes > 0
    assert estimate.selective_total_bytes == (
        estimate.selective_kv_bytes + estimate.metadata_bytes
    )


def test_traffic_model_rejects_impossible_geometry():
    with pytest.raises(ValueError, match="unique_rows.*valid_topk_entries"):
        estimate_selective_kv_traffic(
            prefix_rows=100, valid_topk_entries=4, unique_rows=5
        )
    with pytest.raises(ValueError, match="unique_rows.*prefix_rows"):
        estimate_selective_kv_traffic(
            prefix_rows=4, valid_topk_entries=5, unique_rows=5
        )


def test_conservative_policy_uses_occurrence_upper_bound_without_overlap_guess():
    # 128 selected occurrences out of a long prefix is profitable even if all
    # occurrences are unique.
    assert should_use_selective_kv_dequant(
        prefix_rows=16_384,
        query_tokens=1,
        topk=128,
        safety_factor=1.25,
    )

    # When Q*K covers the full prefix, the conservative policy refuses to
    # assume overlap that has not been measured yet.
    assert not should_use_selective_kv_dequant(
        prefix_rows=1_024,
        query_tokens=8,
        topk=128,
        safety_factor=1.0,
    )


@pytest.mark.parametrize(
    ("prefix_rows", "query_tokens", "topk"),
    [(0, 1, 128), (128, 0, 128), (128, 1, 0)],
)
def test_conservative_policy_rejects_empty_work(prefix_rows, query_tokens, topk):
    assert not should_use_selective_kv_dequant(
        prefix_rows=prefix_rows,
        query_tokens=query_tokens,
        topk=topk,
    )


def test_dense_epoch_policy_accounts_for_full_pool_scan():
    # H20 crossover measurements show that the fixed multi-kernel scan and
    # launch cost dominates short prefixes even when the byte model predicts
    # a saving. Fail closed below the measured profitable region.
    assert not should_use_dense_epoch_kv_dequant(
        prefix_rows=8192,
        query_tokens=1,
        topk=128,
        num_pool_rows=8192,
    )
    assert not should_use_dense_epoch_kv_dequant(
        prefix_rows=32768,
        query_tokens=4,
        topk=2048,
        num_pool_rows=65536,
    )
    assert should_use_dense_epoch_kv_dequant(
        prefix_rows=40960,
        query_tokens=5,
        topk=2048,
        num_pool_rows=65536,
    )
    # Two int64 epoch-table scans over a much larger persistent pool erase the
    # bytes saved from this small logical prefix, so fail closed.
    assert not should_use_dense_epoch_kv_dequant(
        prefix_rows=8192,
        query_tokens=1,
        topk=128,
        num_pool_rows=1_000_000,
    )


def test_dense_epoch_policy_rejects_pool_smaller_than_prefix_physical_extent():
    with pytest.raises(ValueError, match="num_pool_rows"):
        should_use_dense_epoch_kv_dequant(
            prefix_rows=8192,
            query_tokens=1,
            topk=128,
            num_pool_rows=0,
        )


def test_no_dedup_probe_env_gate_defaults_off_and_can_be_enabled():
    field = getattr(
        _ENV_MODULE.envs,
        "SGLANG_EXPERIMENTAL_DSA_SELECTIVE_KV_NO_DEDUP",
        None,
    )
    assert field is not None, "selective no-dedup probe env gate is not registered"
    field.clear()
    try:
        assert field.get() is False
        with field.override(True):
            assert field.get() is True
        assert field.get() is False
    finally:
        field.clear()


def test_dense_epoch_probe_env_gate_defaults_off_and_can_be_enabled():
    field = getattr(
        _ENV_MODULE.envs,
        "SGLANG_EXPERIMENTAL_DSA_SELECTIVE_KV_DENSE_EPOCH",
        None,
    )
    assert field is not None, "selective dense-epoch probe env gate is not registered"
    field.clear()
    try:
        assert field.get() is False
        with field.override(True):
            assert field.get() is True
        assert field.get() is False
    finally:
        field.clear()


def test_prefix_orchestrator_preserves_full_dequant_fallback():
    quant_kv = torch.empty((32, 1, 656), dtype=torch.uint8)
    page_table = torch.arange(32, dtype=torch.int32)
    topk = _i32([[0, 1]])
    workspace = _MODULE.SelectiveKVWorkspace(torch.device("cpu"))
    calls = []

    def fake_dequant(quant, selected, **kwargs):
        calls.append((selected, kwargs))
        return torch.full((selected.numel(), 1, 576), 7, dtype=torch.bfloat16)

    result = dequantize_dsa_prefix_kv_selective(
        quant_kv,
        page_table,
        topk,
        num_pool_rows=32,
        mode="off",
        workspace=workspace,
        dequantize_fn=fake_dequant,
    )

    assert result.mode == "off"
    assert result.remapped_topk is topk
    assert calls == [(page_table, {})]
    assert workspace.bf16_capacity == 0


def test_prefix_orchestrator_no_dedup_does_not_treat_padding_as_prefix():
    quant_kv = torch.empty((4096, 1, 656), dtype=torch.uint8)
    page_table = torch.arange(4096, dtype=torch.int32)
    topk = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    topk[0, 3] = -1
    topk[0, -1] = 200
    workspace = _MODULE.SelectiveKVWorkspace(torch.device("cpu"))
    calls = []

    def fake_dequant(quant, selected, **kwargs):
        calls.append((selected, kwargs.copy()))
        out = kwargs["out"]
        out.fill_(3)
        return out

    result = dequantize_dsa_prefix_kv_selective(
        quant_kv,
        page_table,
        topk,
        num_pool_rows=4096,
        mode="no_dedup",
        workspace=workspace,
        dequantize_fn=fake_dequant,
    )

    assert result.mode == "no_dedup"
    assert result.remapped_topk[0, 3] == -1
    assert result.remapped_topk[0, -1] == 127
    assert calls[0][1].keys() == {"out"}
    assert "num_valid_rows" not in calls[0][1]
    assert calls[0][0][3] == 0  # ignored landing row for the padding occurrence
    physical_buffer, remap_buffer = workspace.get_occurrence_metadata(128)
    assert calls[0][0].data_ptr() == physical_buffer.data_ptr()
    assert result.remapped_topk.data_ptr() == remap_buffer.data_ptr()


def test_prefix_orchestrator_dense_epoch_uses_device_compact_extent():
    quant_kv = torch.empty((40960, 1, 656), dtype=torch.uint8)
    page_table = torch.arange(40960, dtype=torch.int32)
    topk = torch.arange(128, dtype=torch.int32).reshape(1, 128)
    workspace = _MODULE.SelectiveKVWorkspace(torch.device("cpu"))
    calls = []

    def fake_dedup(
        page_table_1_flattened,
        flat_topk_indices,
        *,
        selected_physical_slots,
        remapped_topk,
        num_unique,
        **kwargs,
    ):
        selected_physical_slots[:2].copy_(_i32([11, 29]))
        remapped_topk.copy_(
            (torch.arange(flat_topk_indices.numel(), dtype=torch.int32) % 2).reshape_as(
                flat_topk_indices
            )
        )
        num_unique.fill_(2)
        return selected_physical_slots, remapped_topk, num_unique

    def fake_dequant(quant, selected, **kwargs):
        calls.append((selected, kwargs.copy()))
        out = kwargs["out"]
        out.fill_(5)
        return out

    result = dequantize_dsa_prefix_kv_selective(
        quant_kv,
        page_table,
        topk,
        num_pool_rows=40960,
        mode="dense_epoch",
        workspace=workspace,
        dequantize_fn=fake_dequant,
        deduplicate_fn=fake_dedup,
    )

    assert result.mode == "dense_epoch"
    assert result.remapped_topk.shape == topk.shape
    assert calls[0][0].shape == (128,)
    assert calls[0][1]["num_valid_rows"].item() == 2
    assert calls[0][1]["out"].shape == (128, 1, 576)


def test_prefix_orchestrator_rejects_conflicting_or_unknown_modes():
    quant_kv = torch.empty((1, 1, 656), dtype=torch.uint8)
    page_table = _i32([0])
    topk = _i32([[0]])
    workspace = _MODULE.SelectiveKVWorkspace(torch.device("cpu"))

    with pytest.raises(ValueError, match="unsupported selective KV mode"):
        dequantize_dsa_prefix_kv_selective(
            quant_kv,
            page_table,
            topk,
            num_pool_rows=1,
            mode="both",
            workspace=workspace,
            dequantize_fn=lambda *args, **kwargs: None,
        )


@pytest.mark.parametrize(
    ("no_dedup", "dense_epoch", "expected"),
    [(False, False, "off"), (True, False, "no_dedup"), (False, True, "dense_epoch")],
)
def test_resolve_selective_kv_mode(no_dedup, dense_epoch, expected):
    assert resolve_selective_kv_mode(no_dedup, dense_epoch) == expected


def test_resolve_selective_kv_mode_rejects_two_experiments_at_once():
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_selective_kv_mode(True, True)
