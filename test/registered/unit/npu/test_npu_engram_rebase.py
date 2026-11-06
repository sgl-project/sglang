"""CPU regressions for the merged INT8, FP8 and MXFP4 Engram table paths.

Extract production AST bodies to avoid importing the SGLang/NPU runtime. These
checks cover storage and torch fallback semantics, not device kernels or graphs.
"""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

ENGRAM = ROOT / "python/sglang/srt/layers/engram.py"
CONFIG = ROOT / "python/sglang/srt/arg_groups/engram_config.py"
GATHER = ROOT / "python/sglang/kernels/ops/embeddings/engram_gather.py"
ROWS, DIM, BLOCK = 34, 64, 32


def _extract(path, names, namespace):
    definitions = {
        node.name: node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *(definitions[name] for name in names),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102


@pytest.fixture
def production():
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "FP8_BLOCK_SIZE": BLOCK,
        "get_parallel": Mock(return_value=SimpleNamespace(tp_rank=1, tp_size=2)),
        "resolve_engram_config": Mock(
            return_value=SimpleNamespace(placement="gpu", lookup="reduce_scatter")
        ),
        "_cuda_kernels": Mock(return_value=False),
        "_needs_fp8_byte_view": Mock(return_value=False),
        "engram_gather": Mock(side_effect=AssertionError("unexpected device kernel")),
    }
    _extract(
        CONFIG,
        ["EngramPlacement", "EngramLookup", "EngramReplication", "cols_per_rank"],
        namespace,
    )
    _extract(GATHER, ["unpack_mxfp4_rows"], namespace)
    namespace["unpack_mxfp4_rows"] = Mock(wraps=namespace["unpack_mxfp4_rows"])
    _extract(
        ENGRAM,
        [
            "shard_range",
            "_table_dtype",
            "_table_row_bytes",
            "_e8m0_to_float",
            "EngramEmbedding",
        ],
        namespace,
    )
    return namespace


def _storage():
    return {
        "format": "int8_sym",
        "weight_dtype": "I8",
        "weight_shape": [ROWS, DIM],
        "layout": "row_major",
        "group_size": BLOCK,
        "axis": 1,
        "scale_suffix": "scale",
        "scale_dtype": "F8_E8M0",
        "scale_shape": [ROWS, DIM // BLOCK],
    }


def _table(production, **kwargs):
    return production["EngramEmbedding"](ROWS, DIM, 1, (0, ROWS), **kwargs)


def _expected_owned(table, full_rows, indices):
    owned = (indices >= table.row_start) & (indices < table.row_start + table.rows)
    values = full_rows[indices.clamp(0, ROWS - 1)]
    return values.bfloat16().masked_fill(~owned.unsqueeze(-1), 0)


@pytest.mark.parametrize("raw_scales", [False, True])
def test_a3_int8_load_and_lookup_preserve_e8m0_bytes(production, raw_scales):
    table = _table(production, storage=_storage())
    weights = (torch.arange(ROWS * DIM).reshape(ROWS, DIM) % 256 - 128).to(torch.int8)
    scales = torch.tensor([125, 128], dtype=torch.uint8).expand(ROWS, DIM // BLOCK)
    table._load_rows(table.weight, weights)
    table._load_rows(
        table.scale, scales if raw_scales else scales.view(torch.float8_e8m0fnu)
    )
    assert table.row_start == 17
    assert table.weight.dtype == torch.int8
    assert table.scale.dtype == torch.uint8
    assert torch.equal(table.weight, weights[17:])
    assert torch.equal(table.scale, scales[17:])
    indices = torch.tensor([[17, 33, 16, -1], [18, 17, 34, 0]])
    actual = table._owned_rows(indices)
    full_rows = weights.float() * torch.exp2(scales.float() - 127).repeat_interleave(
        BLOCK, dim=-1
    )
    torch.testing.assert_close(
        actual, _expected_owned(table, full_rows, indices), rtol=0, atol=0
    )
    production["_needs_fp8_byte_view"].assert_not_called()
    production["unpack_mxfp4_rows"].assert_not_called()
    production["engram_gather"].assert_not_called()


def test_a5_fp8_byte_view_load_and_lookup_are_preserved(production):
    table = _table(production)
    weights = ((torch.arange(ROWS * DIM).reshape(ROWS, DIM) % 16 - 8) / 2).to(
        torch.float8_e4m3fn
    )
    scales = torch.tensor([126, 128], dtype=torch.uint8).expand(ROWS, DIM // BLOCK)
    table._load_rows(table.weight, weights)
    table._load_rows(table.scale, scales.view(torch.float8_e8m0fnu))
    assert table.weight.dtype == torch.float8_e4m3fn
    assert table.scale.dtype == torch.float8_e8m0fnu
    assert torch.equal(table.weight.view(torch.uint8), weights.view(torch.uint8)[17:])
    assert torch.equal(table.scale.view(torch.uint8), scales[17:])
    production["_needs_fp8_byte_view"].return_value = True
    indices = torch.tensor([[17, 33, -1, 16, 34, 17]])
    full_rows = weights.float() * torch.exp2(scales.float() - 127).repeat_interleave(
        BLOCK, dim=-1
    )
    torch.testing.assert_close(
        table._owned_rows(indices),
        _expected_owned(table, full_rows, indices),
        rtol=0,
        atol=0,
    )
    production["_needs_fp8_byte_view"].assert_called_once()
    production["unpack_mxfp4_rows"].assert_not_called()
    production["engram_gather"].assert_not_called()


def test_mxfp4_torch_fallback_keeps_packed_storage_and_decoding(production):
    table = _table(production, mxfp4=True)
    packed = torch.arange(ROWS * (DIM // 2)).reshape(ROWS, DIM // 2).to(torch.uint8)
    scales = torch.tensor([125, 129], dtype=torch.uint8).expand(ROWS, DIM // BLOCK)
    table._load_rows(table.weight, packed)
    table._load_rows(table.scale, scales.view(torch.float8_e8m0fnu))
    assert table.weight.dtype == torch.uint8
    assert table.weight.shape == (17, DIM // 2)
    assert table.row_bytes == DIM // 2
    assert torch.equal(table.weight, packed[17:])
    # Independent E2M1 oracle, including both nibble positions and signed zero.
    nibbles = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(-2).long()
    magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    values = magnitudes[nibbles & 7]
    values = torch.where(nibbles >= 8, -values, values)
    full_rows = values * torch.exp2(scales.float() - 127).repeat_interleave(BLOCK, -1)
    indices = torch.tensor([[17, 24, 33, 16, -1, 34]])
    actual = table._owned_rows(indices)
    expected = _expected_owned(table, full_rows, indices)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    production["unpack_mxfp4_rows"].assert_called_once()
    production["engram_gather"].assert_not_called()


def test_int8_storage_cannot_also_select_mxfp4(production):
    with pytest.raises(ValueError, match="INT8.*MXFP4"):
        _table(production, storage=_storage(), mxfp4=True)


def test_load_rejects_incompatible_int8_and_mxfp4_payloads(production):
    int8_table = _table(production, storage=_storage())
    with pytest.raises(ValueError, match="torch.int8"):
        int8_table._load_rows(int8_table.weight, torch.zeros(ROWS, DIM))
    with pytest.raises(ValueError, match="E8M0 bytes"):
        int8_table._load_rows(int8_table.scale, torch.ones(ROWS, DIM // BLOCK))
    mxfp4_table = _table(production, mxfp4=True)
    with pytest.raises(AssertionError, match="MXFP4"):
        mxfp4_table._load_rows(
            mxfp4_table.weight, torch.zeros(ROWS, DIM, dtype=torch.float8_e4m3fn)
        )
