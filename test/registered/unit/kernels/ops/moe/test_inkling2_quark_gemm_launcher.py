"""CPU tests of the launchers of the Inkling-2 Quark expert kernels: the config tables,
the operand checks, and that no launcher copies what it is given. The Triton kernels are
replaced by a stub, so nothing runs on a device."""

import sys
import types

import pytest
import torch

from sglang.kernels.ops.moe import inkling2_quark_gemm_hip as G
from sglang.kernels.ops.moe import inkling2_swiglu_bf16_hip as swiglu_hip
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.inkling2_quark_test_utils import Materialization

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

N = 2304
FORMATS = (G.MXFP4, G.UINT4_AWQ)


class _NoKernel:
    """Stands in for a Triton kernel: a launch does nothing."""

    def __getitem__(self, grid):
        return lambda *args, **options: None


def _cpu_operands(fmt, experts=4, k=2304):
    """Zero tensors with the shapes and dtypes of one layer's packed experts."""
    if fmt == G.MXFP4:
        weight = torch.zeros(experts, N, k // 2, dtype=torch.uint8)
        return weight, torch.zeros(experts, N, k // 32, dtype=torch.uint8), None
    weight = torch.zeros(experts, k, N // 8, dtype=torch.int32)
    scale = torch.zeros(experts, k // 128, N, dtype=torch.bfloat16)
    return weight, scale, torch.zeros(experts, k // 128, N // 8, dtype=torch.int32)


def _cpu_route(experts, rows, block_m):
    """The fields of inkling2_experts_hip.Route the launcher reads."""
    return types.SimpleNamespace(
        dst2src=torch.zeros(rows, dtype=torch.int32),
        offsets=torch.zeros(experts + 1, dtype=torch.int32),
        counts=torch.zeros(experts, dtype=torch.int32),
        block_m=block_m,
        schedule=torch.zeros(8, dtype=torch.int32),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_tables_resolve_to_valid_tiles_for_both_gemms(fmt):
    for _, cfg in G._TABLES[fmt]:
        for k in (2304, 1152):  # w13, w2
            r = G.resolve_config(cfg, fmt, k)
            assert k % r.bk == 0 and (k // r.bk) % r.split_k == 0
            assert N % r.bn == 0 and r.bk % 32 == 0
            assert fmt != G.UINT4_AWQ or 128 % r.bk == 0  # within one scale group


def test_launcher_rejects_inconsistent_operands():
    x = torch.zeros(8, 2304, dtype=torch.bfloat16)
    mx_w, mx_s, _ = _cpu_operands(G.MXFP4)
    u4_w, u4_s, u4_z = _cpu_operands(G.UINT4_AWQ)
    route = _cpu_route(4, 64, 16)
    cases = [
        (x.float(), mx_w, mx_s, None, G.MXFP4, "contiguous 2-D BF16"),
        (x, mx_w, mx_s, u4_z, G.MXFP4, "expected shapes"),
        (x, u4_w, u4_s, None, G.UINT4_AWQ, "expected shapes"),
        (x, mx_w.transpose(0, 1), mx_s, None, G.MXFP4, "must be contiguous"),
        (x[:7], mx_w, mx_s, None, G.MXFP4, "route / operand mismatch"),
    ]
    for x_, weight, scale, zero_point, fmt, error in cases:
        with pytest.raises(ValueError, match=error):
            G.expert_gemm_quark(
                x_, weight, scale, zero_point, route, N, 8, torch.float32, 64,
                fmt=fmt, num_experts=4,
            )  # fmt: skip


def test_the_watch_sees_a_dense_copy():
    weight, scale, zero_point = _cpu_operands(G.UINT4_AWQ)
    with Materialization(weight, scale, zero_point) as watch:
        views = (weight[0], weight.view(torch.uint8), weight.flatten())
    assert not watch.copies
    base = weight.untyped_storage().data_ptr()
    assert all(v.untyped_storage().data_ptr() == base for v in views)
    with Materialization(weight, scale, zero_point) as watch:
        weight.to(torch.bfloat16)
    assert [c[0] for c in watch.copies] == ["aten._to_copy.default"]


@pytest.mark.parametrize("fmt", FORMATS)
def test_launcher_makes_no_weight_copy(monkeypatch, fmt):
    monkeypatch.setattr(G, "_grouped_quark_gemm", _NoKernel())
    rows, experts = 64, 4
    weight, scale, zero_point = _cpu_operands(fmt, experts)
    x = torch.zeros(rows // 8, 2304, dtype=torch.bfloat16)
    config = G.default_config_quark(fmt, rows, experts)
    route = _cpu_route(experts, rows, config.bm)
    with Materialization(weight, scale, zero_point) as watch:
        G.expert_gemm_quark(
            x, weight, scale, zero_point, route, N, 8, torch.float32, rows,
            fmt=fmt, num_experts=experts, config=config,
        )  # fmt: skip
    assert not watch.copies, watch.copies


@pytest.mark.parametrize("splits", (1, 3))
def test_the_swiglu_launcher_materializes_only_its_output(monkeypatch, splits):
    monkeypatch.setattr(swiglu_hip, "_swiglu_bf16", _NoKernel())
    rows, width = 64, 1152
    acc = torch.zeros(splits, rows, 2 * width)
    ids = torch.zeros(rows, dtype=torch.int32)
    gammas = torch.zeros(rows)
    with Materialization(acc, ids, gammas) as watch:
        y = swiglu_hip.swiglu_bf16(acc, ids, gammas, 0, 8, (4.0, 25.0))
    assert y.shape == (rows, width) and y.dtype == torch.bfloat16
    assert not watch.copies, watch.copies


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
