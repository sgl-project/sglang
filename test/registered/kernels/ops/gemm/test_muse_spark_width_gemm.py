"""Muse Spark exact-width GEMM table: off until enabled, gated on device and ROCm build, every row close to the
default path (summation order only) and deterministic, also under graph capture."""

import json
import os

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

OUT = {"bf16": torch.bfloat16, "f32": torch.float32}


def _rows():
    from sglang.kernels.ops.gemm import muse_spark_width_gemm as wg

    name = torch.cuda.get_device_name().replace(" ", "_")
    path = os.path.join(
        wg._CONFIG_DIR, f"muse_spark_width_gemm,device_name={name}.json"
    )
    if not os.path.exists(path):
        return []
    raw = json.load(open(path))
    return [
        (*map(int, key.split(",")[:2]), key.split(",")[2], int(m))
        for key, by_m in raw.items()
        if not key.startswith("_")
        for m in by_m
    ]


ROWS = _rows() if is_hip() and is_gfx95_supported() else []

pytestmark = pytest.mark.skipif(not ROWS, reason="No width table for this GPU.")


@pytest.fixture
def wg(monkeypatch):
    from sglang.kernels.ops.gemm import muse_spark_width_gemm as wg

    monkeypatch.setattr(wg, "_enabled", True)
    return wg


def _default(x, w, out_dtype):
    if out_dtype == torch.float32:
        return torch.mm(x, w.T, out_dtype=torch.float32)
    return F.linear(x, w)


def test_off_until_enabled():
    from sglang.kernels.ops.gemm import muse_spark_width_gemm as wg

    n, k, out, m = ROWS[0]
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    assert not wg.width_gemm_applies(x, w, OUT[out])


def test_other_widths_shapes_and_builds_keep_the_default_path(wg, monkeypatch):
    n, k, out, m = ROWS[0]
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    rows = {(n2, k2, o2, m2) for n2, k2, o2, m2 in ROWS}
    other_m = next(i for i in range(1, 4096) if (n, k, out, i) not in rows)
    assert wg.width_gemm_applies(
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16), w, OUT[out]
    )
    assert not wg.width_gemm_applies(
        torch.randn(other_m, k, device="cuda", dtype=torch.bfloat16), w, OUT[out]
    )
    assert not wg.width_gemm_applies(
        torch.randn(m, 64, device="cuda", dtype=torch.bfloat16),
        torch.randn(n, 64, device="cuda", dtype=torch.bfloat16),
        OUT[out],
    )
    with monkeypatch.context() as mp:
        mp.setattr(torch.version, "hip", "0.0.0")
        wg._table.cache_clear()
        try:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            assert not wg.width_gemm_applies(x, w, OUT[out])
        finally:
            wg._table.cache_clear()


@torch.no_grad()
@pytest.mark.parametrize("n,k,out,m", ROWS)
def test_row_matches_the_default_path_and_is_deterministic(wg, n, k, out, m):
    torch.manual_seed(n + k + m)
    out_dtype = OUT[out]
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    assert wg.width_gemm_applies(x, w, out_dtype)
    y = wg.width_gemm(x, w, out_dtype)
    assert y.dtype == out_dtype and y.shape == (m, n)
    assert torch.equal(wg.width_gemm(x, w, out_dtype), y)
    ref = x.double() @ w.double().T
    err = (y.double() - ref).abs().mean()
    err_default = (_default(x, w, out_dtype).double() - ref).abs().mean()
    assert err <= 1.15 * err_default + 1e-7


@torch.no_grad()
def test_graph_capture_replays_the_eager_result(wg):
    n, k, out, m = max(ROWS, key=lambda r: r[0] * r[1])
    out_dtype = OUT[out]
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    expected = wg.width_gemm(x, w, out_dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = wg.width_gemm(x, w, out_dtype)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, expected)
