"""Tuned cuBLASLt algorithms for FlashInfer bmm_fp8: every shipped entry gives the stock
output byte for byte, and a mismatching or unusable algorithm leaves the stock bytes."""

import json
import os
import sys

import msgspec
import pytest
import torch

from sglang.kernels.ops.gemm import bmm_fp8_cublaslt_algos as algos
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-c", runner_config="2-gpu-gb300")


def _entries():
    params = []
    for name in sorted(os.listdir(algos._CONFIG_DIR)):
        with open(os.path.join(algos._CONFIG_DIR, name)) as f:
            doc = json.load(f)
        device = doc["meta"]["device"]
        params += [
            pytest.param(device, key, id=f"{device}-{key}")
            for key in sorted(algos._parse_table(doc))
        ]
    return params


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(algos, "_tables", {})
    monkeypatch.setattr(algos, "_validated", set())
    monkeypatch.setattr(algos, "_refused", set())
    monkeypatch.setattr(algos, "_logged", set())
    return algos


def _require_table(device_name):
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != device_name:
        pytest.skip(f"the table is for {device_name}")
    table = algos._load(torch.device("cuda", torch.cuda.current_device()))
    if table is None:
        pytest.skip("the table does not load with this cuBLASLt and FlashInfer")
    return table


def _inputs(n, k, m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, k, generator=g, device="cuda") * 3).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.05).to(torch.float8_e4m3fn)
    scales = (
        torch.tensor([0.0123], device="cuda"),
        torch.tensor([0.0021], device="cuda"),
    )
    return a, w.t(), *scales


def _stock(a, w, x_scale, w_scale):
    from flashinfer import bmm_fp8

    out = bmm_fp8(
        a.unsqueeze(0),
        w.unsqueeze(0),
        x_scale,
        w_scale,
        torch.bfloat16,
        backend="cublas",
    )
    return out.view(a.shape[0], w.shape[1])


def _bits(t):
    return t.view(torch.int16)


@pytest.mark.parametrize("device_name,key", _entries())
def test_entry_is_byte_equal_to_stock(fresh, device_name, key):
    _require_table(device_name)
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    n, k, m = key
    for seed in (1, 2):
        a, w, x_scale, w_scale = _inputs(n, k, m, seed)
        got = flashinfer_bmm_fp8(a, w, x_scale, w_scale, torch.bfloat16)
        assert (a.device.index, n, k, m) in fresh._validated, "the entry did not run"
        assert torch.equal(_bits(got), _bits(_stock(a, w, x_scale, w_scale)))


@pytest.mark.parametrize("device_name,key", _entries()[:1])
def test_unusable_or_mismatching_algorithm_leaves_stock_bytes(fresh, device_name, key):
    table = _require_table(device_name)
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_bmm_fp8

    n, k, m = key
    a, w, x_scale, w_scale = _inputs(n, k, m, 3)
    ref = _stock(a, w, x_scale, w_scale)
    off = ref.clone()
    _bits(off)[0, 0] ^= 1
    got = fresh.run_with_tuned_algo(
        q_input=a,
        weight=w,
        x_scale=x_scale,
        weight_scale=w_scale,
        out_dtype=torch.bfloat16,
        stock=lambda: off,
    )
    assert got is off and (a.device.index, n, k, m) in fresh._refused
    # algorithm bytes cuBLASLt cannot use: the caller still gets the stock bytes
    fresh._tables[a.device.index] = msgspec.structs.replace(
        table, algos={(n, k): {m: torch.full((64,), 255, dtype=torch.uint8)}}
    )
    fresh._refused.clear()
    out = flashinfer_bmm_fp8(a, w, x_scale, w_scale, torch.bfloat16)
    assert torch.equal(_bits(out), _bits(ref))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
