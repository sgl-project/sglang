"""Tests for the tuned per-token x per-channel (channelwise) FP8 GEMM tables.

The JSON files under ``sglang/kernels/ops/gemm/configs`` are pure data consumed
by ``get_w8a8_channelwise_fp8_config()``; there is no code path that validates
them. A malformed table either silently disables tuning (wrong filename) or
hands ``triton_scaled_mm`` a tile it cannot run, so the tests below cover:

* table shape: filename matches N/K, every bucket has the expected keys and
  values that the kernel accepts;
* lookup semantics: nearest-M snapping, and a ``null`` bucket keeps its own M
  point instead of being skipped in favour of a neighbouring bucket;
* numeric validity: a tuned tile reproduces the exact dequantized FP32 result.
"""

from __future__ import annotations

import json
import os
import re
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.gemm import fp8_kernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(fp8_kernel.__file__)), "configs"
)

_TABLE_NAME = re.compile(
    r"^N=(?P<N>\d+),K=(?P<K>\d+),device_name=(?P<device>.+),"
    r"dtype=fp8_w8a8_channelwise\.json$"
)

_RTX_PRO_5000 = "NVIDIA RTX PRO 5000 72GB Blackwell"

# Weight shapes shipped by this GPU's tables; (N, K) as seen by the GEMM.
_RTX_SHAPES = [
    (2560, 2048),  # qkv
    (2048, 2048),  # o
    (2048, 6144),  # down
    (12288, 2048),  # gate_up
    (130560, 2048),  # lm_head
]

_ENTRY_KEYS = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "num_warps",
    "num_stages",
}
_ALLOWED_NUM_WARPS = {1, 2, 4, 8}


def _table_paths():
    return sorted(
        os.path.join(CONFIGS_DIR, name)
        for name in os.listdir(CONFIGS_DIR)
        if name.endswith("fp8_w8a8_channelwise.json")
    )


def _load(path):
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def rtx_pro_5000():
    """Resolve tables as if we were running on the RTX PRO 5000."""
    with mock.patch.object(fp8_kernel, "get_device_name", lambda: _RTX_PRO_5000):
        fp8_kernel.get_w8a8_channelwise_fp8_configs.cache_clear()
        yield _RTX_PRO_5000
        fp8_kernel.get_w8a8_channelwise_fp8_configs.cache_clear()


@pytest.mark.parametrize("path", _table_paths(), ids=os.path.basename)
def test_table_is_well_formed(path):
    """Filename agrees with the body, and every bucket is loadable by the kernel."""
    match = _TABLE_NAME.match(os.path.basename(path))
    assert match, f"unexpected table filename: {os.path.basename(path)}"
    N, K = int(match["N"]), int(match["K"])

    table = _load(path)
    assert table, f"{os.path.basename(path)} is empty"

    buckets = sorted(int(key) for key in table)
    assert len(set(buckets)) == len(buckets), "duplicate M bucket"
    assert all(m > 0 for m in buckets), "M bucket must be a positive token count"

    tuned = 0
    for m in buckets:
        entry = table[str(m)]
        if entry is None:
            # null marks an M where the CUTLASS dispatch won; the point is kept
            # so that neighbouring M values cannot snap onto it.
            continue
        tuned += 1
        assert set(entry) == _ENTRY_KEYS, f"M={m} has keys {sorted(entry)}"
        assert all(isinstance(entry[key], int) for key in _ENTRY_KEYS)
        assert entry["num_warps"] in _ALLOWED_NUM_WARPS, f"M={m} num_warps"
        assert entry["num_stages"] >= 1, f"M={m} num_stages"
        assert entry["BLOCK_SIZE_M"] >= 16, f"M={m} BLOCK_SIZE_M"
        assert entry["BLOCK_SIZE_N"] >= 16, f"M={m} BLOCK_SIZE_N"
        assert entry["BLOCK_SIZE_K"] >= 32, f"M={m} BLOCK_SIZE_K"
        # A tile larger than the GEMM it tiles is pure waste, and BLOCK_SIZE_K
        # larger than K makes the kernel read past the contraction dim.
        assert entry["BLOCK_SIZE_N"] <= N, f"M={m} BLOCK_SIZE_N > N"
        assert entry["BLOCK_SIZE_K"] <= K, f"M={m} BLOCK_SIZE_K > K"

    assert tuned, f"{os.path.basename(path)} has no tuned buckets"


def test_rtx_pro_5000_tables_are_present_and_complete(rtx_pro_5000):
    """Every shape this GPU ships is reachable through the production lookup."""
    for N, K in _RTX_SHAPES:
        configs = fp8_kernel.get_w8a8_channelwise_fp8_configs(N=N, K=K)
        assert configs, f"missing channelwise table for N={N}, K={K}"
        assert any(entry is not None for entry in configs.values())
        # Decode always starts in the low single digits, so the small-M buckets
        # are the ones the server actually hits; they must be tuned, not null.
        for m in (1, 2, 4):
            assert configs.get(m) is not None, f"N={N}, K={K} M={m} falls back"


def test_lookup_snaps_to_nearest_bucket(rtx_pro_5000):
    """An M between two buckets gets the closer bucket's tile."""
    for N, K in _RTX_SHAPES:
        configs = fp8_kernel.get_w8a8_channelwise_fp8_configs(N=N, K=K)
        buckets = sorted(configs)
        tuned_bucket = next(
            m
            for m, nxt in zip(buckets, buckets[1:])
            if configs[m] is not None and nxt - m >= 5
        )
        expected = configs[tuned_bucket]
        for offset in (1, 2):
            got = fp8_kernel.get_w8a8_channelwise_fp8_config(
                N=N, K=K, M=tuned_bucket + offset
            )
            assert got == expected, f"N={N}, K={K} M={tuned_bucket + offset}"
        got = fp8_kernel.get_w8a8_channelwise_fp8_config(N=N, K=K, M=tuned_bucket)
        assert got == expected


def test_null_bucket_stays_null(rtx_pro_5000):
    """A null bucket must not borrow a neighbour's tile."""
    checked = 0
    for N, K in _RTX_SHAPES:
        configs = fp8_kernel.get_w8a8_channelwise_fp8_configs(N=N, K=K)
        for m, entry in configs.items():
            if entry is None:
                got = fp8_kernel.get_w8a8_channelwise_fp8_config(N=N, K=K, M=m)
                assert got is None, f"N={N}, K={K} M={m} inherited a tile"
                checked += 1
    assert checked, "no null bucket to check -- did the tables change meaning?"


def test_untuned_device_is_a_no_op(rtx_pro_5000):
    """Tables are device-keyed: another GPU keeps the previous dispatch."""
    N, K = _RTX_SHAPES[0]
    with mock.patch.object(fp8_kernel, "get_device_name", lambda: "NVIDIA H100_SXM"):
        fp8_kernel.get_w8a8_channelwise_fp8_configs.cache_clear()
        assert fp8_kernel.get_w8a8_channelwise_fp8_configs(N=N, K=K) is None
        assert fp8_kernel.get_w8a8_channelwise_fp8_config(N=N, K=K, M=1) is None


@pytest.mark.parametrize("N,K", _RTX_SHAPES, ids=[f"N{n}K{k}" for n, k in _RTX_SHAPES])
def test_tuned_tile_matches_fp32_reference(rtx_pro_5000, N, K):
    """Each table's smallest bucket reproduces the exact dequantized product."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 GEMM requires SM89+")

    configs = fp8_kernel.get_w8a8_channelwise_fp8_configs(N=N, K=K)
    M = min(m for m, entry in configs.items() if entry is not None)
    entry = configs[M]

    device = "cuda"
    a = torch.randn((M, K), dtype=torch.float16, device=device).mul_(0.2)
    w = torch.randn((K, N), dtype=torch.float16, device=device).mul_(0.2)
    scale_a = a.abs().amax(dim=1, keepdim=True).float()
    scale_b = w.abs().amax(dim=0, keepdim=True).float().T
    scale_a = scale_a.div_(448.0).clamp_min_(1e-6)
    scale_b = scale_b.div_(448.0).clamp_min_(1e-6)
    qa = (a / scale_a).to(torch.float8_e4m3fn)
    qw = (w.float() / scale_b.T).to(torch.float8_e4m3fn)

    reference = (qa.to(torch.float32) @ qw.to(torch.float32)) * scale_a * scale_b.T
    out = fp8_kernel.triton_scaled_mm(
        qa,
        qw,
        scale_a,
        scale_b,
        torch.bfloat16,
        block_size_m=entry["BLOCK_SIZE_M"],
        block_size_n=entry["BLOCK_SIZE_N"],
        block_size_k=entry["BLOCK_SIZE_K"],
        use_heuristic=False,
        num_warps=entry["num_warps"],
        num_stages=entry["num_stages"],
    )

    error = (out.float() - reference).pow(2).mean().sqrt()
    relative = (error / reference.pow(2).mean().sqrt()).item()
    assert relative < 2e-2, f"N={N}, K={K}, M={M}: relative RMS {relative:.4f}"
