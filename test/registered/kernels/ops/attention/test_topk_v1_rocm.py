# SPDX-License-Identifier: Apache-2.0
"""ROCm correctness and determinism tests for DeepSeek-V4 JIT top-k v1."""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.srt.utils import is_hip

if not is_hip():
    pytest.skip(
        "DeepSeek-V4 top-k v1 determinism is ROCm-only.", allow_module_level=True
    )
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

import sglang.kernels.ops.attention.dsv4.topk as topk_module  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

DEFAULT_TOPK = 512
CASES = ("signed_zero", "boundary_tie", "overflow_distinct", "ragged")


@pytest.fixture
def force_jit_fallback(monkeypatch):
    """Exercise the path used when the ROCm AOT package lacks this operator."""
    jit_calls = []
    real_jit_module = topk_module._jit_topk_v1_module

    def traced_jit_module():
        jit_calls.append(True)
        return real_jit_module()

    monkeypatch.setattr(topk_module, "_has_dsv4_topk_aot", lambda: False)
    monkeypatch.setattr(topk_module, "_jit_topk_v1_module", traced_jit_module)
    return jit_calls


def _make_case(name: str):
    if name == "signed_zero":
        scores = torch.zeros((1, 768), dtype=torch.float32)
        scores[0, ::2] = -0.0
        return scores, torch.tensor([768], dtype=torch.int32)

    if name == "boundary_tie":
        scores = torch.zeros((1, 768), dtype=torch.float32)
        high = torch.randperm(768, generator=torch.Generator().manual_seed(991))[:500]
        scores[0, high] = torch.arange(1, 501, dtype=torch.float32)
        return scores, torch.tensor([768], dtype=torch.int32)

    if name == "overflow_distinct":
        # All values round into one FP16 coarse bin, but the highest FP32
        # values lie beyond the legacy 6,144-entry LDS candidate buffer.
        scores = torch.linspace(1.00001, 1.00040, 7000).unsqueeze(0)
        return scores, torch.tensor([7000], dtype=torch.int32)

    if name == "ragged":
        scores = torch.full((5, 800), 54321.0, dtype=torch.float32)
        seq_lens = torch.tensor([1, 511, 512, 513, 768], dtype=torch.int32)
        for batch_id, length in enumerate(seq_lens.tolist()):
            valid = torch.arange(length, dtype=torch.float32)
            scores[batch_id, :length] = torch.remainder(valid * 37 + batch_id, 97)
        return scores, seq_lens

    raise AssertionError(f"unknown case: {name}")


def _make_page_table(batch: int, width: int, page_size: int):
    num_pages = (width + page_size - 1) // page_size
    rows = []
    for batch_id in range(batch):
        generator = torch.Generator().manual_seed(1709 + batch_id)
        rows.append(torch.randperm(num_pages, generator=generator, dtype=torch.int32))
    return torch.stack(rows)


def _reference_raw(
    scores: torch.Tensor, seq_lens: torch.Tensor, topk: int
) -> torch.Tensor:
    result = torch.full((scores.shape[0], topk), -1, dtype=torch.int32)
    for batch_id, length_value in enumerate(seq_lens.tolist()):
        length = int(length_value)
        if length <= topk:
            selected = list(range(length))
        else:
            values = scores[batch_id, :length].tolist()
            selected = sorted(
                range(length), key=lambda index: (-float(values[index]), index)
            )[:topk]
            selected.sort()
        result[batch_id, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
    return result


def _reference_page(raw: torch.Tensor, page_table: torch.Tensor, page_size: int):
    page_bits = page_size.bit_length() - 1
    page_mask = page_size - 1
    result = torch.full_like(raw, -1)
    for batch_id in range(raw.shape[0]):
        for column, raw_value in enumerate(raw[batch_id].tolist()):
            if raw_value >= 0:
                physical_page = int(page_table[batch_id, raw_value >> page_bits])
                result[batch_id, column] = (physical_page << page_bits) | (
                    raw_value & page_mask
                )
    return result


def _run_and_check(
    scores_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    page_size: int,
    repeats: int,
    topk: int = DEFAULT_TOPK,
):
    page_table_cpu = _make_page_table(
        scores_cpu.shape[0], scores_cpu.shape[1], page_size
    )
    expected_raw = _reference_raw(scores_cpu, seq_lens_cpu, topk)
    expected_page = _reference_page(expected_raw, page_table_cpu, page_size)

    scores = scores_cpu.cuda()
    seq_lens = seq_lens_cpu.cuda()
    page_table = page_table_cpu.cuda()
    page_out = torch.empty_like(expected_page, device="cuda")
    raw_out = torch.empty_like(expected_raw, device="cuda")
    ordered_outputs = set()

    for _ in range(repeats):
        topk_module.topk_transform_512(
            scores, seq_lens, page_table, page_out, page_size, raw_out
        )
        torch.cuda.synchronize()
        actual_raw = raw_out.cpu()
        actual_page = page_out.cpu()
        assert torch.equal(actual_raw, expected_raw)
        assert torch.equal(actual_page, expected_page)
        ordered_outputs.add(tuple(actual_raw.flatten().tolist()))

    assert len(ordered_outputs) == 1

    # The canonical page output must not depend on requesting the optional raw
    # index buffer.
    topk_module.topk_transform_512(
        scores, seq_lens, page_table, page_out, page_size, None
    )
    torch.cuda.synchronize()
    assert torch.equal(page_out.cpu(), expected_page)


@pytest.mark.parametrize("page_size", [1, 64, 256])
@pytest.mark.parametrize("case_name", CASES)
@torch.inference_mode()
def test_topk_v1_rocm_jit_fallback_is_exact_and_deterministic(
    case_name: str, page_size: int, force_jit_fallback
):
    scores, seq_lens = _make_case(case_name)
    _run_and_check(scores, seq_lens, page_size, repeats=8)
    assert force_jit_fallback


@torch.inference_mode()
def test_topk_v1_rocm_max_benchmarked_length(force_jit_fallback):
    scores = torch.linspace(-1.0, 1.0, 262144).unsqueeze(0)
    seq_lens = torch.tensor([262144], dtype=torch.int32)
    _run_and_check(scores, seq_lens, page_size=64, repeats=2)
    assert force_jit_fallback


@pytest.mark.parametrize("topk", [1, 7, 257, 513, 777, 1024])
@torch.inference_mode()
def test_topk_v1_rocm_runtime_topk_contract(topk: int, force_jit_fallback):
    """Keep main's runtime top-k contract, including non-power-of-two sizes."""
    scores = torch.linspace(-1.0, 1.0, 4096).unsqueeze(0)
    seq_lens = torch.tensor([4096], dtype=torch.int32)
    _run_and_check(scores, seq_lens, page_size=64, repeats=2, topk=topk)
    assert force_jit_fallback


@pytest.mark.parametrize("topk", [257, 777])
@torch.inference_mode()
def test_topk_v1_rocm_long_boundary_tie(topk: int, force_jit_fallback):
    """Choose the lowest logical indices when a long row ties at the boundary."""
    scores = torch.zeros((1, 4096), dtype=torch.float32)
    seq_lens = torch.tensor([4096], dtype=torch.int32)
    _run_and_check(scores, seq_lens, page_size=64, repeats=4, topk=topk)
    assert force_jit_fallback


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
