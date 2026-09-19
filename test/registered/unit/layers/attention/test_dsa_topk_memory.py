"""Exercise the real top-k dispatch with CPU kernels and check page-table storage."""

import ast
import sys
from enum import Enum, IntEnum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _load_backend():
    root = Path(__file__).resolve().parents[5]
    tree = ast.parse(
        (
            root / "python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py"
        ).read_text()
    )
    classes = ast.Module(
        body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[]
    )
    namespace = dict(
        Enum=Enum,
        IntEnum=IntEnum,
        auto=auto,
        torch=torch,
        envs=SimpleNamespace(
            SGLANG_DSA_FUSE_TOPK=SimpleNamespace(get=lambda: True),
            SGLANG_OPT_USE_TOPK_V2=SimpleNamespace(get=lambda: False),
        ),
    )
    exec("from __future__ import annotations\n" + ast.unparse(classes), namespace)
    return namespace["DSATopKBackend"], namespace["TopkTransformMethod"]


@pytest.mark.parametrize(
    "table_rows,batch_indices,cu_q",
    [
        (1, [0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5]),
        (1, [0, 0], [0, 2, 5]),
        (1, None, [0, 5]),
        (2, [1, 0, 1], [0, 1, 3, 5]),
        (2, None, [0, 2, 5]),
    ],
)
def test_paged_topk_reuses_single_request_table(
    monkeypatch, table_rows, batch_indices, cu_q
):
    backend, method = _load_backend()
    # A view with a nonzero storage offset also has to retain the correct base.
    page_table = torch.arange(7 + table_rows * 16, dtype=torch.int32)[7:].view(
        table_rows, 16
    )
    page_table_before = page_table.clone()
    scores = torch.arange(16, dtype=torch.float32).repeat(5, 1)
    lengths = torch.tensor([4, 3, 2, 0, 5], dtype=torch.int32)
    starts = torch.tensor([0, 3, 0, 0, 5], dtype=torch.int32)
    cumulative = torch.tensor(cu_q, dtype=torch.int32)
    indices = None if batch_indices is None else torch.tensor(batch_indices)
    topk = 2

    def cpu_kernel(score, lengths, page_table_size_1, cu_seqlens_q, topk, row_starts):
        if table_rows == 1:
            assert page_table_size_1.untyped_storage().data_ptr() == (
                page_table.untyped_storage().data_ptr()
            ), "Single-request top-k duplicated the context-wide page table"
        assert page_table_size_1.stride(1) == 1
        assert page_table_size_1.shape[0] == len(cu_q) - 1
        torch.testing.assert_close(cu_seqlens_q, cumulative)
        result = torch.full((5, topk), -1, dtype=torch.int32)
        for group, (lo, hi) in enumerate(zip(cu_q, cu_q[1:])):
            for row in range(lo, hi):
                length, start = int(lengths[row]), int(row_starts[row])
                # Match the legacy kernel's short-row fast path.
                chosen = (
                    torch.arange(length)
                    if length <= topk
                    else score[row, start : start + length].topk(topk).indices
                )
                result[row, : len(chosen)] = page_table_size_1[group, chosen]
        return result

    kernels = ModuleType("sgl_kernel")
    kernels.fast_topk_transform_fused = cpu_kernel
    kernels.fast_topk_transform_ragged_fused = lambda **kwargs: pytest.fail(
        "Unexpected ragged dispatch"
    )
    monkeypatch.setitem(sys.modules, "sgl_kernel", kernels)
    actual = backend.SGL_KERNEL.topk_transform(
        logits=scores,
        lengths=lengths,
        topk=topk,
        topk_transform_method=method.PAGED,
        attn_metadata=SimpleNamespace(page_table_1=page_table),
        cu_seqlens_q_topk=cumulative,
        row_starts=starts,
        batch_idx_list=indices,
    )
    local_indices = [[3, 2], [2, 1], [0, 1], [], [4, 3]]
    expected = torch.full((5, topk), -1, dtype=torch.int32)
    for group, (lo, hi) in enumerate(zip(cu_q, cu_q[1:])):
        source = group if batch_indices is None else batch_indices[group]
        for row in range(lo, hi):
            chosen = local_indices[row]
            expected[row, : len(chosen)] = page_table[source, chosen]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(page_table, page_table_before)
