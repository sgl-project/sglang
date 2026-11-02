"""Activated public-auto tests for the locked page64 T3/rho3 policy.

The test JIT adds chooser/fixed exports to the same actual header; the public
Python wrapper still executes FP4IndexerCandidatesKernel::run unchanged.
Cold-first-call graph capture is covered separately by the session harness.
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.fp4_candidate import check, make_inputs

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires SM90",
)


@pytest.fixture(scope="module")
def dispatch_module():
    from sglang.kernels.jit.utils import load_jit

    return load_jit(
        "dpsk_v4_fp4_candidate_dispatch_contract",
        cuda_files=["deepseek_v4/fp4_indexer_candidates.cuh"],
        cuda_wrappers=[
            ("run", "FP4IndexerCandidatesKernel::run"),
            ("query_dispatch", "FP4IndexerCandidatesKernel::query_dispatch"),
            ("query_fixed3", "FP4IndexerCandidatesKernel::query_fixed<3>"),
            ("run_fixed3", "FP4IndexerCandidatesKernel::run_fixed<3>"),
        ],
        extra_dependencies=["cutlass"],
    )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("neighbor", [-1, 0, 1])
def test_public_auto_threshold_neighbors(dispatch_module, monkeypatch, ratio, neighbor):
    from sglang.kernels.ops.attention.dsv4 import fp4_indexer_cuda as scorer

    device = torch.cuda.current_device()
    width = 8192
    sms = int(dispatch_module.query_fixed3(device, 64, ratio, 0))
    resident = int(dispatch_module.query_fixed3(device, 64, ratio, 1))
    groups = (width // 64 + 2) // 3
    transition = (3 * sms * resident + groups - 1) // groups
    rows = max(0, transition + neighbor)
    selected = int(dispatch_module.query_dispatch(device, rows, width, 64, ratio))
    assert selected == (3 if rows and rows >= transition else 1)

    # Warm-query boundary tests deliberately share the exact module/chooser
    # cache with public run. No production Python dispatch override is added.
    monkeypatch.setattr(scorer, "_jit_fp4_candidate_module", lambda: dispatch_module)
    x = make_inputs(
        requests=rows,
        queries=1,
        candidates=width,
        max_len=2 * width,
        ratio=ratio,
        q_kind="fakequant",
        padded=True,
    )
    actual = scorer.fp4_index_logits_candidates_cuda(*x.args())
    expected = torch.empty_like(actual)
    dispatch_module.run_fixed3(
        x.q,
        x.weights,
        x.req,
        x.req_table,
        x.lens,
        x.table,
        x.blocks,
        expected,
        x.page_size,
        x.max_len,
        x.ratio,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    funcs = {
        "cuda": lambda *_: actual,
        "fixed3": lambda *_: expected,
    }
    # These host-dispatch regressions require exact same-source fixed scores
    # and the immutable FP64 oracle. The scorer suite owns Triton comparison.
    check(x, funcs, sample=True)


@pytest.mark.parametrize("page_size", [8, 16, 128])
def test_automatic_generic_page_fallback(dispatch_module, page_size):
    device = torch.cuda.current_device()
    assert dispatch_module.query_dispatch(device, 1 << 40, 1 << 40, page_size, 1) == 1
    assert dispatch_module.query_dispatch(device, 1 << 40, 1 << 40, page_size, 2) == 1


@pytest.mark.parametrize("width", [64, 65, 128])
def test_automatic_requires_three_tiles(dispatch_module, width):
    device = torch.cuda.current_device()
    assert dispatch_module.query_dispatch(device, (1 << 63) - 1, width, 64, 1) == 1


def test_automatic_three_tile_floor_activation(dispatch_module):
    device = torch.cuda.current_device()
    sms = int(dispatch_module.query_fixed3(device, 64, 1, 0))
    resident = int(dispatch_module.query_fixed3(device, 64, 1, 1))
    # Width129 has exactly three K64 tiles and one T3 group per row.
    transition = 3 * sms * resident
    assert dispatch_module.query_dispatch(device, transition - 1, 129, 64, 1) == 1
    assert dispatch_module.query_dispatch(device, transition, 129, 64, 1) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
