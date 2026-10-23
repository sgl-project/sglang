import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    plan_topk_v2,
    topk_transform_paged_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("capacity", [8192, 1049088])
def test_topk_v2_radix_ties_choose_smallest_indices(capacity):
    if not torch.cuda.is_available():
        pytest.skip("CUDA top-k")
    torch.manual_seed(5735716)
    batch, length, k = 6, 4103, 512
    scores = torch.full((batch, capacity), float("nan"), device="cuda")
    expected = []
    for row, ties in enumerate((129, 215, 225, 231, 1024, 2048)):
        values = torch.full((length,), -8.0, device="cuda")
        # The last slots straddle a BF16 score plateau in one coarse bin.
        values[:471] = -1.0
        values[471 : 471 + ties] = -4.15625
        values = values[torch.randperm(length, device="cuda")]
        scores[row, :length] = values
        chosen = torch.argsort(values, descending=True, stable=True)[:k]
        expected.append(chosen.sort().values)
    expected = torch.stack(expected).to(torch.int32)
    lens = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    plan = plan_topk_v2(lens)
    torch.cuda.synchronize()
    out = torch.empty((batch, k), device="cuda", dtype=torch.int32)

    def select():
        topk_transform_paged_v2(scores, lens, None, out, 1, plan)

    select()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select()
    for iteration in range(16):
        if iteration % 2:
            graph.replay()
        else:
            select()
        assert torch.equal(out.sort(dim=1).values, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
