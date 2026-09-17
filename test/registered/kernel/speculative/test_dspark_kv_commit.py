"""The DSpark FlashMLA writer must preserve bytes for stacked KV views."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import fused_k_norm_rope_flashmla
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    _STACKED_WEIGHT_CACHE,
    CommitKvProj,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.parametrize("rows", [1, 5, 6, 8, 64])
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
def test_stacked_projection_writer_graph(rows, pos_dtype):
    device = "cuda"
    x = torch.randn(rows, 5120, dtype=torch.bfloat16, device=device)
    linears = [
        SimpleNamespace(
            weight=torch.randn(512, 5120, device=device, dtype=torch.bfloat16),
            quant_method=SimpleNamespace(),
        )
        for _ in range(3)
    ]
    knw = torch.randn(3, 512, dtype=torch.bfloat16, device=device)
    freqs = torch.polar(
        torch.ones(4096, 32, device=device), torch.randn(4096, 32, device=device)
    )
    positions = torch.arange(rows, device=device, dtype=pos_dtype)
    loc = torch.arange(rows, device=device, dtype=torch.int32) + 120
    page_bytes = ((584 * 128 + 575) // 576) * 576
    ref_caches = [
        torch.full((4, page_bytes), 127, dtype=torch.uint8, device=device)
        for _ in range(3)
    ]
    out_caches = [torch.full_like(c, 127) for c in ref_caches]

    def run(views, caches):
        kvs = CommitKvProj.execute(
            main_x=x, wkv_linears=linears, allow_strided_output=views
        )
        if views:
            assert all(kv.stride(0) == 1536 for kv in kvs)
        for stage, kv in enumerate(kvs):
            fused_k_norm_rope_flashmla(
                kv, knw[stage], 1e-6, freqs, positions, loc, caches[stage], 128
            )
        return kvs

    for _ in range(3):
        run(True, out_caches)
        run(False, ref_caches)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run(True, out_caches)
    for replay in range(7):
        x.normal_()
        knw.normal_()
        positions.copy_(torch.arange(rows, device=device) + 127 * replay)
        loc.copy_(torch.arange(rows, device=device) + 120 + replay)
        loc[::3] = -1
        if replay % 2:
            loc[-1] = 511
        graph.replay()
        refs = run(False, ref_caches)
        for out, ref, cache, ref_cache in zip(outputs, refs, out_caches, ref_caches):
            assert torch.equal(out, ref)
            assert torch.equal(cache, ref_cache)
    _STACKED_WEIGHT_CACHE.pop(id(linears[0]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
