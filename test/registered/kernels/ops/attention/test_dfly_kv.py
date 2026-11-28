import pytest
import torch
from sglang.kernels.ops.attention.dsa.dfly_kv import gather_dfly_kv
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="DFly2's FP8 gather uses CUDA",
)


@pytest.mark.parametrize("cache_dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,topk,dim", [(1, 7, 29), (3, 19, 512), (25, 2051, 512)])
def test_dfly_kv_gather(cache_dtype, output_dtype, batch, topk, dim):
    generator = torch.Generator(device="cuda").manual_seed(123)
    cache = torch.randn(
        8192, dim + 8, generator=generator, dtype=torch.bfloat16, device="cuda"
    ).to(cache_dtype)[:, :dim]
    selection = torch.randint(
        8192,
        (batch + 4, topk + 3),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    )[:, :topk]
    selection[:, 0] = -1
    selection[:, -1] = cache.shape[0]
    request_ids = torch.randperm(batch + 4, generator=generator, device="cuda")[:batch]

    def gather():
        return gather_dfly_kv(
            cache, request_ids, selection, dtype=output_dtype, scale=0.75
        )

    def expect():
        indices = selection.index_select(0, request_ids)
        valid = indices.ge(0) & indices.lt(len(cache))
        rows = indices.masked_fill(~valid, 0).long()
        # FP8 index_select may not be available in PyTorch on all CUDA targets.
        source = (
            cache.view(torch.uint8) if cache_dtype == torch.float8_e4m3fn else cache
        )
        picked = source.index_select(0, rows.flatten()).view(cache_dtype)
        reference = (picked.float() * 0.75).to(output_dtype)
        reference = reference.view(*rows.shape, dim).masked_fill(~valid[..., None], 0)
        return reference, valid

    actual = gather()
    torch.testing.assert_close(actual, expect(), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gather()
    request_ids.copy_((request_ids + 1) % len(selection))
    graph.replay()
    torch.testing.assert_close(actual, expect(), rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
