import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.hicache import _jit_hicache_tma_module
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] < 9,
    reason="HiCache TMA kernel requires SM90+",
)

POOL_TOKENS = 8192
NUM_LAYERS = 3
ROW_DIM = 256  # 512-byte bf16 rows: below the register kernel's 128 B unit width x 4


def _token_indices(num_tokens: int, page_size: int, dtype: torch.dtype, seed: int):
    gen = torch.Generator().manual_seed(seed)
    pages = torch.randperm(POOL_TOKENS // page_size, generator=gen)[
        : num_tokens // page_size
    ]
    idx = (pages[:, None] * page_size + torch.arange(page_size)).reshape(-1)
    return idx.to(device="cuda", dtype=dtype)


def _fill(t: torch.Tensor, seed: int) -> None:
    t.view(torch.int16).copy_(
        torch.randint(
            0,
            30000,
            t.shape,
            dtype=torch.int16,
            generator=torch.Generator().manual_seed(seed),
        )
    )


def _host_view(layout: str, layer: int):
    if layout == "layer_first":
        return torch.empty(POOL_TOKENS, ROW_DIM, dtype=torch.bfloat16, pin_memory=True)
    # page_first: [tokens, layers, dim]; a per-layer view has strided rows
    return torch.empty(
        POOL_TOKENS, NUM_LAYERS, ROW_DIM, dtype=torch.bfloat16, pin_memory=True
    )[:, layer]


@pytest.mark.parametrize("host_layout", ["layer_first", "page_first"])
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("page_size", [128, 1])
def test_one_layer_roundtrip(
    host_layout: str, index_dtype: torch.dtype, page_size: int
) -> None:
    """H2D then D2H of one layer; page runs take the single-op paths (bulk copy,
    tensor-map box, bulk store), scattered rows take the per-row paths, and the
    odd token count leaves a partial tail chunk."""
    module = _jit_hicache_tma_module(block_quota=2)
    num_tokens = 2048 + (96 if page_size == 1 else 0)
    k_host, v_host = _host_view(host_layout, 1), _host_view(host_layout, 2)
    k_dev = torch.zeros(POOL_TOKENS, ROW_DIM, dtype=torch.bfloat16, device="cuda")
    v_dev = torch.zeros_like(k_dev)
    _fill(k_host, 1)
    _fill(v_host, 2)
    host_idx = _token_indices(num_tokens, page_size, index_dtype, seed=3)
    dev_idx = _token_indices(num_tokens, page_size, index_dtype, seed=4)

    module.launch_one(k_dev, v_dev, dev_idx, k_host, v_host, host_idx)
    torch.cuda.synchronize()
    assert torch.equal(k_dev[dev_idx.long()].cpu(), k_host[host_idx.cpu().long()])
    assert torch.equal(v_dev[dev_idx.long()].cpu(), v_host[host_idx.cpu().long()])
    untouched = torch.ones(POOL_TOKENS, dtype=torch.bool, device="cuda")
    untouched[dev_idx.long()] = False
    assert not k_dev[untouched].any() and not v_dev[untouched].any()

    _fill(k_dev, 5)
    _fill(v_dev, 6)
    k_host.zero_()
    v_host.zero_()
    module.launch_one(k_host, v_host, host_idx, k_dev, v_dev, dev_idx)
    torch.cuda.synchronize()
    assert torch.equal(k_host[host_idx.cpu().long()], k_dev[dev_idx.long()].cpu())
    assert torch.equal(v_host[host_idx.cpu().long()], v_dev[dev_idx.long()].cpu())


def _ptr_table(tensors) -> torch.Tensor:
    return torch.tensor(
        [t.data_ptr() for t in tensors], dtype=torch.uint64, device="cuda"
    )


def test_all_layer_tables_lf_to_pf() -> None:
    """All-layer D2H through per-layer pointer tables into a page-first host pool
    (strided destination rows), the write-back shape."""
    module = _jit_hicache_tma_module(block_quota=2)
    k_dev = [
        torch.empty(POOL_TOKENS, ROW_DIM, dtype=torch.bfloat16, device="cuda")
        for _ in range(NUM_LAYERS)
    ]
    v_dev = [torch.empty_like(k_dev[0]) for _ in range(NUM_LAYERS)]
    for i, t in enumerate(k_dev + v_dev):
        _fill(t, 10 + i)
    k_host = torch.zeros(
        POOL_TOKENS, NUM_LAYERS, ROW_DIM, dtype=torch.bfloat16, pin_memory=True
    )
    v_host = torch.zeros_like(k_host).pin_memory()
    host_idx = _token_indices(2048, 128, torch.int64, seed=7)
    dev_idx = _token_indices(2048, 128, torch.int64, seed=8)
    row_bytes = ROW_DIM * 2

    module.launch_all(
        _ptr_table([k_host[:, l] for l in range(NUM_LAYERS)]),
        _ptr_table([v_host[:, l] for l in range(NUM_LAYERS)]),
        host_idx,
        _ptr_table(k_dev),
        _ptr_table(v_dev),
        dev_idx,
        row_bytes,
        NUM_LAYERS * row_bytes,
        row_bytes,
    )
    torch.cuda.synchronize()
    for l in range(NUM_LAYERS):
        assert torch.equal(k_host[host_idx.cpu(), l], k_dev[l][dev_idx].cpu())
        assert torch.equal(v_host[host_idx.cpu(), l], v_dev[l][dev_idx].cpu())
    untouched = torch.ones(POOL_TOKENS, dtype=torch.bool)
    untouched[host_idx.cpu()] = False
    assert not k_host[untouched].any() and not v_host[untouched].any()


def test_mla_single_buffer() -> None:
    """MLA rows (576 x bf16 = 1152 B, not a multiple of 128 B) through the
    single-buffer entry points, one layer and all layers."""
    module = _jit_hicache_tma_module(block_quota=2)
    dim = 576
    dev = [
        torch.empty(POOL_TOKENS, dim, dtype=torch.bfloat16, device="cuda")
        for _ in range(NUM_LAYERS)
    ]
    for i, t in enumerate(dev):
        _fill(t, 20 + i)
    host = torch.zeros(
        POOL_TOKENS, NUM_LAYERS, dim, dtype=torch.bfloat16, pin_memory=True
    )
    host_idx = _token_indices(1024, 128, torch.int64, seed=9)
    dev_idx = _token_indices(1024, 128, torch.int64, seed=10)
    row_bytes = dim * 2

    module.launch_all_mla(
        _ptr_table([host[:, l] for l in range(NUM_LAYERS)]),
        host_idx,
        _ptr_table(dev),
        dev_idx,
        row_bytes,
        NUM_LAYERS * row_bytes,
        row_bytes,
    )
    torch.cuda.synchronize()
    for l in range(NUM_LAYERS):
        assert torch.equal(host[host_idx.cpu(), l], dev[l][dev_idx].cpu())

    dev[0].zero_()
    module.launch_one_mla(dev[0], dev_idx, host[:, 0], host_idx)
    torch.cuda.synchronize()
    assert torch.equal(dev[0][dev_idx].cpu(), host[host_idx.cpu(), 0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
