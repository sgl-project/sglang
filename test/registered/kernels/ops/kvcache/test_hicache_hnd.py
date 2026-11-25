"""HiCache round-trips an HND device pool ([pages, heads, page_size, head_dim]) like NHD slot rows.

Pages are backed up to the host and loaded back into other device pages; the HND pool must end up
holding the same KV as the NHD pool, viewed as slot rows.

    python -m pytest test/registered/kernels/ops/kvcache/test_hicache_hnd.py -v
"""

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_L, _H, _D, _PAGES = 3, 4, 128, 8


def _device_pool(kv_cache_layout, page, dtype):
    return MHATokenToKVPool(
        size=(_PAGES - 1) * page,
        page_size=page,
        dtype=dtype,
        head_num=_H,
        head_dim=_D,
        layer_num=_L,
        device="cuda",
        enable_memory_saver=False,
        kv_cache_layout=kv_cache_layout,
    )


def _as_nhd(hnd: torch.Tensor) -> torch.Tensor:
    pages, heads, page, head_dim = hnd.shape
    return hnd.permute(0, 2, 1, 3).reshape(pages * page, heads, head_dim)


def _host(device, layout):
    host = MHATokenToKVPoolHost(
        buffer_info=device.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=device.page_size,
        layout=layout,
        pin_memory=True,
        device="cpu",
        allocator_type="default",
        device_capacity_tokens=device.size,
        start_layer=device.start_layer,
        end_layer=device.end_layer,
        write_back_staging=device.hicache_write_back_staging,
    )
    return host


def _slots(pages, page):
    return (pages[:, None] * page + torch.arange(page)).flatten()


def _round_trip(device, layout, io_backend):
    page = device.page_size
    host = _host(device, layout)
    try:
        # Host pages out of order, device pages moved to other pages.
        host_slots = host.alloc(4 * page).view(4, page)[[3, 0, 2]].flatten()
        src = _slots(torch.tensor([1, 4, 5]), page)
        dst = _slots(torch.tensor([6, 2, 3]), page)
        backup_slots = host_slots
        if io_backend == "kernel":
            host_slots, src, dst = host_slots.cuda(), src.cuda(), dst.cuda()
            # The staged write-back reads host indices on the CPU.
            if not host.can_use_write_back_jit:
                backup_slots = host_slots
        # Batched direct copies need a non-default stream, as the controller's.
        with torch.cuda.stream(torch.cuda.Stream()):
            host.backup_from_device_all_layer(device, backup_slots, src, io_backend)
            torch.cuda.synchronize()
            for layer in range(_L):
                host.load_to_device_per_layer(
                    device, host_slots, dst, layer, io_backend
                )
            torch.cuda.synchronize()
    finally:
        host.destroy()
    return src.cpu(), dst.cpu()


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize(
    "layout,io_backend",
    [
        ("layer_first", "kernel"),
        ("page_first", "kernel"),
        ("layer_first", "direct"),
        ("page_first_direct", "direct"),
    ],
)
@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_hnd_round_trip_matches_nhd(layout, io_backend, page, dtype):
    torch.manual_seed(0)
    nhd, hnd = _device_pool("nhd", page, dtype), _device_pool("hnd", page, dtype)
    for n_buf, h_buf in zip(nhd.k_buffer + nhd.v_buffer, hnd.k_buffer + hnd.v_buffer):
        _bytes(n_buf).copy_(torch.randint_like(_bytes(n_buf), 0, 120))
        h_buf.copy_(n_buf.view(_PAGES, page, _H, _D).transpose(1, 2))
    want = [_bytes(b).clone() for b in nhd.k_buffer + nhd.v_buffer]

    src, dst = _round_trip(nhd, layout, io_backend)
    _round_trip(hnd, layout, io_backend)

    for n_buf, h_buf, old in zip(
        nhd.k_buffer + nhd.v_buffer, hnd.k_buffer + hnd.v_buffer, want
    ):
        assert torch.equal(_bytes(n_buf)[dst.cuda()], old[src.cuda()])
        assert torch.equal(_bytes(_as_nhd(h_buf)), _bytes(n_buf))


@pytest.mark.parametrize("layout", ["page_head", "page_unified"])
def test_hnd_refuses_head_split_host_layouts(layout):
    hnd = _device_pool("hnd", 64, torch.bfloat16)
    with pytest.raises(ValueError, match="HND"):
        _host(hnd, layout)
