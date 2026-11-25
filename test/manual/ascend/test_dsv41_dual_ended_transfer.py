"""Qualify page relocation and Ascend H2D/D2H using the 910B native row layout."""

import json

import torch
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.base import shared_host_layout_domains
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    get_allocator_from_storage,
)
from sglang.srt.mem_cache.pool_host.dual_ended import DualEndedHostArena


def main():
    torch.npu.set_device(0)
    page = 128
    spec = {"c1": (131072, 1), "idx1": (32768, 1), "c2": (65536, 3), "idx2": (16384, 3)}

    def pinned(size):
        return ALLOC_MEMORY_FUNCS["npu"](
            (size,),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
            allocator=get_allocator_from_storage("default"),
        )

    arena = DualEndedHostArena(spec, (131072, 43), 96, 10, pinned)
    pools = {}
    for name in [*spec, None]:
        region = arena.region(name)
        native = 64 if name in ("c2", "idx2") else page
        bufs = [
            torch.empty(
                (16, native, 1, region.width // (2 * native)),
                dtype=torch.bfloat16,
                device="npu",
            )
            for _ in range(region.layers)
        ]
        pools[name] = DeepSeekV4PagedHostPool(
            str(name),
            bufs,
            region.width,
            16,
            page,
            layout="page_first_direct",
            page_aligned_only=True,
            host_region=region,
        )
    assert shared_host_layout_domains(pools.values()) == [arena]
    F, S = arena.full, arena.swa
    full = F.alloc(16, page)
    swa = S.alloc(8, page)
    expected = {}

    def backup(name, slots, value):
        p = pools[name]
        n = slots.numel() // page
        for i, b in enumerate(p.device_buffers):
            b[:n].copy_(
                (torch.arange(n, dtype=torch.float32).to(torch.bfloat16) + i + value)
                .view(n, 1, 1, 1)
                .expand_as(b[:n])
            )
        p.backup_from_device_all_layer(
            None, slots, torch.arange(n * page), "kernel_ascend"
        )
        expected[name] = value

    def check(name, slots, expected_ids):
        p = pools[name]
        n = len(expected_ids)
        for i, b in enumerate(p.device_buffers):
            b.zero_()
            p.load_to_device_per_layer(
                None, slots, torch.arange(n * page), i, "kernel_ascend"
            )
            actual = b[:n].cpu()
            ref = (
                (torch.tensor(expected_ids, dtype=torch.float32) + i + expected[name])
                .to(torch.bfloat16)
                .view(n, 1, 1, 1)
            )
            assert torch.equal(actual, ref.expand_as(actual)), (name, i)

    for i, name in enumerate(spec):
        backup(name, full, 16 * i)
    backup(None, swa, 80)
    torch.npu.synchronize()
    S.free(swa[3 * page : 4 * page], page)
    keep = torch.cat([swa[: 3 * page], swa[4 * page :]])
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        pools[None].load_to_device_per_layer(
            None, keep, torch.arange(len(keep)), 0, "kernel_ascend"
        )
    old_moves = arena.moves
    extra = F.alloc(F.available(), page)
    assert extra is not None and arena.moves > old_moves
    # The queued H2D must have read the old source correctly before relocation.
    actual = pools[None].device_buffers[0][:7].cpu()
    assert torch.equal(
        actual,
        torch.tensor([80, 81, 82, 84, 85, 86, 87], dtype=torch.bfloat16)
        .view(-1, 1, 1, 1)
        .expand_as(actual),
    )
    check(None, keep, [0, 1, 2, 4, 5, 6, 7])
    for name in spec:
        check(name, full, list(range(16)))
    # Force FULL page moves across alignment phases. The original four regions
    # have different component order depending on the physical page's address.
    torch.npu.synchronize()
    F.free(full[: 14 * page], page)
    extra_s = S.alloc(1, page)
    assert extra_s is not None
    for name in spec:
        check(name, full[14 * page :], [14, 15])
    assert F.alloc(10**7, page) is None
    torch.npu.synchronize()
    F.free(extra, page)
    F.free(full[14 * page :], page)
    S.free(keep, page)
    S.free(extra_s, page)
    F.clear()
    S.clear()
    assert arena.free_bytes() == arena._allocatable_bytes
    print(
        json.dumps(
            dict(
                roundtrip=True,
                swa_layers=43,
                full_regions=4,
                page_moves=arena.moves,
                bytes=arena.budget,
            )
        )
    )


if __name__ == "__main__":
    main()
