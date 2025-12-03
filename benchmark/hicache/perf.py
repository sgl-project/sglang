from __future__ import annotations

from typing import Any, Callable, NamedTuple

import torch


def jit_hicache_impl(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    item_bytes: int,
    block_quota: int,
) -> None:
    from sglang.jit_kernel.hicache import transfer_hicache_one_layer

    _ = item_bytes

    transfer_hicache_one_layer(
        k_cache_dst=k_cache_dst,
        v_cache_dst=v_cache_dst,
        indices_dst=indices_dst,
        k_cache_src=k_cache_src,
        v_cache_src=v_cache_src,
        indices_src=indices_src,
        block_quota=block_quota,
    )


def ref_hicache_impl(
    k_cache_dst: torch.Tensor,
    v_cache_dst: torch.Tensor,
    indices_dst: torch.Tensor,
    k_cache_src: torch.Tensor,
    v_cache_src: torch.Tensor,
    indices_src: torch.Tensor,
    item_bytes: int,
    block_quota: int,
) -> None:
    from sgl_kernel import transfer_kv_per_layer

    transfer_kv_per_layer(
        src_k=k_cache_src,
        src_v=v_cache_src,
        dst_k=k_cache_dst,
        dst_v=v_cache_dst,
        src_indices=indices_src,
        dst_indices=indices_dst,
        item_size=item_bytes,
        block_quota=block_quota,
    )


class HicacheBenchArgs(NamedTuple):
    cache_item_size: int
    dtype: torch.dtype
    block_quota: int


def perf(f: Callable[[], Any], loop: int = 100) -> float:
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    # warm up
    f()
    torch.cuda._sleep(10**8)
    tic.record()
    for _ in range(loop):
        f()
    toc.record()
    toc.synchronize()
    return tic.elapsed_time(toc) / loop


@torch.inference_mode()
def test_hicache_kernel(args: HicacheBenchArgs) -> None:
    CACHE_ITEM_SIZE, DTYPE, BLOCK_QUOTA = args

    CUDA_CACHE_SIZE = 1024 * 1024
    HOST_CACHE_SIZE = CUDA_CACHE_SIZE * 2

    cuda_cache = torch.randn(
        (2, CUDA_CACHE_SIZE, CACHE_ITEM_SIZE),
        dtype=DTYPE,
        device="cuda",
    )
    host_cache = torch.empty(
        (2, HOST_CACHE_SIZE, CACHE_ITEM_SIZE),
        dtype=DTYPE,
        device="cpu",
        pin_memory=True,
    )

    ITEM_BYTES = cuda_cache.element_size() * CACHE_ITEM_SIZE

    def _gen_indices(size: int, bs: int) -> torch.Tensor:
        assert bs <= size
        result = (
            (torch.randperm(size, dtype=torch.int64, device="cuda")[:bs]).sort().values
        )
        if not (torch.all(result >= 0) and torch.all(result < size)):
            where = (result < 0) | (result >= size)
            place = where.nonzero(as_tuple=False)
            print("Invalid indices at positions:", place)
            print("Invalid indices values:", result[place])
            raise ValueError("Generated invalid indices")
        return result

    def _calc_tput(dur: float) -> float:
        return (MEM / (1024**3)) / (dur / 1000)  # GB/s

    def _gain_str(aot_dur: float, jit_dur: float) -> str:
        gain = 100 * (aot_dur / jit_dur - 1)
        if gain >= 0:
            return f"+{gain:>6.2f}%"
        else:
            return f"-{-gain:>6.2f}%"

    print(f"{CACHE_ITEM_SIZE = }, {DTYPE = }, {BLOCK_QUOTA = }")

    def _fast_test_correctness(bs: int):
        src_indices = _gen_indices(CUDA_CACHE_SIZE, bs)
        dst_indices = _gen_indices(HOST_CACHE_SIZE, bs)
        host_cache_cuda = torch.randn_like(host_cache, device="cuda")
        host_cache.copy_(host_cache_cuda, non_blocking=True)

        # copy from cuda to host
        jit_hicache_impl(
            k_cache_dst=host_cache[0],
            v_cache_dst=host_cache[1],
            indices_dst=dst_indices,
            k_cache_src=cuda_cache[0],
            v_cache_src=cuda_cache[1],
            indices_src=src_indices,
            item_bytes=ITEM_BYTES,
            block_quota=BLOCK_QUOTA,
        )
        dst_indices = dst_indices.cpu()
        assert torch.all(
            host_cache[0][dst_indices].cuda() == cuda_cache[0][src_indices]
        )

    BS_RANGE = [2**n for n in range(8, 18)]
    for bs in BS_RANGE:
        _fast_test_correctness(bs)

    print("Correctness passed! Start HiCache kernel performance test...")
    print("=" * 70)

    for bs in BS_RANGE:
        indices_dst = _gen_indices(CUDA_CACHE_SIZE, bs)
        indices_src = _gen_indices(HOST_CACHE_SIZE, bs)
        MEM = 2 * bs * ITEM_BYTES

        def _run_kernel_h2d(impl):
            return impl(
                k_cache_dst=cuda_cache[0],
                v_cache_dst=cuda_cache[1],
                indices_dst=indices_dst,
                k_cache_src=host_cache[0],
                v_cache_src=host_cache[1],
                indices_src=indices_src,
                item_bytes=ITEM_BYTES,
                block_quota=BLOCK_QUOTA,
            )

        our_h2d_dur = perf(lambda: _run_kernel_h2d(jit_hicache_impl))
        ref_h2d_dur = perf(lambda: _run_kernel_h2d(ref_hicache_impl))
        print(
            f"{bs = :6d}, H->D",
            f"| aot {_calc_tput(ref_h2d_dur):<6.2f} GB/s",
            f"| jit {_calc_tput(our_h2d_dur):<6.2f} GB/s",
            f"| {_gain_str(ref_h2d_dur, our_h2d_dur)}",
        )

    print("=" * 70)

    for bs in BS_RANGE:
        indices_dst = _gen_indices(HOST_CACHE_SIZE, bs)
        indices_src = _gen_indices(CUDA_CACHE_SIZE, bs)
        MEM = 2 * bs * ITEM_BYTES

        def _run_kernel_d2h(impl):
            return impl(
                k_cache_dst=host_cache[0],
                v_cache_dst=host_cache[1],
                indices_dst=indices_dst,
                k_cache_src=cuda_cache[0],
                v_cache_src=cuda_cache[1],
                indices_src=indices_src,
                item_bytes=ITEM_BYTES,
                block_quota=BLOCK_QUOTA,
            )

        our_d2h_dur = perf(lambda: _run_kernel_d2h(jit_hicache_impl))
        ref_d2h_dur = perf(lambda: _run_kernel_d2h(ref_hicache_impl))
        print(
            f"{bs = :6d}, D->H",
            f"| aot {_calc_tput(ref_d2h_dur):<6.2f} GB/s",
            f"| jit {_calc_tput(our_d2h_dur):<6.2f} GB/s",
            f"| {_gain_str(ref_d2h_dur, our_d2h_dur)}",
        )

    print("=" * 70)


def main() -> None:
    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)

    BUF_SIZE = 1024 * 1024 * 1024
    cuda_mem = torch.empty(BUF_SIZE, dtype=torch.uint8, device="cuda")
    host_mem = torch.empty(BUF_SIZE, dtype=torch.uint8, device="cpu", pin_memory=True)

    # test peak bandwidth
    tic.record()
    cuda_mem.copy_(host_mem, non_blocking=True)
    toc.record()
    toc.synchronize()
    dur = tic.elapsed_time(toc)
    print(f"Peak H->D Bandwidth: {(BUF_SIZE / (1024**3)) / (dur / 1000):.2f} GB/s")

    tic.record()
    host_mem.copy_(cuda_mem, non_blocking=True)
    toc.record()
    toc.synchronize()
    dur = tic.elapsed_time(toc)
    print(f"Peak D->H Bandwidth: {(BUF_SIZE / (1024**3)) / (dur / 1000):.2f} GB/s")

    for block_quota in [1, 2, 3, 4]:
        for cache_item_size in [128, 256, 512, 1024]:
            args = HicacheBenchArgs(
                cache_item_size=cache_item_size,
                dtype=torch.float16,
                block_quota=block_quota,
            )
            test_hicache_kernel(args)


if __name__ == "__main__":
    main()
