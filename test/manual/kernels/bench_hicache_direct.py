"""HiCache L2 transfers on GLM-5.2 shapes, as a chain of serialised radix-node steps.

Under ``--hicache-write-policy write_back`` with serialised load-back, each radix
node is loaded on its own: evict (a synchronous D2H write-back), then load (a
synchronous H2D), then the next node, with no forward in flight. This times K such
steps per node size through the submission path the controller uses: index moves,
then ``L2TransferEngine`` (or the raw op for the all-layer load the engine does not
offer), then the completion-event sync.

Cases per node size P (pages per step):
  cold   H2D only, distinct host pages per step, device L2 flushed before each step
  warm   H2D only, the same pages every step, no flush
  evict  D2H of P pages, then H2D of P pages, per step (cold)

Arms:
  direct              copy engine over page_first_direct
  direct_kernel       page-row kernels through the host pools, one launch per layer
  direct_kernel_all(raw)  the same kernels loading every layer in one launch per
                      pool, as raw ops (no L2TransferEngine, no per-submission events)
  kernel/q<n>         page_first TMA load + staged write-back, n CTAs

Host indices are pageable or pinned (``--pin``), since a non_blocking upload of
pageable indices may block the submitting thread.

    python test/manual/kernels/bench_hicache_direct.py --layers 20 78 --check
"""

import argparse
import ctypes
import json
import platform
import statistics
import subprocess
import time
from typing import Callable, NamedTuple

import torch

import sglang.kernels.ops.kvcache.hicache as hicache_ops
from sglang.kernels.ops.kvcache.hicache_direct import (
    make_hicache_direct_tables,
    transfer_hicache_direct,
)
from sglang.kernels.ops.kvcache.page_row import page_row_host_link_coherent
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import (
    prepare_host_pool_config,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import get_parallel

PAGE_SIZE = 64
KV_ROW_BYTES = PAGE_SIZE * 576
INDEXER_ROW_BYTES = PAGE_SIZE * 132
# GLM-5.2 (index_topk_freq 4): 7 indexer layers on the first 20-layer PP4 stage,
# 21 of 78 on a DEP rank.
GLM52_INDEXER_LAYERS = {20: 7, 78: 21}


class Pools(NamedTuple):
    target: DSATokenToKVPool
    kv: MLATokenToKVPoolHost
    indexer: DSAIndexerPoolHost

    def destroy(self):
        self.kv.destroy()
        self.indexer.destroy()

    def transfers(self, h, d) -> list[L2Transfer]:
        # The indexer owns a subset of layers; it skips the rest itself.
        return [
            L2Transfer(self.kv, self.target, h, d),
            L2Transfer(
                self.indexer, self.target, h, d, layer_mapper=lambda layer: layer
            ),
        ]


def _build(layers, indexer_layers, device_tokens, host_tokens, layout) -> Pools:
    # Spread the indexer layers evenly, as shared-topk models interleave them.
    owned = {round(i * layers / indexer_layers) for i in range(indexer_layers)}
    target = DSATokenToKVPool(
        size=device_tokens,
        page_size=PAGE_SIZE,
        kv_lora_rank=512,
        dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=64,
        layer_num=layers,
        device="cuda",
        enable_memory_saver=False,
        kv_cache_dim=576,
        index_head_dim=128,
        skip_topk_layers=[i not in owned for i in range(layers)],
    )
    kv_host = MLATokenToKVPoolHost(
        buffer_info=target.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=host_tokens / device_tokens,
        host_size=0,
        page_size=PAGE_SIZE,
        layout=layout,
        pin_memory=True,
        device="cpu",
        override_kv_cache_dim=target.kv_cache_dim,
        device_capacity_tokens=target.size,
        start_layer=target.start_layer,
        end_layer=target.end_layer,
        target_layer_num=target.layer_num,
    )
    indexer_config = prepare_host_pool_config(
        root=target,
        full_layer_mapping={i: i for i in range(target.layer_num)},
        transfer_layer_id_max=target.layer_num,
        transfer_page_size=kv_host.page_size,
    ).sidecars[0]
    indexer_host = DSAIndexerPoolHost(
        indexer_config.buffer_info,
        layer_mapping=indexer_config.device_to_host_layer,
        target_device_layer_num=target.layer_num,
        num_host_pages=kv_host.page_num,
        transfer_page_size=kv_host.page_size,
        layout=kv_host.layout,
        buffer_device=target.device,
        dtype=target.store_dtype,
        layer_sharded=target.layer_shard_enabled,
    )
    return Pools(target, kv_host, indexer_host)


class Step(NamedTuple):
    """One synchronous transfer: wall and submit time on the CPU, GPU time between events."""

    wall_ms: float
    submit_ms: float
    gpu_ms: float


class Arm(NamedTuple):
    name: str
    write_back: Callable[[torch.Tensor, torch.Tensor], Step]
    load: Callable[[torch.Tensor, torch.Tensor], Step]


def _upload(host_indices):
    """HiCacheController.move_indices for the GPU-index backends."""
    return host_indices.to("cuda", non_blocking=True)


def _engine_arm(pools: Pools, backend: str, name: str, staged_write_cpu=False) -> Arm:
    engine = L2TransferEngine(backend)
    layers = pools.target.layer_num

    def move(h, d):
        if backend == "direct":
            return h, d.cpu()
        return _upload(h), d

    def finish(t0, t1, completion):
        completion.finish_event.synchronize()
        t2 = time.perf_counter()
        gpu = completion.start_event.elapsed_time(completion.finish_event)
        return Step((t2 - t0) * 1e3, (t1 - t0) * 1e3, gpu)

    def write_back(h, d):
        t0 = time.perf_counter()
        # HiCacheController._move_write_operation keeps CPU host indices for the
        # staged page_first write-back.
        h2, d2 = (h, d) if staged_write_cpu else move(h, d)
        completion = engine.submit_device_to_host(pools.transfers(h2, d2))
        return finish(t0, time.perf_counter(), completion)

    # Preallocated and recorded once, as LayerLoadingEvent reuses its event sets:
    # an event's first record() initialises it, which would bill the per-layer path.
    start = torch.cuda.Event()
    events = [torch.cuda.Event() for _ in range(layers)]
    for event in (start, *events):
        event.record()

    def load(h, d):
        t0 = time.perf_counter()
        h2, d2 = move(h, d)
        start.record()
        completion = engine.submit_host_to_device(
            pools.transfers(h2, d2),
            transfer_layer_id_max=layers,
            start_event=start,
            on_layer_done=lambda layer: events[layer].record(),
        )
        return finish(t0, time.perf_counter(), completion)

    return Arm(name, write_back, load)


def _all_layer_arm(pools: Pools) -> Arm:
    """direct_kernel with one launch per pool in both directions, around the raw op
    with the controller's index move and stream discipline."""
    tables = [
        make_hicache_direct_tables(pools.kv.kv_buffer, pools.kv.device_data_ptrs),
        make_hicache_direct_tables(
            pools.indexer.index_k_with_scale_buffer, pools.indexer.index_k_device_ptrs
        ),
    ]
    d2h, h2d = torch.cuda.Stream(), torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()

    def timed(stream, h, d, load):
        t0 = time.perf_counter()
        hg = _upload(h)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            start.record()
            for t in tables:
                transfer_hicache_direct(t, hg, d, page_size=PAGE_SIZE, load=load)
            end.record()
        t1 = time.perf_counter()
        end.synchronize()
        t2 = time.perf_counter()
        return Step((t2 - t0) * 1e3, (t1 - t0) * 1e3, start.elapsed_time(end))

    return Arm(
        "direct_kernel_all(raw)",
        lambda h, d: timed(d2h, h, d, False),
        lambda h, d: timed(h2d, h, d, True),
    )


def _token_indices(pages: torch.Tensor) -> torch.Tensor:
    return (pages[:, None] * PAGE_SIZE + torch.arange(PAGE_SIZE)).reshape(-1)


def _check(arm: Arm, pools: Pools, host_pages, device_pages, gen):
    """Byte-exact round trip through a page_first_direct arm."""
    dp = device_pages.cuda()
    kv_layers = pools.kv.layer_num
    idx_layers = pools.indexer.index_k_device_ptrs.numel()
    host_rows = (
        pools.kv.kv_buffer.view(torch.uint8).view(-1, kv_layers, KV_ROW_BYTES),
        pools.indexer.index_k_with_scale_buffer.view(torch.uint8).view(
            -1, idx_layers, INDEXER_ROW_BYTES
        ),
    )
    device_rows = (
        [
            b.view(torch.uint8).view(-1, KV_ROW_BYTES)
            for b in pools.kv.device_kv_buffers
        ],
        [
            b.view(torch.uint8).view(-1, INDEXER_ROW_BYTES)
            for b in pools.indexer.packed_device_index_buffers
        ],
    )
    for rows in device_rows:
        for r in rows:
            r[dp] = torch.randint(
                0, 256, (len(dp), r.shape[1]), dtype=torch.uint8, generator=gen
            ).cuda()
    expected = [torch.stack([r[dp].cpu() for r in rows], 1) for rows in device_rows]
    for h in host_rows:
        h[host_pages] = 0
    h_idx, d_idx = _token_indices(host_pages), _token_indices(device_pages).cuda()
    arm.write_back(h_idx, d_idx)
    for h, e, tag in zip(host_rows, expected, ("kv", "idx")):
        assert torch.equal(h[host_pages], e), f"{arm.name}: {tag} write-back mismatch"
    for rows in device_rows:
        for r in rows:
            r[dp] = 0
    arm.load(h_idx, d_idx)
    for rows, e, tag in zip(device_rows, expected, ("kv", "idx")):
        got = torch.stack([r[dp].cpu() for r in rows], 1)
        assert torch.equal(got, e), f"{arm.name}: {tag} load mismatch"


def _chain(
    arm: Arm, case: str, host_pages, device_pages, pin: bool, l2_flush, steps: int
):
    """Median per-step load (and write-back, for `evict`) over a chain of node steps."""
    loads, writes = [], []
    for k in range(steps + 1):  # step 0 warms caches and allocators
        h = _token_indices(host_pages[0 if case == "warm" else k])
        h = h.pin_memory() if pin else h
        d = _token_indices(device_pages[k % len(device_pages)]).cuda()
        if case != "warm":
            l2_flush.zero_()
        torch.cuda.synchronize()
        w = arm.write_back(h, d) if case == "evict" else None
        step = arm.load(h, d)
        if k:
            loads.append(step)
            if w is not None:
                writes.append(w)

    def median(xs):
        if not xs:
            return None
        return Step(
            *(statistics.median(getattr(x, f) for x in xs) for f in Step._fields)
        )

    return median(loads), median(writes)


def _environment(pools: Pools) -> dict:
    """What the numbers depend on beyond the arms: runtime path and NUMA placement."""
    driver = ctypes.c_int()
    ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(driver))
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver": driver.value,
        # The `direct` arm batches only when dlsym(RTLD_DEFAULT) finds this (transfer.cu).
        "cudaMemcpyBatchAsync_symbol": hasattr(
            ctypes.CDLL(None), "cudaMemcpyBatchAsync"
        ),
        "host_link_coherent": page_row_host_link_coherent(torch.cuda.current_device()),
        "machine": platform.machine(),
        "gpu_numa_node": _gpu_numa_node(f"GPU-{props.uuid}"),
        "host_pool_numa": _numa_of(pools.kv.kv_buffer.data_ptr()),
        "numactl_show": _cmd(["numactl", "--show"]),
    }


def _gpu_numa_node(gpu_id):
    bus = _cmd(
        ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", gpu_id]
    )
    if not bus:
        return None
    # nvidia-smi prints an 8-digit PCI domain; sysfs uses 4.
    domain, rest = bus.strip().lower().split(":", 1)
    try:
        with open(f"/sys/bus/pci/devices/{domain[-4:]}:{rest}/numa_node") as f:
            return int(f.read())
    except OSError:
        return None


def _numa_of(address: int):
    """Per-node page counts of the mapping that holds `address`, from numa_maps."""
    try:
        with open("/proc/self/numa_maps") as f:
            lines = f.readlines()
    except OSError:
        return None
    best = None
    for line in lines:
        start = int(line.split()[0], 16)
        if start <= address and (best is None or start > best[0]):
            best = (start, line)
    if best is None:
        return None
    words = (w.split("=") for w in best[1].split()[1:] if "=" in w)
    return {k: v for k, v in words if k.startswith("N")}


def _cmd(argv):
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _append(path, record):
    if path:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def _arms(pfd: Pools, pf, tma_quotas) -> list:
    """(TMA quota or None, Arm) pairs."""
    out = [
        (None, _engine_arm(pfd, "direct", "direct")),
        (None, _engine_arm(pfd, "direct_kernel", "direct_kernel")),
        (None, _all_layer_arm(pfd)),
    ]
    if pf is not None:
        staged = pf.kv.can_use_write_back_jit
        out += [
            (q, _engine_arm(pf, "kernel", f"kernel/q{q}", staged_write_cpu=staged))
            for q in tma_quotas
        ]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[20, 78], help="MLA layers per rank"
    )
    parser.add_argument(
        "--indexer-layers",
        type=int,
        nargs="+",
        default=None,
        help="indexer layers per rank, one per --layers (default: GLM-5.2's)",
    )
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 256, 1024],
        help="pages per node step",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cases", nargs="+", default=["cold", "warm", "evict"])
    parser.add_argument("--tma-quotas", type=int, nargs="*", default=[16, 152])
    parser.add_argument("--pin", nargs="+", default=["pageable", "pinned"])
    parser.add_argument(
        "--check", action="store_true", help="byte-check page_first_direct arms"
    )
    parser.add_argument("--json", help="append one JSON record per measurement here")
    args = parser.parse_args()
    indexer_layers = args.indexer_layers or [
        GLM52_INDEXER_LAYERS.get(n, n) for n in args.layers
    ]
    assert len(indexer_layers) == len(args.layers)

    gen = torch.Generator().manual_seed(0)
    l2_flush = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    max_pages = max(args.pages)
    device_tokens = 2 * max_pages * PAGE_SIZE + PAGE_SIZE
    host_tokens = (args.steps + 2) * max_pages * PAGE_SIZE
    env_printed = False
    print(
        f"{'L':>3} {'Li':>3} {'pages':>5} {'case':>5} {'pin':>8} {'arm':>22} {'load_ms':>8} "
        f"{'submit':>7} {'gpu_ms':>8} {'GB/s':>6} {'wb_ms':>8} {'wb_sub':>7} {'wbGB/s':>6}"
    )
    for layers, idx_layers in zip(args.layers, indexer_layers):
        pfd = _build(
            layers, idx_layers, device_tokens, host_tokens, "page_first_direct"
        )
        pfd.kv.prepare_io_backend("direct_kernel")
        pfd.indexer.prepare_io_backend("direct_kernel")
        pf = None
        if args.tma_quotas:
            pf = _build(layers, idx_layers, device_tokens, host_tokens, "page_first")
        if not env_printed:
            print("# env", json.dumps(_environment(pfd)), flush=True)
            env_printed = True
        arms = _arms(pfd, pf, args.tma_quotas)
        if args.check:
            hp = torch.randperm(pfd.kv.page_num, generator=gen)[:17]
            dp = torch.randperm(device_tokens // PAGE_SIZE - 1, generator=gen)[:17] + 1
            for quota, arm in arms[1:]:
                if quota is None:
                    _check(arm, pfd, hp, dp, gen)
            print(
                f"# check ok: {layers} layers, {idx_layers} indexer layers", flush=True
            )
        bytes_per_page = KV_ROW_BYTES * layers + INDEXER_ROW_BYTES * idx_layers

        for pages in args.pages:
            # Distinct host pages per step; device pages alternate between two sets.
            host_perm = torch.randperm(pfd.kv.page_num, generator=gen)
            host_pages = [
                host_perm[k * pages : (k + 1) * pages] for k in range(args.steps + 1)
            ]
            device_perm = (
                torch.randperm(device_tokens // PAGE_SIZE - 1, generator=gen) + 1
            )
            device_pages = [device_perm[k * pages : (k + 1) * pages] for k in range(2)]
            moved = pages * bytes_per_page
            for tma_quota, arm in arms:
                if tma_quota is not None:
                    # Read at call time: TMA load-back and the staged write-back.
                    hicache_ops.TMA_BLOCK_QUOTA = tma_quota
                    hicache_ops.DEFAULT_BLOCK_QUOTA = tma_quota
                for case in args.cases:
                    for pin in args.pin:
                        load, wb = _chain(
                            arm,
                            case,
                            host_pages,
                            device_pages,
                            pin == "pinned",
                            l2_flush,
                            args.steps,
                        )
                        wb_cols = ""
                        if wb is not None:
                            wb_gbps = moved / wb.wall_ms / 1e6
                            wb_cols = f"{wb.wall_ms:>8.3f} {wb.submit_ms:>7.3f} {wb_gbps:>6.1f}"
                        print(
                            f"{layers:>3} {idx_layers:>3} {pages:>5} {case:>5} {pin:>8} {arm.name:>22} "
                            f"{load.wall_ms:>8.3f} {load.submit_ms:>7.3f} {load.gpu_ms:>8.3f} "
                            f"{moved / load.wall_ms / 1e6:>6.1f} {wb_cols}",
                            flush=True,
                        )
                        _append(
                            args.json,
                            dict(
                                layers=layers,
                                indexer_layers=idx_layers,
                                pages=pages,
                                case=case,
                                pin=pin,
                                arm=arm.name,
                                bytes=moved,
                                load=load._asdict(),
                                write_back=wb and wb._asdict(),
                            ),
                        )
        pfd.destroy()
        if pf is not None:
            pf.destroy()
        pfd = pf = None
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with get_parallel().override(attn_dcp_size=1):
        main()
