r"""Compare full-layer and active-page NVFP4 prefill transfers on 2/4 GPUs.

Run inside an existing allocation (this script never requests GPUs)::

    python scripts/bench_dsa_nvfp4_prefill_pages.py \
        --world-sizes 2 4 --pool-pages 256 4096 65536 --active-pages 64 \
        --near-full-fraction 0.9 --output prefill-pages.json

Every mode reads the same pages. The ordinary NVFP4 pool is the correctness
reference. Timed steps include CPU page planning, metadata communication,
packing, broadcast, scatter, index gathering, NVFP4 dequantization, the real
H64 FlashMLA sparse-prefill consumer, and final device synchronization.
Initialization, correctness checks, and the inter-rank start barrier are not
timed. This is a local transfer/consumer benchmark, not model throughput.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import socket
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

PAGE_SIZE = 64
KV_ROW_BYTES = 416
INDEX_ROW_BYTES = 132
MODES = ("full_sync", "full_prefetch", "partial_prefetch")


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _command(argv: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _source_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = [
        "scripts/bench_dsa_nvfp4_prefill_pages.py",
        "python/sglang/srt/mem_cache/dsa_cache_layer_split.py",
        "python/sglang/srt/mem_cache/memory_pool.py",
        "python/sglang/srt/layers/attention/dsa/nvfp4_k_cache.py",
        "python/sglang/srt/layers/attention/dsa_backend.py",
        "python/sglang/srt/layers/attention/dsa/dsa_indexer.py",
        "python/sglang/srt/layers/attention/dsa/utils.py",
        "python/sglang/srt/model_executor/pool_configurator.py",
        "python/sglang/kernels/ops/kvcache/dsa_index_k_cache.py",
    ]
    git_diff = _command(["git", "diff", "HEAD", "--", *files], root)
    versions = {}
    for package in ("torch", "sglang", "sglang-kernel", "triton", "flashinfer-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    native_module = importlib.util.find_spec("sgl_kernel.flashmla_ops")
    native_path = (
        Path(native_module.origin) if native_module and native_module.origin else None
    )
    return {
        "git_head": _command(["git", "rev-parse", "HEAD"], root),
        "source_commit_env": os.environ.get("SOURCE_COMMIT"),
        "tracked_diff_sha256": (
            hashlib.sha256(git_diff.encode()).hexdigest()
            if git_diff is not None
            else None
        ),
        "file_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
        },
        "versions": versions,
        "flashmla_native": (
            {
                "path": str(native_path),
                "sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            }
            if native_path is not None and native_path.is_file()
            else None
        ),
        "torch_cuda": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _summary(values: list[float]) -> dict:
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p10": quantile(0.1),
        "p90": quantile(0.9),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _metadata(pool_pages: int, active_pages: int, world: int, rank: int, args):
    device = f"cuda:{rank}"
    generator = torch.Generator().manual_seed(args.seed + pool_pages + active_pages)
    pages_cpu = torch.randperm(pool_pages, generator=generator)[:active_pages] + 1
    # Production DSA metadata uses int32 physical page tables. Keep that dtype
    # here so the benchmark exercises the same gather address arithmetic.
    pages = pages_cpu.to(device=device, dtype=torch.int32)
    slots = (
        pages[:, None] * PAGE_SIZE + torch.arange(PAGE_SIZE, device=device)
    ).flatten()
    rows_per_request = args.pages_per_request * PAGE_SIZE
    requests = math.ceil(active_pages / args.pages_per_request)
    lengths_cpu = torch.tensor(
        [
            min(rows_per_request, slots.numel() - i * rows_per_request)
            for i in range(requests)
        ],
        dtype=torch.int32,
    )
    table = torch.zeros((requests, rows_per_request), dtype=torch.int32, device=device)
    table.flatten()[: slots.numel()] = slots.to(torch.int32)
    ends = lengths_cpu.to(torch.int64).cumsum(0).to(device=device) - 1
    reference_slots = torch.arange(PAGE_SIZE, PAGE_SIZE + slots.numel(), device=device)
    reference_pages = torch.arange(
        1, active_pages + 1, device=device, dtype=torch.int32
    )
    query_count = max(16, math.ceil(requests / world))
    request_ids = (rank + torch.arange(query_count, device=device) * world) % requests
    lengths = lengths_cpu.to(device=device)
    query_lengths = lengths[request_ids].clamp(max=2048)
    relative = torch.arange(2048, device=device)
    indices = request_ids[:, None] * rows_per_request + relative
    indices = torch.where(relative < query_lengths[:, None], indices, -1)
    query_generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    return SimpleNamespace(
        pages=pages,
        pages_cpu=pages_cpu,
        slots=slots,
        full_page_table=table,
        seq_lens_cpu=lengths_cpu,
        extend_seq_lens_cpu=[1] * requests,
        out_cache_loc=slots[ends],
        # The consumer gathers the same flattened batch on every rank.
        consumer_lens=torch.tensor([slots.numel()], device=device, dtype=torch.int32),
        reference_slots=reference_slots,
        reference_pages=reference_pages,
        q=torch.randn(
            query_count,
            64,
            576,
            generator=query_generator,
            device=device,
            dtype=torch.bfloat16,
        ),
        topk=indices.to(torch.int32).unsqueeze(1).contiguous(),
        topk_length=query_lengths.to(torch.int32),
        requests=requests,
    )


def _initialize(pool, reference, meta, layers: int, args) -> None:
    """Use the unchanged production writer for both physical page layouts."""
    for layer_id in range(layers):
        layer = SimpleNamespace(layer_id=layer_id)
        global_scale = 0.5 + layer_id * 0.375
        pool.set_mla_kv_global_scale(layer_id, global_scale)
        reference.set_mla_kv_global_scale(layer_id, global_scale)
        generator = torch.Generator(device=pool.device).manual_seed(
            args.seed + layer_id
        )
        for start in range(0, meta.slots.numel(), 16384):
            stop = min(start + 16384, meta.slots.numel())
            count = stop - start
            nope = torch.randn(
                count,
                1,
                512,
                generator=generator,
                device=pool.device,
                dtype=torch.bfloat16,
            )
            rope = torch.randn(
                count,
                1,
                64,
                generator=generator,
                device=pool.device,
                dtype=torch.bfloat16,
            )
            index_keys = torch.randn(
                count, 128, generator=generator, device=pool.device
            ).to(torch.float8_e4m3fn)
            index_scales = (
                torch.rand(count, 1, generator=generator, device=pool.device) + 0.25
            )
            for current, slots in (
                (pool, meta.slots[start:stop]),
                (reference, meta.reference_slots[start:stop]),
            ):
                current.set_mla_kv_buffer(layer, slots, nope, rope)
                current.index_key_cache.store_quantized(
                    layer_id, slots, index_keys, index_scales
                )
    torch.cuda.synchronize()


def _begin(pool, meta, mode: str) -> None:
    if mode == "partial_prefetch":
        pool.begin_prefill_page_plan(
            meta.full_page_table,
            meta.seq_lens_cpu,
            meta.extend_seq_lens_cpu,
            meta.out_cache_loc,
        )
    else:
        pool.clear_prefill_page_plan()


def _read_and_consume(pool, meta, layer_id: int, mode: str):
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd
    from sglang.kernels.ops.kvcache import dsa_index_k_cache
    from sglang.srt.layers.attention.dsa.nvfp4_k_cache import (
        dequantize_nvfp4_k_cache_paged,
    )

    if mode == "reference":
        index_buffer = pool.index_key_cache.get_buffer(layer_id)
        pages, slots = meta.reference_pages, meta.reference_slots
    elif mode == "partial_prefetch":
        pool.index_key_cache.prepare_prefill_buffer(layer_id)
        index_buffer = pool.index_key_cache.get_prefill_buffer(layer_id)
        pages, slots = meta.pages, meta.slots
    else:
        index_buffer = pool.index_key_cache.get_buffer(layer_id)
        if mode == "full_prefetch":
            pool.prefetch_kv_buffer(layer_id)
        pages, slots = meta.pages, meta.slots

    # This production index gather can overlap the async main-KV transfer.
    index_keys, index_scales = dsa_index_k_cache.GetKAndS.execute(
        pool,
        index_buffer,
        page_indices=pages.view(1, -1),
        seq_len_tensor=meta.consumer_lens,
        seq_len_sum=slots.numel(),
        max_seq_len=slots.numel(),
    )
    raw = (
        pool.get_prefill_key_buffer(layer_id)
        if mode == "partial_prefetch"
        else pool.get_key_buffer(layer_id)
    )
    decoded = dequantize_nvfp4_k_cache_paged(
        raw, slots, pool.get_mla_kv_global_scale(layer_id), dtype=torch.bfloat16
    )
    attention = flash_mla_sparse_fwd(
        q=meta.q,
        kv=decoded,
        indices=meta.topk,
        sm_scale=576**-0.5,
        d_v=512,
        topk_length=meta.topk_length,
    )[0]
    return raw, index_buffer, index_keys, index_scales, decoded, attention


def _check(pool, reference, meta, layers: int, mode: str) -> None:
    _begin(pool, meta, mode)
    # A prior full transfer must not hide a missing page in the partial path.
    pool.remote_kv_buffer.fill_(0xA5)
    pool.index_key_cache.remote_buffer.fill_(0x5A)
    if mode == "partial_prefetch":
        torch.testing.assert_close(
            pool.prefill_page_ids[: pool.prefill_num_pages],
            meta.pages.sort().values.to(torch.int64),
            rtol=0,
            atol=0,
        )
    for layer_id in range(layers):
        actual = _read_and_consume(pool, meta, layer_id, mode)
        expected = _read_and_consume(reference, meta, layer_id, "reference")
        # Check every consumed packed page, including all FP8 index scale bytes.
        for start in range(0, meta.pages.numel(), 1024):
            pages = meta.pages[start : start + 1024]
            reference_pages = meta.reference_pages[start : start + 1024]
            torch.testing.assert_close(
                actual[0].view(-1, PAGE_SIZE * KV_ROW_BYTES)[pages],
                expected[0].view(-1, PAGE_SIZE * KV_ROW_BYTES)[reference_pages],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual[1][pages], expected[1][reference_pages], rtol=0, atol=0
            )
        for got, wanted in zip(actual[2:5], expected[2:5]):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        # Retain the existing native NVFP4 attention tolerance.
        torch.testing.assert_close(actual[5], expected[5], atol=8e-4, rtol=2.01 / 128)
        del actual, expected
    torch.cuda.synchronize()


def _step(pool, meta, layers: int, mode: str) -> float:
    plan_start = time.perf_counter()
    _begin(pool, meta, mode)
    plan_ms = (time.perf_counter() - plan_start) * 1000
    for layer_id in range(layers):
        result = _read_and_consume(pool, meta, layer_id, mode)
        del result
    return plan_ms


def _time_step(pool, meta, layers: int, mode: str) -> dict:
    torch.cuda.synchronize()
    torch.distributed.barrier()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    wall_start = time.perf_counter()
    start.record()
    plan_ms = _step(pool, meta, layers, mode)
    end.record()
    torch.cuda.synchronize()
    return {
        "wall_ms": (time.perf_counter() - wall_start) * 1000,
        "cuda_elapsed_ms": start.elapsed_time(end),
        "plan_host_ms": plan_ms,
    }


def _byte_counts(
    pool_pages: int, active_pages: int, layers: int, world: int, mode: str
):
    pages = active_pages if mode == "partial_prefetch" else pool_pages + 1
    main = pages * PAGE_SIZE * KV_ROW_BYTES * layers
    index = pages * PAGE_SIZE * INDEX_ROW_BYTES * layers
    plan_ids = active_pages * 8 if mode == "partial_prefetch" else 0
    return {
        "main_kv_broadcast_payload_bytes": main,
        "index_broadcast_payload_bytes": index,
        "plan_id_broadcast_payload_bytes": plan_ids,
        "broadcast_payload_bytes": main + index + plan_ids,
        "logical_receiver_bytes": (main + index + plan_ids) * (world - 1),
        "main_bytes_consumed_per_rank": active_pages
        * PAGE_SIZE
        * KV_ROW_BYTES
        * layers,
        "index_bytes_consumed_per_rank": active_pages
        * PAGE_SIZE
        * INDEX_ROW_BYTES
        * layers,
        "note": "Payload sizes exclude NCCL protocol and small plan-object framing; logical receiver bytes are not measured fabric traffic.",
    }


@torch.inference_mode()
def _worker(rank: int, world: int, args, path: str, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
    )
    torch.cuda.set_device(rank)
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.runtime_context import get_parallel
    from sglang.test.test_utils import publish_build_topology

    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="nccl",
    )
    publish_build_topology(tp_size=world, attn_cp_size=world, world_rank=rank)
    initialize_model_parallel()
    group = get_parallel().attn_cp_group.cpu_group
    props = torch.cuda.get_device_properties(rank)
    identity = {
        "rank": rank,
        "hostname": socket.gethostname(),
        "cuda_device": rank,
        "name": props.name,
        "uuid": str(props.uuid),
        "total_memory_bytes": props.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(rank)),
    }
    identities = [None] * world
    torch.distributed.all_gather_object(identities, identity, group=group)
    if len({item["uuid"] for item in identities}) != world:
        raise RuntimeError("Every CP rank must use a distinct physical GPU")
    document = {
        "schema_version": 1,
        "completed": False,
        "cp_size": world,
        "started_utc": datetime.now(UTC).isoformat(),
        "source": _source_identity(),
        "gpus": identities,
        "nvidia_smi_csv": _command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "cases": [],
    }
    if rank == 0:
        _write_json(Path(path), document)
    layers = world + 1
    # A later config scope can re-project default widths without rebuilding the
    # process groups. State the complete TP/attention/MoE factorization here,
    # then check the live ranks instead of overriding or masking them.
    with get_parallel().override(
        tp_size=world,
        attn_tp_size=1,
        attn_dp_size=1,
        attn_cp_size=world,
        attn_dcp_size=1,
        moe_tp_size=world,
        moe_ep_size=1,
        moe_dp_size=1,
    ):
        parallel = get_parallel()
        assert parallel.tp_rank == parallel.attn_cp_rank == rank
        assert parallel.attn_tp_rank == parallel.attn_dp_rank == 0
        assert (
            parallel.tp_group.world_size == parallel.attn_cp_group.world_size == world
        )
        for pool_pages in args.pool_pages:
            active_counts = [
                ("sparse", args.active_pages),
                (
                    "near_full",
                    max(
                        args.active_pages,
                        math.ceil(pool_pages * args.near_full_fraction),
                    ),
                ),
            ]
            for label, active_pages in active_counts:
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "initialize_and_validate",
                                "cp": world,
                                "pool_pages": pool_pages,
                                "access": label,
                                "active_pages": active_pages,
                            }
                        ),
                        flush=True,
                    )
                meta = _metadata(pool_pages, active_pages, world, rank, args)
                kwargs = {
                    "size": pool_pages * PAGE_SIZE,
                    "page_size": PAGE_SIZE,
                    "kv_lora_rank": 512,
                    "dtype": torch.float4_e2m1fn_x2,
                    "qk_rope_head_dim": 64,
                    "layer_num": layers,
                    "device": f"cuda:{rank}",
                    "index_head_dim": 128,
                    "enable_memory_saver": False,
                    "kv_cache_dim": KV_ROW_BYTES,
                }
                pool = LayerSplitDSATokenToKVPool(
                    **kwargs,
                    layer_shard_rank=rank,
                    layer_shard_size=world,
                    unified_decode=True,
                )
                reference = DSATokenToKVPool(
                    **{**kwargs, "size": active_pages * PAGE_SIZE}
                )
                _initialize(pool, reference, meta, layers, args)
                rank_correctness = {}
                for mode in MODES:
                    _check(pool, reference, meta, layers, mode)
                    rank_correctness[mode] = True
                del reference
                gc.collect()
                torch.cuda.empty_cache()
                for _ in range(args.warmup):
                    for mode in MODES:
                        _step(pool, meta, layers, mode)
                torch.cuda.synchronize()
                local = {mode: [] for mode in MODES}
                for sample in range(args.samples):
                    order = MODES[sample % len(MODES) :] + MODES[: sample % len(MODES)]
                    for order_index, mode in enumerate(order):
                        timing = _time_step(pool, meta, layers, mode)
                        timing.update(sample=sample, order_index=order_index)
                        local[mode].append(timing)
                gathered = [None] * world
                torch.distributed.all_gather_object(
                    gathered,
                    {
                        "rank": rank,
                        "correctness": rank_correctness,
                        "samples": local,
                        "pool_allocated_bytes": int(pool.get_kv_size_bytes()),
                        "transfer_workspace_bytes": {
                            name: int(value)
                            for name, value in pool.get_prefill_transfer_allocation_bytes().items()
                        },
                    },
                    group=group,
                )
                if rank == 0:
                    record = {
                        "access": label,
                        "usable_pool_pages": pool_pages,
                        "physical_pool_pages_including_padding": pool_pages + 1,
                        "active_pages": active_pages,
                        "active_tokens": active_pages * PAGE_SIZE,
                        "layers": layers,
                        "requests": meta.requests,
                        "queries_per_rank": meta.q.shape[0],
                        "query_heads": 64,
                        "pages_per_request": args.pages_per_request,
                        "page_selection_sha256": hashlib.sha256(
                            meta.pages_cpu.numpy().tobytes()
                        ).hexdigest(),
                        "correctness": {
                            "packed_main_bytes": "exact",
                            "packed_index_bytes": "exact",
                            "dequantized_rows": "exact",
                            "index_gather_outputs": "exact",
                            "attention_atol": 8e-4,
                            "attention_rtol": 2.01 / 128,
                            "all_ranks_passed": True,
                        },
                        "ranks": gathered,
                        "modes": {},
                    }
                    for mode in MODES:
                        maxima = [
                            max(
                                item["samples"][mode][i]["wall_ms"] for item in gathered
                            )
                            for i in range(args.samples)
                        ]
                        record["modes"][mode] = {
                            "bytes_per_step": _byte_counts(
                                pool_pages, active_pages, layers, world, mode
                            ),
                            "max_rank_wall_ms_samples": maxima,
                            "max_rank_wall_ms": _summary(maxima),
                        }
                    document["cases"].append(record)
                    _write_json(Path(path), document)
                    print(
                        json.dumps(
                            {
                                "cp": world,
                                "pool_pages": pool_pages,
                                "access": label,
                                "active_pages": active_pages,
                                "median_wall_ms": {
                                    mode: record["modes"][mode]["max_rank_wall_ms"][
                                        "median"
                                    ]
                                    for mode in MODES
                                },
                            }
                        ),
                        flush=True,
                    )
                pool.clear_prefill_page_plan()
                del pool, meta
                gc.collect()
                torch.cuda.empty_cache()
                torch.distributed.barrier()
    if rank == 0:
        document.update(completed=True, finished_utc=datetime.now(UTC).isoformat())
        _write_json(Path(path), document)
    torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--world-sizes", nargs="+", type=int, default=[2, 4], choices=[2, 4]
    )
    parser.add_argument("--pool-pages", nargs="+", type=int, default=[256, 4096, 65536])
    parser.add_argument("--active-pages", type=int, default=64)
    parser.add_argument("--near-full-fraction", type=float, default=0.9)
    parser.add_argument("--pages-per-request", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.world_sizes)) != len(args.world_sizes) or len(
        set(args.pool_pages)
    ) != len(args.pool_pages):
        parser.error("world sizes and pool pages must not contain duplicates")
    if (
        min(args.pool_pages) < args.active_pages
        or args.active_pages < 1
        or args.pages_per_request < 1
    ):
        parser.error(
            "pool pages >= active pages >= 1 and pages per request >= 1 are required"
        )
    if not 0 < args.near_full_fraction <= 1 or args.samples < 3 or args.warmup < 1:
        parser.error("require 0 < near-full fraction <= 1, samples >= 3, warmup >= 1")
    outputs = [
        args.output.with_name(f"{args.output.stem}.cp{world}.json")
        for world in args.world_sizes
    ]
    if any(path.exists() for path in [args.output, *outputs]):
        parser.error("output already exists; use a new path to preserve prior results")
    if torch.cuda.device_count() < max(args.world_sizes):
        parser.error("not enough visible GPUs for the requested CP group")
    if any(
        torch.cuda.get_device_capability(i)[0] != 10
        for i in range(max(args.world_sizes))
    ):
        parser.error("this benchmark requires SM100-family GPUs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for world, path in zip(args.world_sizes, outputs):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(world, args, str(path), port), nprocs=world, join=True)
    _write_json(
        args.output,
        {
            "schema_version": 1,
            "completed": True,
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "timing_contract": "Wall time includes CPU plan + metadata + pack + broadcast + scatter + index gather + original NVFP4 dequant + H64 FlashMLA sparse-prefill + device wait. The start barrier is excluded. Modes rotate within each sample round. CUDA elapsed time is auxiliary and can include CPU submission gaps. No model, dataset, or full-model warmup is used.",
            "runs": [json.loads(path.read_text()) for path in outputs],
        },
    )


if __name__ == "__main__":
    main()
