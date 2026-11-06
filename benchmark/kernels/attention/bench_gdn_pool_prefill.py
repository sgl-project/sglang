import argparse
import gc
import inspect
import json
import math
import os
import statistics
import time
from types import SimpleNamespace

import sgl_kernel_npu  # noqa: F401
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import (
    AscendGDNAttnBackend,
)
from sglang.srt.mem_cache.memory_pool import MambaPool

MIB = 1024 * 1024


def summarize(samples):
    ordered = sorted(samples)
    return {
        "rounds": len(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[math.ceil(0.95 * len(samples)) - 1],
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


class PrefillCase:
    def __init__(self, workload, batch, length, layers, args):
        self.workload = workload
        self.batch = batch
        self.length = length
        self.layers = layers
        self.nk = args.key_heads
        self.nv = args.value_heads
        self.dk = args.head_dim
        self.device = torch.device("npu:0")
        torch.manual_seed(args.seed + batch * 1009 + length)
        total = batch * length
        self.q = torch.randn(
            total, self.nk, self.dk, dtype=torch.bfloat16, device=self.device
        )
        self.k = torch.randn_like(self.q)
        self.v = 0.1 * torch.randn(
            total, self.nv, self.dk, dtype=torch.bfloat16, device=self.device
        )
        self.q_norm = F.normalize(self.q.float(), dim=-1).to(torch.bfloat16)
        self.k_norm = F.normalize(self.k.float(), dim=-1).to(torch.bfloat16)
        self.g = F.logsigmoid(torch.randn(total, self.nv, device=self.device)).float()
        self.beta = torch.rand(total, self.nv, dtype=torch.bfloat16, device=self.device)
        self.actual = torch.full(
            (batch,), length, dtype=torch.int32, device=self.device
        )
        self.cu_seqlens = torch.arange(
            0, total + 1, length, dtype=torch.int32, device=self.device
        )
        self.h = torch.empty(
            batch * ((length + 63) // 64),
            self.nv,
            self.dk,
            self.dk,
            dtype=torch.bfloat16,
            device=self.device,
        )
        if workload == "cached":
            self.src = torch.full((batch,), 5, dtype=torch.int32, device=self.device)
        else:
            self.src = torch.arange(
                17, 17 + batch, dtype=torch.int32, device=self.device
            )
        self.dst = torch.arange(17, 17 + batch, dtype=torch.int32, device=self.device)
        self.has_initial = torch.full(
            (batch,), workload != "fresh", dtype=torch.bool, device=self.device
        )
        slots = max(64, 18 + batch)
        self.base = 0.05 * torch.randn(
            layers,
            slots,
            self.nv,
            self.dk,
            self.dk,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.conv_base = 0.1 * torch.randn(
            layers,
            slots,
            args.conv_kernel - 1,
            (2 * self.nk + self.nv) * self.dk,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.backend = object.__new__(AscendGDNAttnBackend)
        self.backend.actual_seq_lengths = self.actual
        self.scale = self.dk**-0.5

    def make_path(self, use_pool):
        owner = object.__new__(MambaPool)
        owner.mamba_cache = MambaPool.State(
            conv=[self.conv_base.clone()], temporal=self.base.clone()
        )
        owner.replayssm_write_pos = None
        owner.debug_memory_pool = False
        owner._slot_siblings = []
        return SimpleNamespace(use_pool=use_pool, owner=owner)

    def prepare(self, path, include_cow):
        pool = path.owner.mamba_cache.temporal
        if self.workload == "cached":
            if not include_cow and not path.use_pool:
                pool[:, self.dst] = pool[:, self.src]
        elif self.workload == "chunked":
            pool[:, self.dst] = self.base[:, self.dst]
        else:
            pool[:, self.dst] = 0

    def invoke(self, path, include_cow, backend, capture=False):
        if include_cow and self.workload == "cached":
            path.owner.copy_from(self.src, self.dst, copy_temporal=not path.use_pool)
        outputs = []
        chunk_states = []
        for layer in range(self.layers):
            pool = path.owner.mamba_cache.temporal[layer]
            if backend:
                out, final, h = self.backend.chunk_gdn(
                    self.q,
                    self.k,
                    self.v,
                    self.g,
                    self.beta,
                    pool,
                    self.src if path.use_pool else self.dst,
                    self.dst,
                    self.has_initial if path.use_pool else None,
                    self.cu_seqlens,
                    use_pool=path.use_pool,
                )
            elif path.use_pool:
                out = torch.ops.npu.chunk_gated_delta_rule_pool(
                    self.q_norm,
                    self.k_norm,
                    self.v,
                    pool,
                    self.src,
                    self.dst,
                    beta=self.beta,
                    actual_seq_lengths=self.actual,
                    scale=self.scale,
                    chunk_state=self.h,
                    g=self.g,
                    has_initial_state=self.has_initial,
                )
                final = None
                h = self.h
            else:
                out, final = torch.ops.npu.chunk_gated_delta_rule(
                    self.q_norm,
                    self.k_norm,
                    self.v,
                    beta=self.beta,
                    initial_state=pool[self.dst],
                    actual_seq_lengths=self.actual,
                    scale=self.scale,
                    chunk_state=self.h,
                    g=self.g,
                )
                h = self.h
            if final is not None:
                pool[self.dst] = final
            if capture:
                outputs.append(out)
                chunk_states.append(h.clone() if not backend else h)
            else:
                del out, final, h
        return outputs, chunk_states


def check_correctness(case, legacy, pooled, include_cow, backend):
    for path in (legacy, pooled):
        case.prepare(path, include_cow)
    expected_out, expected_h = case.invoke(legacy, include_cow, backend, capture=True)
    actual_out, actual_h = case.invoke(pooled, include_cow, backend, capture=True)
    torch.npu.synchronize()
    for actual, expected in zip(actual_out, expected_out, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(actual_h, expected_h, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for field in ("temporal", "conv"):
        actual = getattr(pooled.owner.mamba_cache, field)
        expected = getattr(legacy.owner.mamba_cache, field)
        if isinstance(actual, list):
            actual = actual[0]
            expected = expected[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if case.workload == "cached":
        for path in (legacy, pooled):
            torch.testing.assert_close(
                path.owner.mamba_cache.temporal[:, 5],
                case.base[:, 5],
                rtol=0,
                atol=0,
            )


def peak_allocated(case, path, include_cow, backend):
    case.prepare(path, include_cow)
    gc.collect()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    case.invoke(path, include_cow, backend)
    torch.npu.synchronize()
    return {
        "incremental_peak_bytes": torch.npu.max_memory_allocated() - before,
        "retained_bytes_after_call": torch.npu.memory_allocated() - before,
    }


def compare(case, include_cow, backend, args):
    paths = [case.make_path(False), case.make_path(True)]
    check_correctness(case, paths[0], paths[1], include_cow, backend)
    for _ in range(args.warmup):
        for path in paths:
            case.prepare(path, include_cow)
            case.invoke(path, include_cow, backend)
    torch.npu.synchronize()
    latencies = [[], []]
    for round_index in range(args.rounds):
        order = (0, 1) if round_index % 2 == 0 else (1, 0)
        for index in order:
            path = paths[index]
            case.prepare(path, include_cow)
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            case.invoke(path, include_cow, backend)
            torch.npu.synchronize()
            latencies[index].append((time.perf_counter_ns() - start) / 1e6)
    memory = [peak_allocated(case, path, include_cow, backend) for path in paths]
    old = summarize(latencies[0])
    new = summarize(latencies[1])
    if backend:
        name = (
            "backend_chunk_gdn_with_production_cow"
            if include_cow
            else "backend_chunk_gdn_without_prefix_cow"
        )
    else:
        name = "state_io_only_gather_oldop_scatter_vs_poolop"
    result = {
        "name": name,
        "workload": case.workload,
        "batch": case.batch,
        "suffix_tokens_per_request": case.length,
        "layers_per_timed_call": case.layers,
        "includes_prefix_cow": include_cow and case.workload == "cached",
        "correctness": "bitwise output, chunk_state, temporal pool and conv pool match",
        "legacy": old,
        "pool": new,
        "median_delta_ms": new["median_ms"] - old["median_ms"],
        "median_reduction_percent": 100 * (1 - new["median_ms"] / old["median_ms"]),
        "median_speedup": old["median_ms"] / new["median_ms"],
        "legacy_memory": memory[0],
        "pool_memory": memory[1],
        "incremental_peak_saved_bytes": memory[0]["incremental_peak_bytes"]
        - memory[1]["incremental_peak_bytes"],
    }
    print(
        f"{case.workload} B={case.batch} L={case.length} layers={case.layers} "
        f"{name}: legacy={old['median_ms']:.4f} ms, pool={new['median_ms']:.4f} ms, "
        f"reduction={result['median_reduction_percent']:.2f}%, "
        f"speedup={result['median_speedup']:.3f}x, "
        f"peak saved={result['incremental_peak_saved_bytes'] / MIB:.2f} MiB",
        flush=True,
    )
    del paths
    gc.collect()
    return result


def compare_cow(case, args):
    paths = [case.make_path(False), case.make_path(True)]

    def copy(path):
        path.owner.copy_from(case.src, case.dst, copy_temporal=not path.use_pool)

    for path in paths:
        copy(path)
    torch.npu.synchronize()
    for path in paths:
        if path.use_pool:
            torch.testing.assert_close(
                path.owner.mamba_cache.temporal, case.base, rtol=0, atol=0
            )
        else:
            torch.testing.assert_close(
                path.owner.mamba_cache.temporal[:, case.dst],
                case.base[:, case.src],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            path.owner.mamba_cache.conv[0][:, case.dst],
            case.conv_base[:, case.src],
            rtol=0,
            atol=0,
        )
    latencies = [[], []]
    for _ in range(args.warmup):
        for path in paths:
            copy(path)
    torch.npu.synchronize()
    for round_index in range(args.rounds):
        order = (0, 1) if round_index % 2 == 0 else (1, 0)
        for index in order:
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            copy(paths[index])
            torch.npu.synchronize()
            latencies[index].append((time.perf_counter_ns() - start) / 1e6)
    memory = []
    for path in paths:
        gc.collect()
        torch.npu.synchronize()
        before = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        copy(path)
        torch.npu.synchronize()
        memory.append(
            {
                "incremental_peak_bytes": torch.npu.max_memory_allocated() - before,
                "retained_bytes_after_call": torch.npu.memory_allocated() - before,
            }
        )
    old = summarize(latencies[0])
    new = summarize(latencies[1])
    result = {
        "name": "real_mamba_pool_copy_from_temporal_and_conv_vs_conv_only",
        "workload": "cached",
        "batch": case.batch,
        "layers_per_timed_call": case.layers,
        "legacy": old,
        "pool": new,
        "median_delta_ms": new["median_ms"] - old["median_ms"],
        "median_reduction_percent": 100 * (1 - new["median_ms"] / old["median_ms"]),
        "median_speedup": old["median_ms"] / new["median_ms"],
        "legacy_memory": memory[0],
        "pool_memory": memory[1],
        "incremental_peak_saved_bytes": memory[0]["incremental_peak_bytes"]
        - memory[1]["incremental_peak_bytes"],
    }
    print(
        f"CoW B={case.batch} layers={case.layers}: temporal+conv={old['median_ms']:.4f} ms, "
        f"conv-only={new['median_ms']:.4f} ms, "
        f"reduction={result['median_reduction_percent']:.2f}%, "
        f"speedup={result['median_speedup']:.3f}x, "
        f"peak saved={result['incremental_peak_saved_bytes'] / MIB:.2f} MiB",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="GDN pool-I/O microbenchmark; not an end-to-end model benchmark."
    )
    parser.add_argument("--sizes", nargs="+", default=["2x128"])
    parser.add_argument(
        "--workloads", nargs="+", default=["cached", "chunked", "fresh"]
    )
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument(
        "--mode", choices=["all", "core", "backend", "cow"], default="all"
    )
    parser.add_argument("--skip-backend-without-cow", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.warmup < 1 or args.rounds < 2 or args.layers < 1:
        raise ValueError("warmup, layers must be positive; rounds must be at least two")
    torch.npu.set_device("npu:0")
    result = {
        "schema": 1,
        "device": torch.npu.get_device_name(0),
        "logical_device": torch.npu.current_device(),
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "benchmark_arguments": vars(args),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "dtype": "bfloat16",
        "key_heads": args.key_heads,
        "value_heads": args.value_heads,
        "head_dim": args.head_dim,
        "backend_source": inspect.getfile(AscendGDNAttnBackend),
        "memory_pool_source": inspect.getfile(MambaPool),
        "scope": "kernel/backend microbenchmark only; not model-server or end-to-end performance",
        "timing": "one timed call per round with synchronize before and after; alternating path order",
        "memory": "torch.npu max_memory_allocated minus allocated bytes before one call; excludes resident pool",
        "state_preparation": "outside timed regions; cached total includes real COW only in the legacy timed region",
        "state_slot_bytes": args.value_heads * args.head_dim * args.head_dim * 2,
        "results": [],
    }
    for size in args.sizes:
        batch_text, length_text = size.split("x", maxsplit=1)
        batch, length = int(batch_text), int(length_text)
        if batch < 1 or length < 1:
            raise ValueError(size)
        for workload in args.workloads:
            if workload not in ("cached", "chunked", "fresh"):
                raise ValueError(workload)
            case = PrefillCase(workload, batch, length, args.layers, args)
            if args.mode in ("all", "core"):
                result["results"].append(compare(case, False, False, args))
            if args.mode in ("all", "backend"):
                if workload == "cached" and not args.skip_backend_without_cow:
                    result["results"].append(compare(case, False, True, args))
                result["results"].append(compare(case, True, True, args))
            if workload == "cached" and args.mode in ("all", "cow"):
                result["results"].append(compare_cow(case, args))
            del case
            gc.collect()
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
