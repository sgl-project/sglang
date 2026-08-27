"""Benchmark the SM120 QSA direct-paged sparse decode path."""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import subprocess
import sys

import torch

from sglang.srt.utils import is_sm120_supported

# Qwen/Qwen3.8-Flash-Next-FP8 full-attention shapes at TP=1.
NUM_Q_HEADS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 256
COMPRESS_RATIO = 4
TOKEN_TOPK = 2048
FINAL_TOPK = TOKEN_TOPK + COMPRESS_RATIO - 1
SOFTMAX_SCALE = HEAD_DIM**-0.5
TRTLLM_PAGE = 64


@functools.cache
def _sparse_ops():
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        qsa_sparse_decode_triton,
        qwen_sparse_fa2_cu_seqlens_triton,
        qwen_sparse_kv_extraction_compact_triton,
        qwen_sparse_valid_counts_triton,
    )

    return {
        "decode": qsa_sparse_decode_triton,
        "cu_seqlens": qwen_sparse_fa2_cu_seqlens_triton,
        "compact": qwen_sparse_kv_extraction_compact_triton,
        "valid_counts": qwen_sparse_valid_counts_triton,
    }


def _make_case(rows: int, topk: int = FINAL_TOPK):
    torch.manual_seed(20260827 + rows)
    device = torch.device("cuda")
    max_seq_len = topk + 64
    req_to_token = torch.randperm(
        rows * max_seq_len, dtype=torch.int32, device=device
    ).reshape(rows, max_seq_len)
    req_indices = torch.arange(rows, dtype=torch.int32, device=device)
    seq_lens = torch.full((rows,), max_seq_len, dtype=torch.int32, device=device)
    seq_lens -= torch.arange(rows, dtype=torch.int32, device=device) % 17

    counts = torch.full((rows,), topk, dtype=torch.int64, device=device)
    if rows > 1:
        counts[1] = topk // 3
    if rows > 2:
        counts[2] = 7
    indices = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        count = int(counts[row])
        selected = torch.randperm(int(seq_lens[row]), dtype=torch.int32, device=device)[
            :count
        ]
        indices[row, :count] = torch.sort(selected).values

    slots = rows * max_seq_len
    q = torch.randn(rows, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(slots, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    return {
        "q": q,
        "k": k,
        "v": torch.randn_like(k),
        "req_to_token": req_to_token,
        "req_indices": req_indices,
        "indices": indices,
        "seq_lens": seq_lens,
        "rows": rows,
        "topk": topk,
    }


def _physical_slots(case):
    indices = case["indices"]
    valid = (indices >= 0) & (indices < case["seq_lens"][:, None])
    safe = indices.clamp(min=0, max=case["req_to_token"].shape[1] - 1).long()
    slots = case["req_to_token"][case["req_indices"].long()[:, None], safe]
    return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)


def _reference_fp32(case):
    slots = _physical_slots(case)
    group_size = NUM_Q_HEADS // NUM_KV_HEADS
    output = torch.zeros(
        case["rows"],
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.float32,
        device=case["q"].device,
    )
    for row in range(case["rows"]):
        selected = slots[row]
        selected = selected[selected >= 0].long()
        if selected.numel() == 0:
            continue
        keys = case["k"][selected].float().permute(1, 0, 2)
        values = case["v"][selected].float().permute(1, 0, 2)
        query = case["q"][row].float().view(NUM_KV_HEADS, group_size, HEAD_DIM)
        logits = torch.einsum("hgd,hnd->hgn", query, keys) * SOFTMAX_SCALE
        probabilities = torch.softmax(logits, dim=-1)
        output[row] = torch.einsum("hgn,hnd->hgd", probabilities, values).reshape(
            NUM_Q_HEADS, HEAD_DIM
        )
    return output


def _triton_call(case):
    decode = _sparse_ops()["decode"]
    return decode(
        case["q"],
        case["k"],
        case["v"],
        case["req_to_token"],
        case["req_indices"],
        case["indices"],
        case["seq_lens"],
        SOFTMAX_SCALE,
    )


def _resolve_flash_attn():
    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        pass
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        return None


class _FlashAttnPath:
    def __init__(self, case):
        self.fn = _resolve_flash_attn()
        if self.fn is None:
            raise RuntimeError("flash_attn varlen is unavailable")
        self.case = case
        self.rows, self.topk = case["indices"].shape
        device = case["q"].device
        self.valid_counts = torch.zeros(self.rows, dtype=torch.int32, device=device)
        self.cu_k = torch.zeros(self.rows + 1, dtype=torch.int32, device=device)
        self.cu_q = torch.arange(self.rows + 1, dtype=torch.int32, device=device)
        shape = (self.rows * self.topk, NUM_KV_HEADS, HEAD_DIM)
        self.packed_k = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.packed_v = torch.empty_like(self.packed_k)

    def full(self):
        ops = _sparse_ops()
        case = self.case
        ops["cu_seqlens"](
            case["seq_lens"],
            case["indices"],
            self.valid_counts,
            self.cu_k,
            self.rows,
            self.topk,
        )
        ops["compact"](
            case["k"],
            case["v"],
            case["req_to_token"],
            case["req_indices"],
            case["indices"],
            case["seq_lens"],
            self.cu_k,
            self.packed_k,
            self.packed_v,
            self.rows,
            self.topk,
        )
        output = self.fn(
            q=case["q"],
            k=self.packed_k,
            v=self.packed_v,
            cu_seqlens_q=self.cu_q,
            cu_seqlens_k=self.cu_k,
            max_seqlen_q=1,
            max_seqlen_k=self.topk,
            softmax_scale=SOFTMAX_SCALE,
            causal=True,
        )
        return output[0] if isinstance(output, tuple) else output


class _TrtllmPath:
    def __init__(self, case):
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache

        self.fn = trtllm_batch_decode_with_kv_cache
        self.case = case
        self.rows, self.topk = case["indices"].shape
        self.pages_per_row = (self.topk + TRTLLM_PAGE - 1) // TRTLLM_PAGE
        self.stride = self.pages_per_row * TRTLLM_PAGE
        device = case["q"].device
        self.valid_counts = torch.zeros(self.rows, dtype=torch.int32, device=device)
        self.cu_strided = (
            torch.arange(self.rows + 1, dtype=torch.int32, device=device) * self.stride
        )
        self.block_tables = (
            torch.arange(self.rows, dtype=torch.int32, device=device)[:, None]
            * self.pages_per_row
            + torch.arange(self.pages_per_row, dtype=torch.int32, device=device)[
                None, :
            ]
        ).contiguous()
        shape = (self.rows * self.stride, NUM_KV_HEADS, HEAD_DIM)
        self.packed_k = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.packed_v = torch.empty_like(self.packed_k)
        self.k_cache = self.packed_k.view(
            -1, TRTLLM_PAGE, NUM_KV_HEADS, HEAD_DIM
        ).permute(0, 2, 1, 3)
        self.v_cache = self.packed_v.view(
            -1, TRTLLM_PAGE, NUM_KV_HEADS, HEAD_DIM
        ).permute(0, 2, 1, 3)
        self.workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )

    def gather(self):
        ops = _sparse_ops()
        case = self.case
        ops["valid_counts"](
            case["seq_lens"],
            case["indices"],
            self.valid_counts,
            self.rows,
            self.topk,
        )
        ops["compact"](
            case["k"],
            case["v"],
            case["req_to_token"],
            case["req_indices"],
            case["indices"],
            case["seq_lens"],
            self.cu_strided,
            self.packed_k,
            self.packed_v,
            self.rows,
            self.topk,
        )

    def attention(self):
        return self.fn(
            query=self.case["q"],
            kv_cache=(self.k_cache, self.v_cache),
            workspace_buffer=self.workspace,
            block_tables=self.block_tables,
            seq_lens=self.valid_counts,
            max_seq_len=self.stride,
            bmm1_scale=SOFTMAX_SCALE,
            bmm2_scale=1.0,
        )

    def full(self):
        self.gather()
        return self.attention()


def _errors(actual, reference):
    delta = (actual.float().reshape(reference.shape) - reference).abs()
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
    }


def _bf16_floor(reference):
    delta = (reference.bfloat16().float() - reference).abs()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item()}


def _bench(fn, *, warmup: int, iterations: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _graph_check(case):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _triton_call(case)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = _triton_call(case)
    graph.replay()
    torch.cuda.synchronize()
    first = output.clone()
    saved_query = case["q"].clone()
    case["q"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    changed = not torch.equal(first, output)
    case["q"].copy_(saved_query)
    graph.replay()
    torch.cuda.synchronize()
    return {
        "finite": bool(torch.isfinite(output).all()),
        "input_change_observed": changed,
        "restore_max_abs_delta": (output.float() - first.float()).abs().max().item(),
    }


def _optional_benchmark(factory, args):
    try:
        path = factory()
        path.full()
        torch.cuda.synchronize()
        return (
            path,
            _bench(path.full, warmup=args.warmup, iterations=args.iterations),
            None,
        )
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _check_case(rows, args):
    case = _make_case(rows)
    reference = _reference_fp32(case)
    floor = _bf16_floor(reference)

    first = _triton_call(case)
    second = _triton_call(case)
    torch.cuda.synchronize()
    error = _errors(first, reference)
    deterministic = bool(torch.equal(first, second))
    triton_ms = _bench(
        lambda: _triton_call(case),
        warmup=args.warmup,
        iterations=args.iterations,
    )

    xqa, xqa_full_ms, xqa_error = _optional_benchmark(lambda: _TrtllmPath(case), args)
    xqa_core_ms = (
        None
        if xqa is None
        else _bench(xqa.attention, warmup=args.warmup, iterations=args.iterations)
    )
    _, flash_ms, flash_error = _optional_benchmark(lambda: _FlashAttnPath(case), args)
    graph = _graph_check(case) if rows in (1, 16, 64) else None

    numeric_ok = (
        error["finite"]
        and deterministic
        and error["max_abs"] <= floor["max_abs"] + args.floor_slack
    )
    latency_ok = rows not in (1, 16) or xqa_full_ms is None or triton_ms <= xqa_full_ms
    batch64_ok = rows != 64 or triton_ms <= args.batch64_limit_ms
    graph_ok = graph is None or (
        graph["finite"]
        and graph["input_change_observed"]
        and graph["restore_max_abs_delta"] == 0.0
    )
    return {
        "rows": rows,
        "bf16_floor_max_abs": floor["max_abs"],
        "triton_max_abs": error["max_abs"],
        "finite": error["finite"],
        "deterministic": deterministic,
        "triton_ms": triton_ms,
        "xqa_full_ms": xqa_full_ms,
        "xqa_core_ms": xqa_core_ms,
        "xqa_error": xqa_error,
        "flash_ms": flash_ms,
        "flash_error": flash_error,
        "cuda_graph": graph,
        "numeric_ok": numeric_ok,
        "latency_ok": latency_ok,
        "batch64_ok": batch64_ok,
        "graph_ok": graph_ok,
        "passed": numeric_ok and latency_ok and batch64_ok and graph_ok,
    }


def _environment_line():
    import triton

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()[0]
    return (
        f"gpu={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} "
        f"driver={driver} cuda={torch.version.cuda} torch={torch.__version__} "
        f"triton={triton.__version__} sglang_commit={commit} "
        "model=Qwen3.8-Flash-Next quant=FP8/NVFP4-attention-BF16 "
        f"heads={NUM_Q_HEADS} kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} "
        f"topk={FINAL_TOPK} dtype=bfloat16"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 4, 16, 64])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--floor-slack", type=float, default=2e-3)
    parser.add_argument("--batch64-limit-ms", type=float, default=0.210)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    decode = _sparse_ops()["decode"]
    print(f"kernel={inspect.getfile(decode)}")
    print(_environment_line())
    results = [_check_case(rows, args) for rows in args.rows]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(
            "| rows | Triton ms | XQA full ms | XQA core ms | flash ms | "
            "max abs | bf16 floor | graph drift | result |"
        )
        print("|---:|---:|---:|---:|---:|---:|---:|---:|:---|")

        def value(item):
            return "n/a" if item is None else f"{item:.4f}"

        for result in results:
            graph = result["cuda_graph"]
            drift = "n/a" if graph is None else f'{graph["restore_max_abs_delta"]:.3g}'
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f'| {result["rows"]} | {result["triton_ms"]:.4f} | '
                f'{value(result["xqa_full_ms"])} | {value(result["xqa_core_ms"])} | '
                f'{value(result["flash_ms"])} | {result["triton_max_abs"]:.6f} | '
                f'{result["bf16_floor_max_abs"]:.6f} | {drift} | {status} |'
            )
            for backend in ("xqa", "flash"):
                if result[f"{backend}_error"]:
                    print(
                        f'rows={result["rows"]} {backend}={result[f"{backend}_error"]}'
                    )
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    if not torch.cuda.is_available() or not is_sm120_supported():
        print("[skip] QSA sparse decode benchmark requires SM120 CUDA.")
        sys.exit(0)
    main()
