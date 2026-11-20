"""Paired, offline comparison of the upstream and SM90 QSA prefill chains."""

import argparse
import json
from statistics import median
from types import SimpleNamespace

import torch


@torch.inference_mode()
def run_case(lengths, prefixes, heads, kv_heads, samples, stride):
    from sglang.kernels.ops.attention.qsa_sm90 import paged_prefill
    from sglang.srt.layers.attention.qsa.config import QSAProfile
    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )
    from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_impl
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.test.mem_cache_utils import routed

    device = "cuda"
    torch.manual_seed(20261114)
    sequence_lengths = [length + prefix for length, prefix in zip(lengths, prefixes)]
    page_counts = [(length + 63) // 64 for length in sequence_lengths]
    physical_pages = torch.randperm(sum(page_counts), device=device)
    table = torch.zeros(
        (len(lengths), max(sequence_lengths)), device=device, dtype=torch.int32
    )
    offsets = torch.arange(64, device=device)
    index = 0
    for request, (length, pages) in enumerate(zip(sequence_lengths, page_counts)):
        slots = physical_pages[index : index + pages, None] * 64 + offsets
        table[request, :length] = slots.flatten()[:length]
        index += pages

    positions = torch.cat(
        [
            torch.arange(prefix, prefix + length, device=device, dtype=torch.int32)
            for length, prefix in zip(lengths, prefixes)
        ]
    )
    extend = torch.tensor(lengths, device=device, dtype=torch.int32)
    seq = torch.tensor(sequence_lengths, device=device, dtype=torch.int32)
    rows = sum(lengths)
    req_ids = torch.arange(len(lengths), device=device)
    request = torch.repeat_interleave(req_ids, extend, output_size=rows)
    row_lens = seq[request]
    new_slots = table[request, positions.long()].long()
    visible = ((positions + 1) // 4)[:, None]
    columns = torch.arange(512, device=device, dtype=torch.int32)[None, :]
    # Shift the start of each selection to control reuse between adjacent queries.
    phase = (torch.arange(rows, device=device, dtype=torch.int32) * stride)[:, None]
    blocks = torch.where(
        columns < visible, (columns + phase) % visible.clamp_min(1), -1
    )
    del visible, phase

    def rand(shape):
        return torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.15

    q = rand((rows, heads, 256))
    keys = rand((sum(page_counts) * 64, kv_heads, 256))
    values = rand(keys.shape)
    new_keys, new_values = keys[new_slots].contiguous(), values[new_slots].contiguous()

    class Pool:
        def set_kv_buffer(self, layer, loc, k, v):
            _set_kv_buffer_impl(
                k,
                v,
                keys,
                values,
                loc,
                kv_heads * 256,
                keys.dtype,
                torch.cuda,
                keys.shape[0],
            )

        def get_key_buffer(self, layer_id):
            return keys

        def get_value_buffer(self, layer_id):
            return values

    backend = QwenSparseAttnBackend()
    backend.qsa_profile = QSAProfile(4, 1, 128, 2048, 4)
    backend.token_to_kv_pool = routed(Pool())
    backend.req_to_token_pool = SimpleNamespace(req_to_token=table)
    backend.forward_metadata = SimpleNamespace(
        is_cuda_graph=False,
        sequence_lengths=seq,
        token_slot_table=table,
        indexer_metadata=SimpleNamespace(get_seqlens_expanded=lambda: row_lens),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        out_cache_loc=new_slots,
        req_pool_indices=req_ids,
        positions=positions,
        extend_seq_lens=extend,
        extend_seq_lens_cpu=lengths,
        seq_lens_cpu=sequence_lengths,
    )
    layer = SimpleNamespace(
        layer_id=0, tp_q_head_num=heads, head_dim=256, scaling=0.0625
    )

    def baseline():
        expanded = expand_qsa_block_indices(blocks, positions, row_lens, 4, 2048)
        return backend.forward_extend(
            q, new_keys, new_values, layer, batch, True, expanded
        )

    def candidate():
        return backend.forward_extend(
            q, new_keys, new_values, layer, batch, True, blocks
        )

    backend._sm90_bf16_prefill = True
    implementation = paged_prefill.sparse_prefill_blocks
    launches = 0

    def record_launch(*args):
        nonlocal launches
        launches += 1
        return implementation(*args)

    expected = baseline()
    paged_prefill.sparse_prefill_blocks = record_launch
    try:
        actual = candidate()
    finally:
        paged_prefill.sparse_prefill_blocks = implementation
    assert launches == 1, "benchmark fell back to Triton"
    max_error = 0.0
    for start in range(0, rows, 1024):
        a, b = actual[start : start + 1024], expected[start : start + 1024]
        torch.testing.assert_close(a, b, atol=0.025, rtol=0.025)
        max_error = max(max_error, float((a.float() - b.float()).abs().max()))
    del actual, expected

    for _ in range(5):
        baseline()
        candidate()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = {"triton": [], "sm90": []}
    implementations = (("triton", baseline), ("sm90", candidate))
    for index in range(samples):
        for name, run in (
            implementations if index % 2 == 0 else reversed(implementations)
        ):
            start.record()
            run()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end))

    pairs = sum(
        min((position + 1) // 4, 512) * 4 + (position + 1) % 4
        for length, prefix in zip(lengths, prefixes)
        for position in range(prefix, prefix + length)
    )
    flops = 4 * heads * 256 * pairs
    return {
        "extend": lengths,
        "prefix": prefixes,
        "max_abs_error": max_error,
        "ms": {name: median(values) for name, values in timings.items()},
        "effective_T": {
            name: flops / (median(values) * 1e9) for name, values in timings.items()
        },
        "paired_speedup": median(
            a / b for a, b in zip(timings["triton"], timings["sm90"])
        ),
        "raw_ms": timings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument(
        "--case", choices=("1k-b16", "4k-b4", "4k", "8k", "16k", "prefix", "varlen")
    )
    parser.add_argument(
        "--stride", type=int, default=28, help="Block selection shift per query row"
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        parser.error("an NVIDIA SM90 GPU is required")
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "scope": "KV write + index expansion/metadata + attention; CUDA events",
                "heads": args.heads,
                "kv_heads": args.kv_heads,
            }
        ),
        flush=True,
    )
    cases = {
        "1k-b16": ([1024] * 16, [0] * 16),
        "4k-b4": ([4096] * 4, [0] * 4),
        "4k": ([4096] * 32, [0] * 32),
        "8k": ([8192] * 16, [0] * 16),
        "16k": ([16384] * 8, [0] * 8),
        "prefix": ([16384] * 4, [49152] * 4),
        "varlen": ([16000, 384, 513], [49152, 0, 257]),
    }
    for case in (args.case,) if args.case else cases:
        result = run_case(
            *cases[case], args.heads, args.kv_heads, args.samples, args.stride
        )
        print(json.dumps({"case": case, **result}), flush=True)


if __name__ == "__main__":
    main()
