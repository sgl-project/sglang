import argparse
import json
import math

import flashinfer
import torch
from common import require_single_visible_gpu


def make_inputs(num_groups, beam_width, context_len, decode_len):
    torch.manual_seed(0)
    q_heads = 16
    kv_heads = 8
    head_dim = 128
    num_rows = num_groups * beam_width
    context_capacity = num_groups * context_len
    suffix_capacity = num_rows * decode_len
    num_slots = context_capacity + suffix_capacity

    q = torch.randn(
        num_rows,
        q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        num_slots,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    kv_cache = (k.unsqueeze(1), v.unsqueeze(1))

    context_indices = torch.arange(context_capacity, device="cuda", dtype=torch.int32)
    context_indptr = (
        torch.arange(num_groups + 1, device="cuda", dtype=torch.int32) * context_len
    )
    suffix_indices = torch.arange(
        context_capacity,
        context_capacity + suffix_capacity,
        device="cuda",
        dtype=torch.int32,
    )
    suffix_indptr = (
        torch.arange(num_rows + 1, device="cuda", dtype=torch.int32) * decode_len
    )

    full_indices = torch.empty(
        num_rows * (context_len + decode_len),
        device="cuda",
        dtype=torch.int32,
    )
    full = full_indices.view(num_rows, context_len + decode_len)
    group_rows = torch.arange(num_rows, device="cuda", dtype=torch.int64) // beam_width
    context_offsets = (
        group_rows[:, None] * context_len
        + torch.arange(context_len, device="cuda", dtype=torch.int64)[None, :]
    )
    full[:, :context_len] = context_indices[context_offsets]
    full[:, context_len:] = suffix_indices.view(num_rows, decode_len)
    full_indptr = torch.arange(num_rows + 1, device="cuda", dtype=torch.int32) * (
        context_len + decode_len
    )

    one_per_group = torch.ones(num_groups, device="cuda", dtype=torch.int32)
    one_per_row = torch.ones(num_rows, device="cuda", dtype=torch.int32)
    group_q_indptr = (
        torch.arange(num_groups + 1, device="cuda", dtype=torch.int32) * beam_width
    )
    row_q_indptr = torch.arange(num_rows + 1, device="cuda", dtype=torch.int32)

    return {
        "q": q,
        "kv_cache": kv_cache,
        "full_indptr": full_indptr,
        "full_indices": full_indices,
        "full_last_page_len": one_per_row,
        "group_q_indptr": group_q_indptr,
        "context_indptr": context_indptr,
        "context_indices": context_indices,
        "context_last_page_len": one_per_group,
        "row_q_indptr": row_q_indptr,
        "suffix_indptr": suffix_indptr,
        "suffix_indices": suffix_indices,
        "suffix_last_page_len": one_per_row,
    }


def make_wrappers(inputs, num_groups, beam_width, decode_len):
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    standard = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    standard.plan(
        inputs["full_indptr"],
        inputs["full_indices"],
        inputs["full_last_page_len"],
        16,
        8,
        128,
        1,
        sm_scale=1.0 / math.sqrt(128),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    cascade = flashinfer.MultiLevelCascadeAttentionWrapper(2, workspace, "NHD")
    cascade.plan(
        [inputs["group_q_indptr"], inputs["row_q_indptr"]],
        [inputs["context_indptr"], inputs["suffix_indptr"]],
        [inputs["context_indices"], inputs["suffix_indices"]],
        [
            inputs["context_last_page_len"],
            inputs["suffix_last_page_len"],
        ],
        16,
        8,
        128,
        1,
        sm_scale=1.0 / math.sqrt(128),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    suffix_wrappers = []
    for group_idx in range(num_groups):
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
        row_start = group_idx * beam_width
        row_end = row_start + beam_width
        indices_start = row_start * decode_len
        indices_end = row_end * decode_len
        wrapper.plan(
            inputs["suffix_indptr"][row_start : row_end + 1]
            - inputs["suffix_indptr"][row_start],
            inputs["suffix_indices"][indices_start:indices_end],
            inputs["suffix_last_page_len"][row_start:row_end],
            16,
            8,
            128,
            1,
            sm_scale=1.0 / math.sqrt(128),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        suffix_wrappers.append(wrapper)
    suffix_all = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    suffix_all.plan(
        inputs["suffix_indptr"],
        inputs["suffix_indices"],
        inputs["suffix_last_page_len"],
        16,
        8,
        128,
        1,
        sm_scale=1.0 / math.sqrt(128),
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    return standard, cascade, suffix_wrappers, suffix_all


def run_shared_prefix(
    inputs,
    suffix_wrappers,
    num_groups,
    beam_width,
    context_len,
    use_fp16_qk_reduction,
    gather_context,
):
    outputs = []
    k_cache, v_cache = inputs["kv_cache"]
    for group_idx in range(num_groups):
        row_start = group_idx * beam_width
        row_end = row_start + beam_width
        context_start = group_idx * context_len
        context_end = context_start + context_len
        q = inputs["q"][row_start:row_end]
        if gather_context:
            slots = inputs["context_indices"][context_start:context_end].long()
            k_shared = k_cache[slots, 0]
            v_shared = v_cache[slots, 0]
        else:
            k_shared = k_cache[context_start:context_end, 0]
            v_shared = v_cache[context_start:context_end, 0]
        prefix_output, prefix_lse = flashinfer.single_prefill_with_kv_cache(
            q,
            k_shared,
            v_shared,
            causal=False,
            sm_scale=1.0 / math.sqrt(128),
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            return_lse=True,
        )
        suffix_output, suffix_lse = suffix_wrappers[group_idx].forward_return_lse(
            q,
            inputs["kv_cache"],
            sm_scale=1.0 / math.sqrt(128),
        )
        output, _ = flashinfer.merge_state(
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
        )
        outputs.append(output)
    return torch.cat(outputs)


def run_shared_prefix_global_suffix(
    inputs,
    suffix_wrapper,
    num_groups,
    beam_width,
    context_len,
    use_fp16_qk_reduction,
    gather_context,
):
    prefix_outputs = []
    prefix_lses = []
    k_cache, v_cache = inputs["kv_cache"]
    for group_idx in range(num_groups):
        row_start = group_idx * beam_width
        row_end = row_start + beam_width
        context_start = group_idx * context_len
        context_end = context_start + context_len
        q = inputs["q"][row_start:row_end]
        if gather_context:
            slots = inputs["context_indices"][context_start:context_end].long()
            k_shared = k_cache[slots, 0]
            v_shared = v_cache[slots, 0]
        else:
            k_shared = k_cache[context_start:context_end, 0]
            v_shared = v_cache[context_start:context_end, 0]
        prefix_output, prefix_lse = flashinfer.single_prefill_with_kv_cache(
            q,
            k_shared,
            v_shared,
            causal=False,
            sm_scale=1.0 / math.sqrt(128),
            use_fp16_qk_reduction=use_fp16_qk_reduction,
            return_lse=True,
        )
        prefix_outputs.append(prefix_output)
        prefix_lses.append(prefix_lse)
    suffix_output, suffix_lse = suffix_wrapper.forward_return_lse(
        inputs["q"],
        inputs["kv_cache"],
        sm_scale=1.0 / math.sqrt(128),
    )
    output, _ = flashinfer.merge_state(
        torch.cat(prefix_outputs),
        torch.cat(prefix_lses),
        suffix_output,
        suffix_lse,
    )
    return output


def benchmark(fn, warmup=10, iterations=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    return {
        "median_ms": times[len(times) // 2],
        "mean_ms": sum(times) / len(times),
        "min_ms": times[0],
        "max_ms": times[-1],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare FlashInfer decode and shared-prefix alternatives."
    )
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 4, 10])
    parser.add_argument("--beam-width", type=int, default=2000)
    parser.add_argument("--context-len", type=int, default=384)
    parser.add_argument("--decode-len", type=int, default=3)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()
    require_single_visible_gpu(args.physical_gpu)

    print(
        json.dumps(
            {
                "flashinfer_version": flashinfer.__version__,
                "beam_width": args.beam_width,
                "context_len": args.context_len,
                "decode_len": args.decode_len,
            },
            sort_keys=True,
        )
    )
    for groups in args.groups:
        inputs = make_inputs(groups, args.beam_width, args.context_len, args.decode_len)
        standard, cascade, suffix_wrappers, suffix_all = make_wrappers(
            inputs, groups, args.beam_width, args.decode_len
        )
        standard_output = standard.run(inputs["q"], inputs["kv_cache"])
        cascade_output = cascade.run(inputs["q"], inputs["kv_cache"])
        shared_prefix_output = run_shared_prefix(
            inputs,
            suffix_wrappers,
            groups,
            args.beam_width,
            args.context_len,
            False,
            False,
        )
        difference = (standard_output.float() - cascade_output.float()).abs()
        shared_prefix_difference = (
            standard_output.float() - shared_prefix_output.float()
        ).abs()
        result = {
            "groups": groups,
            "rows": groups * args.beam_width,
            "correctness": {
                "cascade": {
                    "max_abs": difference.max().item(),
                    "mean_abs": difference.mean().item(),
                    "allclose_rtol_5e-2_atol_5e-2": torch.allclose(
                        standard_output.float(),
                        cascade_output.float(),
                        rtol=5e-2,
                        atol=5e-2,
                    ),
                },
                "shared_prefix": {
                    "max_abs": shared_prefix_difference.max().item(),
                    "mean_abs": shared_prefix_difference.mean().item(),
                    "allclose_rtol_5e-2_atol_5e-2": torch.allclose(
                        standard_output.float(),
                        shared_prefix_output.float(),
                        rtol=5e-2,
                        atol=5e-2,
                    ),
                },
            },
            "standard_decode": benchmark(
                lambda: standard.run(inputs["q"], inputs["kv_cache"])
            ),
            "cascade": benchmark(lambda: cascade.run(inputs["q"], inputs["kv_cache"])),
            "shared_prefix_fp32_contiguous": benchmark(
                lambda: run_shared_prefix(
                    inputs,
                    suffix_wrappers,
                    groups,
                    args.beam_width,
                    args.context_len,
                    False,
                    False,
                )
            ),
            "shared_prefix_fp16_contiguous": benchmark(
                lambda: run_shared_prefix(
                    inputs,
                    suffix_wrappers,
                    groups,
                    args.beam_width,
                    args.context_len,
                    True,
                    False,
                )
            ),
            "shared_prefix_fp16_gathered": benchmark(
                lambda: run_shared_prefix(
                    inputs,
                    suffix_wrappers,
                    groups,
                    args.beam_width,
                    args.context_len,
                    True,
                    True,
                )
            ),
            "shared_prefix_global_suffix_fp32_contiguous": benchmark(
                lambda: run_shared_prefix_global_suffix(
                    inputs,
                    suffix_all,
                    groups,
                    args.beam_width,
                    args.context_len,
                    False,
                    False,
                )
            ),
            "shared_prefix_global_suffix_fp16_gathered": benchmark(
                lambda: run_shared_prefix_global_suffix(
                    inputs,
                    suffix_all,
                    groups,
                    args.beam_width,
                    args.context_len,
                    True,
                    True,
                )
            ),
        }
        result["cascade_speedup"] = (
            result["standard_decode"]["median_ms"] / result["cascade"]["median_ms"]
        )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
