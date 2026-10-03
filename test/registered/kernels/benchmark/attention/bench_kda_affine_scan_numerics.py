import argparse
import json
import math

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.kda import (
    chunk_gla_fwd_o_gk,
    chunk_kda_fwd_intra,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="jit-kernel-benchmark-test-amd")

CHUNK_SIZE = 64
HEAD_DIM = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare rounded and scan-compatible KDA recurrences"
    )
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--heads", type=int, choices=(8, 16), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefix",
        choices=("zero", "random"),
        default="random",
    )
    parser.add_argument(
        "--decay",
        choices=("weak", "realistic", "strong"),
        default="realistic",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("rounded", "no_h_round", "affine"),
        default=("rounded", "no_h_round", "affine"),
    )
    parser.add_argument("--segment-sizes", nargs="*", type=int, default=())
    return parser.parse_args()


def make_inputs(args):
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (1, args.tokens, args.heads, HEAD_DIM)

    def randn(*size, dtype=torch.bfloat16, scale=1.0):
        return (
            torch.randn(
                *size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * scale
        )

    q = F.normalize(randn(*shape).float(), dim=-1).to(torch.bfloat16)
    k = F.normalize(randn(*shape).float(), dim=-1).to(torch.bfloat16)
    v = randn(*shape, scale=0.1)
    beta = torch.sigmoid(randn(1, args.tokens, args.heads).float())
    decay_scale = {
        "weak": 1e-4,
        "realistic": 1e-2,
        "strong": 1e-1,
    }[args.decay]
    gate_increment = (
        -torch.rand(
            shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * decay_scale
    )
    cu_seqlens = torch.tensor(
        [0, args.tokens],
        device=device,
        dtype=torch.int64,
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    g = chunk_local_cumsum(
        gate_increment,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    w, u, _, kg, A, _ = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=HEAD_DIM**-0.5,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
        safe_gate=False,
        fuse_diagonal=False,
        fuse_recompute=False,
    )
    if args.prefix == "zero":
        initial_state = torch.zeros(
            (1, args.heads, HEAD_DIM, HEAD_DIM),
            device=device,
            dtype=torch.float32,
        )
    else:
        initial_state = randn(
            1,
            args.heads,
            HEAD_DIM,
            HEAD_DIM,
            dtype=torch.float32,
            scale=0.01,
        )
    return {
        "q": q,
        "g": g,
        "w": w,
        "u": u,
        "kg": kg,
        "A": A,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
    }


def run_kernel_reference(inputs):
    state = inputs["initial_state"].clone()
    state_indices = torch.tensor([0], device="cuda", dtype=torch.int64)
    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=inputs["kg"],
        w=inputs["w"],
        u=inputs["u"],
        gk=inputs["g"],
        initial_state=state,
        initial_state_indices=state_indices,
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        use_exp2=True,
    )
    output = torch.empty_like(inputs["u"])
    chunk_gla_fwd_o_gk(
        q=inputs["q"],
        v=v_new,
        g=inputs["g"],
        A=inputs["A"],
        h=h,
        o=output,
        scale=HEAD_DIM**-0.5,
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
    )
    return output, state, h, v_new


def _output_chunk(q, g, A, state, v_new):
    q_scaled = (q * (HEAD_DIM**-0.5)).to(torch.bfloat16)
    qg = (q_scaled.float() * torch.exp2(g)).to(torch.bfloat16)
    h_out = state.to(torch.bfloat16)
    inter = torch.einsum("thk,hvk->thv", qg.float(), h_out.float())
    length = q.shape[0]
    local_A = torch.tril(A[:length, :, :length].permute(1, 0, 2))
    local = torch.einsum(
        "hts,shv->thv",
        local_A.float(),
        v_new.float(),
    )
    return (inter + local).to(torch.bfloat16)


def run_sequential_reference(inputs, variant):
    state = inputs["initial_state"][0].clone()
    chunks = math.ceil(inputs["q"].shape[1] / CHUNK_SIZE)
    h_chunks = []
    fp32_state_chunks = []
    v_new_chunks = []
    output_chunks = []
    for chunk in range(chunks):
        begin = chunk * CHUNK_SIZE
        end = min(begin + CHUNK_SIZE, inputs["q"].shape[1])
        q = inputs["q"][0, begin:end]
        g = inputs["g"][0, begin:end]
        w = inputs["w"][0, begin:end]
        u = inputs["u"][0, begin:end]
        kg = inputs["kg"][0, begin:end]
        A = inputs["A"][0, begin:end]

        fp32_state_chunks.append(state)
        h_chunks.append(state.to(torch.bfloat16))
        h_operand = state.to(torch.bfloat16) if variant == "rounded" else state
        projected = torch.einsum(
            "thk,hvk->thv",
            w.float(),
            h_operand.float(),
        )
        v_new_fp32 = u.float() - projected
        v_new_output = v_new_fp32.to(torch.bfloat16)
        v_new_chunks.append(v_new_output)
        output_chunks.append(_output_chunk(q, g, A, state, v_new_output))

        decay = torch.exp2(g[-1]).unsqueeze(1)
        if variant == "affine":
            transition = torch.diag_embed(decay.squeeze(1))
            transition -= torch.einsum(
                "thk,thj->hkj",
                w.float(),
                kg.float(),
            )
            bias = torch.einsum(
                "thv,thk->hvk",
                u.float(),
                kg.float(),
            )
            state = torch.einsum("hvk,hkj->hvj", state, transition) + bias
        else:
            update_value = v_new_output
            state = state * decay
            state += torch.einsum(
                "thv,thk->hvk",
                update_value.float(),
                kg.float(),
            )
    return (
        torch.cat(output_chunks, dim=0).unsqueeze(0),
        state.unsqueeze(0),
        torch.stack(h_chunks, dim=0).unsqueeze(0),
        torch.cat(v_new_chunks, dim=0).unsqueeze(0),
        torch.stack(fp32_state_chunks, dim=0),
    )


def _chunk_transform(inputs, chunk):
    begin = chunk * CHUNK_SIZE
    end = min(begin + CHUNK_SIZE, inputs["q"].shape[1])
    g = inputs["g"][0, begin:end]
    w = inputs["w"][0, begin:end].float()
    u = inputs["u"][0, begin:end].float()
    kg = inputs["kg"][0, begin:end].float()
    decay = torch.exp2(g[-1])
    transition = torch.diag_embed(decay)
    transition -= torch.einsum("thk,thj->hkj", w, kg)
    bias = torch.einsum("thv,thk->hvk", u, kg)
    return transition, bias


def _identity_transform(heads, device):
    transition = torch.eye(
        HEAD_DIM,
        device=device,
        dtype=torch.float32,
    ).expand(heads, -1, -1)
    bias = torch.zeros(
        (heads, HEAD_DIM, HEAD_DIM),
        device=device,
        dtype=torch.float32,
    )
    return transition, bias


def _compose(first, second):
    first_transition, first_bias = first
    second_transition, second_bias = second
    return (
        torch.bmm(first_transition, second_transition),
        torch.bmm(first_bias, second_transition) + second_bias,
    )


def _segment_summaries(inputs, segment_size):
    chunks = math.ceil(inputs["q"].shape[1] / CHUNK_SIZE)
    summaries = []
    for segment_begin in range(0, chunks, segment_size):
        summary = _identity_transform(
            inputs["q"].shape[2],
            inputs["q"].device,
        )
        for chunk in range(
            segment_begin,
            min(segment_begin + segment_size, chunks),
        ):
            summary = _compose(summary, _chunk_transform(inputs, chunk))
        summaries.append(summary)
    return summaries


def _left_prefix(summaries):
    prefix = _identity_transform(
        summaries[0][0].shape[0],
        summaries[0][0].device,
    )
    prefixes = []
    for summary in summaries:
        prefixes.append(prefix)
        prefix = _compose(prefix, summary)
    return prefixes


def _tree_prefix(summaries):
    count = len(summaries)
    padded_count = 1 << (count - 1).bit_length()
    values = list(summaries)
    values.extend(
        _identity_transform(
            summaries[0][0].shape[0],
            summaries[0][0].device,
        )
        for _ in range(padded_count - count)
    )

    step = 2
    while step <= padded_count:
        half = step // 2
        for right in range(step - 1, padded_count, step):
            values[right] = _compose(values[right - half], values[right])
        step *= 2

    values[-1] = _identity_transform(
        summaries[0][0].shape[0],
        summaries[0][0].device,
    )
    step = padded_count
    while step >= 2:
        half = step // 2
        for right in range(step - 1, padded_count, step):
            left = right - half
            left_total = values[left]
            values[left] = values[right]
            values[right] = _compose(values[right], left_total)
        step //= 2
    return values[:count]


def _apply_transform(state, transform):
    transition, bias = transform
    return torch.bmm(state, transition) + bias


def _rounded_chunk_step(inputs, chunk, state):
    begin = chunk * CHUNK_SIZE
    end = min(begin + CHUNK_SIZE, inputs["q"].shape[1])
    q = inputs["q"][0, begin:end]
    g = inputs["g"][0, begin:end]
    w = inputs["w"][0, begin:end]
    u = inputs["u"][0, begin:end]
    kg = inputs["kg"][0, begin:end]
    A = inputs["A"][0, begin:end]
    projected = torch.einsum(
        "thk,hvk->thv",
        w.float(),
        state.to(torch.bfloat16).float(),
    )
    v_new = (u.float() - projected).to(torch.bfloat16)
    output = _output_chunk(q, g, A, state, v_new)
    state = state * torch.exp2(g[-1]).unsqueeze(1)
    state += torch.einsum(
        "thv,thk->hvk",
        v_new.float(),
        kg.float(),
    )
    return state, output


def _replay_segments(inputs, prefixes, segment_size):
    chunks = math.ceil(inputs["q"].shape[1] / CHUNK_SIZE)
    initial_state = inputs["initial_state"][0]
    outputs = []
    starts = []
    final_state = None
    for segment, segment_begin in enumerate(range(0, chunks, segment_size)):
        state = _apply_transform(initial_state, prefixes[segment])
        starts.append(state)
        segment_outputs = []
        for chunk in range(
            segment_begin,
            min(segment_begin + segment_size, chunks),
        ):
            state, output = _rounded_chunk_step(inputs, chunk, state)
            segment_outputs.append(output)
        outputs.extend(segment_outputs)
        if segment == len(prefixes) - 1:
            final_state = state
    return (
        torch.cat(outputs, dim=0).unsqueeze(0),
        final_state.unsqueeze(0),
        torch.stack(starts, dim=0),
    )


def _scan_cost(tokens, heads, segment_size):
    chunks = math.ceil(tokens / CHUNK_SIZE)
    segments = math.ceil(chunks / segment_size)
    padded_segments = 1 << (segments - 1).bit_length()
    matrix_elements = heads * HEAD_DIM * HEAD_DIM
    summary_bytes = segments * 2 * matrix_elements * 4
    chunk_transform_flops = 4 * CHUNK_SIZE * HEAD_DIM * HEAD_DIM
    local_compose_flops = 4 * HEAD_DIM**3
    summary_flops = heads * (
        chunks * chunk_transform_flops + (chunks - segments) * local_compose_flops
    )
    scan_combines = 2 * (padded_segments - 1)
    scan_flops = heads * scan_combines * local_compose_flops
    replay_flops = heads * chunks * chunk_transform_flops
    current_flops = replay_flops
    return {
        "chunks": chunks,
        "segments": segments,
        "summary_mib": summary_bytes / 2**20,
        "summary_gflop": summary_flops / 1e9,
        "scan_gflop": scan_flops / 1e9,
        "replay_gflop": replay_flops / 1e9,
        "work_ratio_vs_current": (summary_flops + scan_flops + replay_flops)
        / current_flops,
    }


def metrics(actual, expected):
    actual = actual.float()
    expected = expected.float()
    difference = actual - expected
    denominator = torch.sqrt(torch.mean(expected.square())).clamp_min(1e-12)
    relative_rmse = torch.sqrt(torch.mean(difference.square())) / denominator
    cosine = F.cosine_similarity(
        actual.flatten(),
        expected.flatten(),
        dim=0,
    )
    return {
        "max_abs": difference.abs().max().item(),
        "relative_rmse": relative_rmse.item(),
        "cosine": cosine.item(),
        "allclose": torch.allclose(
            actual,
            expected,
            atol=2e-2,
            rtol=1e-2,
        ),
        "finite": torch.isfinite(actual).all().item(),
    }


def main():
    args = parse_args()
    inputs = make_inputs(args)
    kernel_output, kernel_state, kernel_h, kernel_v_new = run_kernel_reference(inputs)
    results = {
        "shape": {
            "tokens": args.tokens,
            "heads": args.heads,
            "seed": args.seed,
            "prefix": args.prefix,
            "decay": args.decay,
        },
        "variants": {},
        "segments": {},
    }
    variant_tensors = {}
    for variant in args.variants:
        output, state, h, v_new, fp32_states = run_sequential_reference(inputs, variant)
        variant_tensors[variant] = {
            "output": output,
            "state": state,
            "fp32_states": fp32_states,
        }
        results["variants"][variant] = {
            "output": metrics(output, kernel_output),
            "state": metrics(state, kernel_state),
            "h": metrics(h, kernel_h),
            "v_new": metrics(v_new, kernel_v_new),
        }

    if args.segment_sizes:
        if "affine" not in variant_tensors:
            affine = run_sequential_reference(inputs, "affine")
            variant_tensors["affine"] = {
                "output": affine[0],
                "state": affine[1],
                "fp32_states": affine[4],
            }
        for segment_size in args.segment_sizes:
            if segment_size <= 0:
                raise ValueError("segment sizes must be positive")
            summaries = _segment_summaries(inputs, segment_size)
            left_prefixes = _left_prefix(summaries)
            tree_prefixes = _tree_prefix(summaries)
            left_transition = torch.stack([item[0] for item in left_prefixes])
            tree_transition = torch.stack([item[0] for item in tree_prefixes])
            left_bias = torch.stack([item[1] for item in left_prefixes])
            tree_bias = torch.stack([item[1] for item in tree_prefixes])
            replay_output, replay_state, tree_starts = _replay_segments(
                inputs,
                tree_prefixes,
                segment_size,
            )
            segment_chunks = torch.arange(
                0,
                kernel_h.shape[1],
                segment_size,
                device=kernel_h.device,
            )
            current_starts = kernel_h[0, segment_chunks].float()
            affine_starts = variant_tensors["affine"]["fp32_states"][segment_chunks]
            per_segment_rmse = [
                metrics(tree_starts[index], current_starts[index])["relative_rmse"]
                for index in range(len(tree_starts))
            ]
            results["segments"][str(segment_size)] = {
                "tree_vs_left_transition": metrics(
                    tree_transition,
                    left_transition,
                ),
                "tree_vs_left_bias": metrics(tree_bias, left_bias),
                "start_vs_affine": metrics(tree_starts, affine_starts),
                "start_vs_current": metrics(tree_starts, current_starts),
                "replay_output": metrics(replay_output, kernel_output),
                "replay_state": metrics(replay_state, kernel_state),
                "start_relative_rmse": per_segment_rmse,
                "cost": _scan_cost(
                    args.tokens,
                    args.heads,
                    segment_size,
                ),
            }
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
