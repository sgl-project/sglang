"""Differential microbenchmark for Kimi-K3 native vs Gluon KDA prefill."""

import argparse
import csv
from dataclasses import dataclass
from itertools import accumulate

import torch

from sglang.kernels.ops.attention import kda_prefill_gluon_hip
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn


@dataclass
class Inputs:
    qkv: torch.Tensor
    gate: torch.Tensor
    forget_a: torch.Tensor
    beta: torch.Tensor
    forget_weight: torch.Tensor
    conv_weight: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    conv_state: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    lengths: list[int]
    has_initial_state: torch.Tensor


def make_inputs(
    rows: int,
    sequences: int,
    seed: int,
    initial: bool | list[bool],
    lengths: list[int] | None = None,
) -> Inputs:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    bf16 = torch.bfloat16
    lengths = lengths or [rows // sequences] * sequences
    assert len(lengths) == sequences and sum(lengths) == rows
    cu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device)

    def randn(shape, dtype=bf16, scale=0.1):
        return (torch.randn(shape, device=device, dtype=dtype) * scale).contiguous()

    wide = randn((rows, 4 * 12 * 128))
    bfa = randn((rows, 144))
    slots = sequences + 3
    ids = torch.randperm(slots, device=device)[:sequences].to(torch.int32)
    initial_flags = torch.tensor(
        [initial] * sequences if isinstance(initial, bool) else initial,
        dtype=torch.bool,
        device=device,
    )
    state = randn((slots, 12, 128, 128), torch.float32, scale=0.01)
    # Native chunk_kda expects the scheduler to clear fresh recurrent slots.
    # has_initial_state controls convolution; it is not passed to chunk_kda.
    state[ids[~initial_flags].long()] = 0
    return Inputs(
        qkv=wide[:, : 3 * 12 * 128],
        gate=wide[:, 3 * 12 * 128 :],
        forget_a=bfa[:, :128],
        beta=bfa[:, 128:140],
        forget_weight=randn((12 * 128, 128)),
        conv_weight=randn((3 * 12 * 128, 4), torch.float32),
        a_log=randn((12,), torch.float32, scale=0.01),
        dt_bias=randn((12 * 128,), torch.float32, scale=0.01),
        norm_weight=torch.ones((128,), device=device, dtype=bf16),
        conv_state=randn((slots, 3, 3 * 12 * 128)),
        state=state,
        state_indices=ids,
        cu_seqlens=cu,
        lengths=lengths,
        has_initial_state=initial_flags,
    )


def native(inputs: Inputs, conv_state: torch.Tensor, state: torch.Tensor):
    forget = kimi_k3_tiny_gemm(inputs.forget_a, inputs.forget_weight)
    convolved = causal_conv1d_fn(
        inputs.qkv.transpose(0, 1),
        inputs.conv_weight,
        None,
        conv_states=conv_state.transpose(-1, -2),
        cache_indices=inputs.state_indices,
        has_initial_state=inputs.has_initial_state,
        query_start_loc=inputs.cu_seqlens,
        seq_lens_cpu=inputs.lengths,
        activation="silu",
    ).transpose(0, 1)
    q, k, v = convolved.split(12 * 128, dim=-1)
    q = q.view(1, -1, 12, 128)
    k = k.view(1, -1, 12, 128)
    v = v.view(1, -1, 12, 128)
    output = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=forget.view(1, -1, 12, 128),
        beta=inputs.beta.float().sigmoid().unsqueeze(0),
        initial_state=state,
        initial_state_indices=inputs.state_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inputs.cu_seqlens,
        A_log=inputs.a_log,
        dt_bias=inputs.dt_bias,
        lower_bound=-5.0,
    )
    return rms_norm_gated(
        output,
        inputs.gate.view(1, -1, 12, 128),
        inputs.norm_weight,
        None,
        activation="sigmoid",
        eps=1e-5,
    ).view(-1, 12 * 128)


def gluon(inputs: Inputs, conv_state: torch.Tensor, state: torch.Tensor):
    return kda_prefill_gluon_hip.run(
        inputs.qkv,
        inputs.gate,
        inputs.forget_a,
        inputs.beta,
        inputs.forget_weight,
        inputs.conv_weight,
        inputs.a_log,
        inputs.dt_bias,
        inputs.norm_weight,
        conv_state,
        state,
        inputs.state_indices,
        inputs.cu_seqlens,
        inputs.has_initial_state,
        lower_bound=-5.0,
        norm_eps=1e-5,
    )[0]


def elapsed_ms(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / rep


def compare(
    rows: int,
    sequences: int,
    warmup: int,
    rep: int,
    lengths: list[int] | None = None,
    initial: bool | list[bool] = False,
) -> dict:
    inputs = make_inputs(
        rows, sequences, seed=rows + sequences, initial=initial, lengths=lengths
    )
    native_conv = inputs.conv_state.clone()
    native_state = inputs.state.clone()
    gluon_conv = inputs.conv_state.clone()
    gluon_state = inputs.state.clone()
    native_out = native(inputs, native_conv, native_state)
    gluon_out = gluon(inputs, gluon_conv, gluon_state)
    torch.cuda.synchronize()

    def cosine(lhs, rhs):
        return torch.nn.functional.cosine_similarity(
            lhs.float().flatten(), rhs.float().flatten(), dim=0
        ).item()

    correctness = {
        "output_max_abs": (native_out - gluon_out).abs().max().item(),
        "output_cosine": cosine(native_out, gluon_out),
        "conv_max_abs": (native_conv - gluon_conv).abs().max().item(),
        "conv_cosine": cosine(native_conv, gluon_conv),
        "state_max_abs": (native_state - gluon_state).abs().max().item(),
        "state_cosine": cosine(native_state, gluon_state),
    }
    native_ms = elapsed_ms(
        lambda: native(inputs, native_conv, native_state), warmup, rep
    )
    gluon_ms = elapsed_ms(lambda: gluon(inputs, gluon_conv, gluon_state), warmup, rep)
    return {
        "rows": rows,
        "sequences": sequences,
        "lengths": inputs.lengths,
        "initial": initial,
        "native_ms": native_ms,
        "gluon_ms": gluon_ms,
        "speedup": native_ms / gluon_ms,
        **correctness,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rep", type=int, default=10)
    parser.add_argument("--output")
    parser.add_argument("--irregular", action="store_true")
    args = parser.parse_args()
    cases = [(1024, 1), (2048, 2), (4096, 1), (4096, 4), (6144, 2), (8192, 8)]
    rows = [compare(m, sequences, args.warmup, args.rep) for m, sequences in cases]
    if args.irregular:
        for lengths, initial in (
            ([4022, 4022, 148], False),
            ([3874, 4022], [True, False]),
            ([2001, 2018], False),
            ([17, 4002], [True, False]),
        ):
            rows.append(
                compare(
                    sum(lengths),
                    len(lengths),
                    args.warmup,
                    args.rep,
                    lengths=lengths,
                    initial=initial,
                )
            )
    for row in rows:
        print(
            f"M={row['rows']:4d} S={row['sequences']:2d} "
            f"native={row['native_ms']:.4f}ms gluon={row['gluon_ms']:.4f}ms "
            f"speedup={row['speedup']:.2f}x cosine={row['output_cosine']:.8f}"
        )
    if args.output:
        with open(args.output, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
