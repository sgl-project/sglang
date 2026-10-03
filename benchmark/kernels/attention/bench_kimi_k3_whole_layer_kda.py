"""Compare Kimi-K3's native decode chain with whole-layer Gluon KDA.

This is a single-rank TP8 microbenchmark. It includes both input projections,
f_b, convolution/recurrent KDA, gated RMSNorm, and the local output projection.
It intentionally excludes the TP all-reduce, which is unchanged by the port.
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
from aiter.tuned_gemm import tgemm

from sglang.kernels.ops.attention import (
    kda_whole_layer_gluon_hip as whole_kda,
)
from sglang.kernels.ops.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

HIDDEN = 7168
HEADS = 12
HEAD_DIM = 128
LOCAL = HEADS * HEAD_DIM
QKVG = 4 * LOCAL
BFA = HEAD_DIM + HEADS + 4
CONV = 3 * LOCAL


def _randn(shape, *, scale: float = 1.0, dtype=torch.bfloat16):
    return torch.randn(shape, device="cuda", dtype=dtype) * scale


def _make_inputs(rows: int):
    slots = rows + 2
    qkvg_weight = _randn((QKVG, HIDDEN), scale=HIDDEN**-0.5)
    beta_forget_weight = _randn((BFA, HIDDEN), scale=HIDDEN**-0.5)
    output_weight = _randn((HIDDEN, LOCAL), scale=LOCAL**-0.5)
    forget_weight = _randn((LOCAL, HEAD_DIM), scale=HEAD_DIM**-0.5)
    norm = FusedRMSNormGated(
        HEAD_DIM,
        eps=1e-5,
        activation="sigmoid",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    norm.weight.data.fill_(1)
    return {
        "hidden_states": _randn((rows, HIDDEN)),
        "qkvg_weight": qkvg_weight,
        "beta_forget_weight": beta_forget_weight,
        "merged_input_weight": torch.cat(
            (qkvg_weight, beta_forget_weight), dim=0
        ).contiguous(),
        "output_weight": output_weight,
        "forget_weight": forget_weight,
        "conv_weight": _randn((CONV, 4), scale=0.5, dtype=torch.float32),
        "A_log": torch.arange(1, HEADS + 1, device="cuda", dtype=torch.float32).log(),
        "dt_bias": torch.zeros(LOCAL, device="cuda", dtype=torch.float32),
        "norm_weight": torch.ones(HEAD_DIM, device="cuda", dtype=torch.bfloat16),
        "norm": norm,
        "conv_state": torch.zeros(slots, 3, CONV, device="cuda", dtype=torch.bfloat16),
        "state": torch.zeros(
            slots, HEADS, HEAD_DIM, HEAD_DIM, device="cuda", dtype=torch.float32
        ),
        "state_indices": torch.arange(1, rows + 1, device="cuda", dtype=torch.int32),
        "query_start_loc": torch.arange(rows + 1, device="cuda", dtype=torch.int32),
    }


def _native(inputs, conv_state, state):
    rows = inputs["hidden_states"].shape[0]
    projected = tgemm.mm(
        inputs["hidden_states"],
        inputs["merged_input_weight"],
        None,
        otype=torch.bfloat16,
    )
    mixed_qkv = projected[:, :CONV]
    output_gate = projected[:, CONV:QKVG].view(rows, HEADS, HEAD_DIM)
    f_a = projected[:, QKVG : QKVG + HEAD_DIM]
    raw_beta = projected[:, QKVG + HEAD_DIM : QKVG + HEAD_DIM + HEADS]
    forget_gate = kimi_k3_tiny_gemm(f_a, inputs["forget_weight"])
    qkv = causal_conv1d_update(
        mixed_qkv,
        conv_state.transpose(-1, -2),
        inputs["conv_weight"],
        None,
        activation="silu",
        conv_state_indices=inputs["state_indices"],
    )
    q, k, v = qkv.split([LOCAL, LOCAL, LOCAL], dim=-1)
    q = q.view(1, rows, HEADS, HEAD_DIM)
    k = k.view(1, rows, HEADS, HEAD_DIM)
    v = v.view(1, rows, HEADS, HEAD_DIM)
    core = fused_sigmoid_gating_delta_rule_update(
        A_log=inputs["A_log"],
        a=forget_gate,
        dt_bias=inputs["dt_bias"],
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=raw_beta.unsqueeze(0),
        initial_state_source=state,
        initial_state_indices=inputs["state_indices"],
        cu_seqlens=inputs["query_start_loc"],
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=-5.0,
    )
    core = inputs["norm"](core, output_gate)
    return tgemm.mm(
        core.squeeze(0).flatten(-2),
        inputs["output_weight"],
        None,
        otype=torch.bfloat16,
    )


def _whole(inputs, conv_state, state, output_tensor=None):
    return whole_kda.run(
        hidden_states=inputs["hidden_states"],
        qkvg_weight=inputs["qkvg_weight"],
        beta_forget_weight=inputs["beta_forget_weight"],
        output_weight=inputs["output_weight"],
        forget_weight=inputs["forget_weight"],
        conv_weight=inputs["conv_weight"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        norm_weight=inputs["norm_weight"],
        conv_state=conv_state,
        state=state,
        state_indices=inputs["state_indices"],
        lower_bound=-5.0,
        norm_eps=1e-5,
        output_tensor=output_tensor,
    )[0]


def _errors(actual: torch.Tensor, expected: torch.Tensor):
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    denom = expected_f.abs().clamp_min(1e-5)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    return diff.max().item(), (diff / denom).mean().item(), cosine.item()


def benchmark(rows: int, warmup: int, rep: int):
    inputs = _make_inputs(rows)

    native_conv = inputs["conv_state"].clone()
    native_state = inputs["state"].clone()
    whole_conv = inputs["conv_state"].clone()
    whole_state = inputs["state"].clone()
    expected = _native(inputs, native_conv, native_state)
    actual = _whole(inputs, whole_conv, whole_state)
    torch.cuda.synchronize()

    out_max, out_mean_rel, out_cos = _errors(actual, expected)
    conv_max, _, conv_cos = _errors(whole_conv, native_conv)
    state_max, _, state_cos = _errors(whole_state, native_state)
    native_conv_absmax = native_conv.float().abs().max().item()
    whole_conv_absmax = whole_conv.float().abs().max().item()
    native_state_absmax = native_state.float().abs().max().item()
    whole_state_absmax = whole_state.float().abs().max().item()
    if not all(math.isfinite(value) for value in (out_cos, conv_cos, state_cos)):
        raise RuntimeError("non-finite whole-layer KDA validation result")

    native_conv.zero_()
    native_state.zero_()
    whole_conv.zero_()
    whole_state.zero_()
    output_tensor = torch.empty((rows, HIDDEN), device="cuda", dtype=torch.bfloat16)
    native_ms = triton.testing.do_bench(
        lambda: _native(inputs, native_conv, native_state), warmup=warmup, rep=rep
    )
    whole_ms = triton.testing.do_bench(
        lambda: _whole(inputs, whole_conv, whole_state, output_tensor),
        warmup=warmup,
        rep=rep,
    )
    return {
        "rows": rows,
        "native_ms": native_ms,
        "whole_ms": whole_ms,
        "speedup": native_ms / whole_ms,
        "out_max_abs": out_max,
        "out_mean_rel": out_mean_rel,
        "out_cosine": out_cos,
        "native_out_absmax": expected.float().abs().max().item(),
        "whole_out_absmax": actual.float().abs().max().item(),
        "conv_max_abs": conv_max,
        "conv_cosine": conv_cos,
        "native_conv_absmax": native_conv_absmax,
        "whole_conv_absmax": whole_conv_absmax,
        "state_max_abs": state_max,
        "state_cosine": state_cos,
        "native_state_absmax": native_state_absmax,
        "whole_state_absmax": whole_state_absmax,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)

    keys = (
        "rows",
        "native_ms",
        "whole_ms",
        "speedup",
        "out_max_abs",
        "out_mean_rel",
        "out_cosine",
        "native_out_absmax",
        "whole_out_absmax",
        "conv_max_abs",
        "conv_cosine",
        "native_conv_absmax",
        "whole_conv_absmax",
        "state_max_abs",
        "state_cosine",
        "native_state_absmax",
        "whole_state_absmax",
    )
    print(",".join(keys), flush=True)
    for rows in args.rows:
        result = benchmark(rows, args.warmup, args.rep)
        print(
            ",".join(
                str(result[key]) if key == "rows" else f"{result[key]:.8g}"
                for key in keys
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
