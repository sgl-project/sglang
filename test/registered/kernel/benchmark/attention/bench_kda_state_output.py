import argparse
import statistics
import sys

import torch
import triton

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fla import chunk_delta_h
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_o_128,
)
from sglang.kernels.ops.attention.fla.chunk_intra import (
    chunk_kda_fwd_kernel_intra_sub_chunk,
)
from sglang.kernels.ops.attention.fla.chunk_intra import (
    is_gather_supported as intra_is_gather_supported,
)
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.kda import (
    _recompute_w_u_fwd_kernel,
    chunk_gla_fwd_kernel_o,
    chunk_gla_fwd_o_gk,
    chunk_kda,
    recompute_w_u_fwd,
    recompute_w_u_fwd_kernel,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="jit-kernel-benchmark-test-amd")


@marker.parametrize("tokens", [128, 256, 512, 768, 1024, 8192], [128, 512, 768])
@marker.parametrize("heads", [8, 16])
@marker.benchmark("implementation", ["current", "fused"])
def benchmark_kda_state_output(tokens: int, heads: int, implementation: str):
    shape = (1, tokens, heads, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    w = torch.randn_like(q)
    g = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.001
    A = torch.randn(
        (1, tokens, heads, 64),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state = torch.randn(
        (1, heads, 128, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state_indices = torch.tensor([0], device="cuda", dtype=torch.int64)
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int64)
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    if implementation == "current":

        def fn():
            h, v_new = chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=v,
                gk=g,
                initial_state=state,
                initial_state_indices=state_indices,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=True,
            )
            return chunk_gla_fwd_o_gk(
                q=q,
                v=v_new,
                g=g,
                A=A,
                h=h,
                o=v,
                scale=128**-0.5,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

    else:

        def fn():
            return chunk_gated_delta_rule_fwd_o_128(
                q=q,
                k=k,
                v=v,
                w=w,
                gk=g,
                A=A,
                scale=128**-0.5,
                initial_state=state,
                initial_state_indices=state_indices,
                cu_seqlens=cu_seqlens,
            )

    return marker.do_bench(
        fn,
        input_args=(),
        graph_clone_args=(),
        graph_clone_kwargs=(),
        memory_args=None,
        memory_output=None,
    )


@marker.parametrize("tokens", [128, 256, 512, 768, 8192], [128, 256])
@marker.parametrize("heads", [8, 16])
@marker.benchmark("implementation", ["current", "fused"])
def benchmark_complete_kda(tokens: int, heads: int, implementation: str):
    shape = (1, tokens, heads, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
    k = torch.randn_like(q) * 0.02
    v = torch.randn_like(q) * 0.02
    g = torch.randn_like(q) * 0.02
    beta = torch.randn(
        (1, tokens, heads),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state = torch.randn(
        (1, heads, 128, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state_indices = torch.tensor([0], device="cuda", dtype=torch.int64)
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int64)
    A_log = torch.randn(
        (1, 1, heads, 1),
        device="cuda",
        dtype=torch.float32,
    )
    dt_bias = torch.randn(heads * 128, device="cuda", dtype=torch.float32)

    def fn():
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            initial_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    original = chunk_delta_h.is_gfx95_supported
    chunk_delta_h.is_gfx95_supported = lambda: implementation == "fused"
    try:
        return marker.do_bench(
            fn,
            input_args=(),
            graph_clone_args=(),
            graph_clone_kwargs=(),
            memory_args=None,
            memory_output=None,
        )
    finally:
        chunk_delta_h.is_gfx95_supported = original


def _parse_profile_args():
    parser = argparse.ArgumentParser(description="Bounded KDA rocprof target")
    parser.add_argument(
        "--rocprof-target",
        choices=("state", "output", "recompute", "intra"),
        required=True,
    )
    parser.add_argument("--heads", type=int, choices=(8, 16), required=True)
    parser.add_argument("--tokens", type=int, choices=(8192, 131072), required=True)
    parser.add_argument(
        "--state-config",
        choices=("original", "tuned"),
        default="tuned",
    )
    parser.add_argument(
        "--output-config",
        choices=("auto", "static"),
        default="auto",
    )
    parser.add_argument("--output-bk", type=int, choices=(32, 64), default=64)
    parser.add_argument("--output-bv", type=int, choices=(64, 128), default=128)
    parser.add_argument("--output-warps", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--output-stages", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument(
        "--recompute-config",
        choices=("auto", "static"),
        default="auto",
    )
    parser.add_argument("--recompute-bk", type=int, choices=(64, 128), default=128)
    parser.add_argument("--recompute-bv", type=int, choices=(64, 128), default=128)
    parser.add_argument("--recompute-warps", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--recompute-stages", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument(
        "--intra-config",
        choices=("auto", "static"),
        default="auto",
    )
    parser.add_argument("--intra-warps", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--intra-stages", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--launches", type=int, default=5)
    return parser.parse_args()


def _run_profile_target(args):
    generator = torch.Generator(device="cuda").manual_seed(args.tokens + args.heads)
    shape = (1, args.tokens, args.heads, 128)

    def randn(*size, dtype=torch.bfloat16, scale=0.02):
        return (
            torch.randn(
                *size,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * scale
        )

    q = randn(*shape)
    k = randn(*shape)
    v = randn(*shape)
    w = randn(*shape)
    g = randn(*shape, dtype=torch.float32, scale=0.001)
    beta = torch.sigmoid(randn(1, args.tokens, args.heads).float())
    A = randn(1, args.tokens, args.heads, 64)
    state = randn(1, args.heads, 128, 128, scale=0.01)
    state_seed = state.clone()
    state_indices = torch.tensor([0], device="cuda", dtype=torch.int64)
    cu_seqlens = torch.tensor(
        [0, args.tokens],
        device="cuda",
        dtype=torch.int64,
    )
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)

    original_gfx95_check = chunk_delta_h.is_gfx95_supported
    chunk_delta_h.is_gfx95_supported = lambda: args.state_config == "tuned"

    def state_fn():
        return chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=v,
            gk=g,
            initial_state=state,
            initial_state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )

    state.copy_(state_seed)
    h, v_new = state_fn()
    output = torch.empty_like(v)

    def output_fn():
        if args.output_config == "static":
            grid = (
                triton.cdiv(128, args.output_bv),
                len(chunk_indices),
                args.heads,
            )
            chunk_gla_fwd_kernel_o.fn[grid](
                q=q,
                v=v_new,
                g=g,
                h=h,
                o=output,
                A=A,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                scale=128**-0.5,
                T=args.tokens,
                H=args.heads,
                K=128,
                V=128,
                BT=64,
                BK=args.output_bk,
                BV=args.output_bv,
                IS_VARLEN=True,
                num_warps=args.output_warps,
                num_stages=args.output_stages,
            )
            return output
        return chunk_gla_fwd_o_gk(
            q=q,
            v=v_new,
            g=g,
            A=A,
            h=h,
            o=output,
            scale=128**-0.5,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    recompute_w = torch.empty_like(k)
    recompute_u = torch.empty_like(v)
    recompute_kg = torch.empty_like(k)

    def recompute_fn():
        if args.recompute_config == "static":
            grid = (len(chunk_indices), args.heads)
            _recompute_w_u_fwd_kernel[grid](
                k=k,
                kg=recompute_kg,
                v=v,
                beta=beta,
                w=recompute_w,
                u=recompute_u,
                A=A,
                gk=g,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                T=args.tokens,
                H=args.heads,
                K=128,
                V=128,
                BT=64,
                BK=args.recompute_bk,
                BV=args.recompute_bv,
                STORE_KG=True,
                IS_VARLEN=True,
                DOT_PRECISION="ieee",
                num_warps=args.recompute_warps,
                num_stages=args.recompute_stages,
            )
            return recompute_w, recompute_u, recompute_kg
        return recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            gk=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    intra_Aqk = torch.empty_like(A)
    intra_Akk = torch.empty(
        (1, args.tokens, args.heads, 16),
        device="cuda",
        dtype=torch.float32,
    )

    def intra_fn():
        grid = (len(chunk_indices), 4, args.heads)
        kernel = (
            chunk_kda_fwd_kernel_intra_sub_chunk.fn.fn
            if args.intra_config == "static"
            else chunk_kda_fwd_kernel_intra_sub_chunk
        )
        launch_kwargs = (
            {
                "num_warps": args.intra_warps,
                "num_stages": args.intra_stages,
            }
            if args.intra_config == "static"
            else {}
        )
        kernel[grid](
            q=q,
            k=k,
            g=g,
            beta=beta,
            Aqk=intra_Aqk,
            Akk=intra_Akk,
            scale=128**-0.5,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=args.tokens,
            H=args.heads,
            K=128,
            BT=64,
            BC=16,
            BK=128,
            IS_VARLEN=True,
            USE_GATHER=intra_is_gather_supported,
            **launch_kwargs,
        )
        return intra_Aqk, intra_Akk

    target_functions = {
        "state": state_fn,
        "output": output_fn,
        "recompute": recompute_fn,
        "intra": intra_fn,
    }
    target_fn = target_functions[args.rocprof_target]
    range_name = f"kda_{args.rocprof_target}"
    try:
        for _ in range(10):
            if args.rocprof_target == "state":
                state.copy_(state_seed)
            target_fn()
        torch.cuda.synchronize()
        selected_output_config = getattr(
            chunk_gla_fwd_kernel_o,
            "best_config",
            None,
        )
        selected_recompute_config = getattr(
            recompute_w_u_fwd_kernel,
            "best_config",
            None,
        )
        selected_intra_config = getattr(
            chunk_kda_fwd_kernel_intra_sub_chunk.fn,
            "best_config",
            None,
        )
        timings = []
        for _ in range(args.launches):
            if args.rocprof_target == "state":
                state.copy_(state_seed)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(range_name)
            start.record()
            target_fn()
            end.record()
            torch.cuda.nvtx.range_pop()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        print(
            {
                "target": args.rocprof_target,
                "heads": args.heads,
                "tokens": args.tokens,
                "state_config": args.state_config,
                "launches": args.launches,
                "median_ms": statistics.median(timings),
                "output_config": (
                    str(selected_output_config)
                    if args.output_config == "auto"
                    else {
                        "BK": args.output_bk,
                        "BV": args.output_bv,
                        "num_warps": args.output_warps,
                        "num_stages": args.output_stages,
                    }
                ),
                "recompute_config": (
                    str(selected_recompute_config)
                    if args.recompute_config == "auto"
                    else {
                        "BK": args.recompute_bk,
                        "BV": args.recompute_bv,
                        "num_warps": args.recompute_warps,
                        "num_stages": args.recompute_stages,
                    }
                ),
                "intra_config": (
                    str(selected_intra_config)
                    if args.intra_config == "auto"
                    else {
                        "num_warps": args.intra_warps,
                        "num_stages": args.intra_stages,
                    }
                ),
            }
        )
    finally:
        chunk_delta_h.is_gfx95_supported = original_gfx95_check


if __name__ == "__main__":
    if "--rocprof-target" in sys.argv:
        _run_profile_target(_parse_profile_args())
    else:
        benchmark_kda_state_output.run()
        benchmark_complete_kda.run()
