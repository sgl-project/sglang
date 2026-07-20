"""Probe the AITER A8W8 preshuffle routes used by DSV4 Pro CUDA graphs."""

import argparse
import subprocess
import sys
import traceback

import torch

SHAPES = {
    # The new tuned CSV routes this shape from ASM to Triton in AITER #4202.
    # The DSV4 Pro job reports a GPU memory access fault while capturing bs=144.
    "pro": (144, 8192, 1536),
    # AITER #4264 routes this shape from CK to a Triton BM128/BN32/KSPLIT8
    # config, matching the compiler assertion in the DSV4 Pro-MTP job.
    "pro-mtp": (128, 2048, 7168),
}


def run_dispatch(shape_name: str) -> None:
    import aiter
    from aiter.jit.core import AITER_CONFIGS
    from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import dtypes

    m, n, k = SHAPES[shape_name]
    tuned_file = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    config = get_CKGEMM_config(m, n, k, tuned_file)
    commit = subprocess.check_output(
        ["git", "-C", "/sgl-workspace/aiter", "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"ROUTE shape={shape_name} M={m} N={n} K={k} commit={commit} "
        f"config_file={tuned_file} config={config}",
        flush=True,
    )

    scale_n = (n + 127) // 128
    scale_k = (k + 127) // 128
    x = (torch.rand((m, k), dtype=torch.float32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=torch.float32, device="cuda") / 10).to(
        dtypes.fp8
    )
    x_scale = torch.rand((m, scale_k), dtype=torch.float32, device="cuda")
    x_scale = x_scale.transpose(0, 1).contiguous().view(m, scale_k)
    w_scale = torch.rand((scale_n, scale_k), dtype=torch.float32, device="cuda")
    weight = shuffle_weight(weight, layout=(16, 16))

    def gemm():
        return aiter.gemm_a8w8_blockscale_bpreshuffle(
            x, weight, x_scale, w_scale, dtypes.bf16
        )

    # Surface asynchronous compiler/launch failures before graph capture.
    eager_output = None
    for _ in range(3):
        eager_output = gemm()
    torch.cuda.synchronize()
    if eager_output is None or not torch.isfinite(eager_output).all().item():
        raise RuntimeError("eager output is missing or non-finite")
    print("EAGER_SUCCESS", flush=True)

    # The production failures occur while DSV4 captures its decode CUDA graphs.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = gemm()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    if not torch.isfinite(graph_output).all().item():
        raise RuntimeError("CUDA graph output is non-finite")
    checksum = graph_output.float().abs().mean().item()
    print(f"GRAPH_SUCCESS checksum={checksum:.8f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", choices=sorted(SHAPES), required=True)
    args = parser.parse_args()

    try:
        run_dispatch(args.shape)
    except BaseException:
        print(f"PROBE_FAILED shape={args.shape}", flush=True)
        traceback.print_exc()
        return 2
    print(f"PROBE_SUCCESS shape={args.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
