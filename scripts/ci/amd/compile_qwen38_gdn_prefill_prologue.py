"""Offline gfx950 compilation check for the Qwen3.8 GDN prefill kernel."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource


def _load_kernel():
    """Load the kernel source without importing the full SGLang package."""
    stub = types.ModuleType("sglang.kernels.jit.utils")
    stub.is_arch_support_pdl = lambda: False
    for name in ("sglang", "sglang.kernels", "sglang.kernels.jit"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sglang.kernels.jit.utils"] = stub

    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/kernels/ops/mamba/causal_conv1d_triton.py"
    )
    spec = importlib.util.spec_from_file_location("_qwen38_causal_conv", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._causal_conv1d_fwd_kernel


_causal_conv1d_fwd_kernel = _load_kernel()


def _signature(
    constants: dict[str, object],
    *,
    dt_bias_type: str,
) -> dict[str, str]:
    pointer_types = {
        "x_ptr": "*bf16",
        "w_ptr": "*bf16",
        "bias_ptr": "*bf16",
        "initial_states_ptr": "*bf16",
        "cache_indices_ptr": "*i32",
        "has_initial_states_ptr": "*i1",
        "query_start_loc_ptr": "*i32",
        "o_ptr": "*bf16",
        "q_ptr": "*bf16",
        "k_ptr": "*bf16",
        "v_ptr": "*bf16",
        "g_ptr": "*fp32",
        "beta_ptr": "*fp32",
        "A_log_ptr": "*fp32",
        "a_ptr": "*bf16",
        "b_ptr": "*bf16",
        "dt_bias_ptr": dt_bias_type,
    }
    signature = {}
    for name in _causal_conv1d_fwd_kernel.arg_names:
        if name in constants:
            signature[name] = "constexpr"
        elif name in pointer_types:
            signature[name] = pointer_types[name]
        elif name == "seqlen":
            signature[name] = "i32"
        else:
            raise RuntimeError(f"offline compile has no type for {name}")
    return signature


def _constants(*, fused: bool) -> dict[str, object]:
    return {
        "dim": 10240,
        "num_cache_lines": 64,
        "stride_x_seq": 0,
        "stride_x_dim": 1,
        "stride_x_token": 16384 if fused else 10240,
        "stride_w_dim": 4,
        "stride_w_width": 1,
        "stride_istate_seq": 30720,
        "stride_istate_dim": 3,
        "stride_istate_token": 1,
        "stride_o_seq": 0,
        "stride_o_dim": 0 if fused else 1,
        "stride_o_token": 0 if fused else 10240,
        "stride_a_token": 96 if fused else 0,
        "stride_a_head": 1 if fused else 0,
        "stride_b_token": 96 if fused else 0,
        "stride_b_head": 1 if fused else 0,
        "pad_slot_id": -1,
        "HAS_BIAS": True,
        "KERNEL_WIDTH": 4,
        "SILU_ACTIVATION": True,
        "HAS_INITIAL_STATES": True,
        "HAS_CACHE": True,
        "IS_CONTINUOUS_BATCHING": True,
        "USE_PAD_SLOT": True,
        "NP2_STATELEN": 4,
        "BLOCK_M": 8,
        "BLOCK_N": 256,
        "FUSE_QWEN38_GDN_PREP": fused,
        "NUM_QK_HEADS": 16 if fused else 0,
        "NUM_V_HEADS": 48 if fused else 0,
        "HEAD_DIM": 128 if fused else 0,
        "SOFTPLUS_BETA": 1.0,
        "SOFTPLUS_THRESHOLD": 20.0,
    }


def _compile(
    target: GPUTarget,
    *,
    fused: bool,
    dt_bias_type: str,
) -> int:
    constants = _constants(fused=fused)
    source = ASTSource(
        _causal_conv1d_fwd_kernel,
        _signature(constants, dt_bias_type=dt_bias_type),
        constants,
    )
    compiled = triton.compile(
        source,
        target=target,
        options={
            "num_warps": 4,
            "num_stages": 2,
        },
    )
    return len(compiled.asm["hsaco"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="gfx950")
    args = parser.parse_args()
    target = GPUTarget("hip", args.arch, 64)
    modes = (
        ("fallback", False, "*bf16"),
        ("fused_bf16_dt", True, "*bf16"),
        ("fused_fp32_dt", True, "*fp32"),
    )
    for mode, fused, dt_bias_type in modes:
        binary_size = _compile(
            target,
            fused=fused,
            dt_bias_type=dt_bias_type,
        )
        print(f"{mode} {args.arch} compile ok: {binary_size} bytes")


if __name__ == "__main__":
    main()
