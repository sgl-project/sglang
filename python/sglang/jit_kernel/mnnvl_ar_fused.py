"""sglang jit_kernel module: MNNVL fused allreduce + residual add + rmsnorm.

Verbatim port of flashinfer 0.6.12's `oneshotAllreduceFusionKernel`
(`trtllm_mnnvl_allreduce.cuh`) into the sglang jit_kernel tree; the kernel
source under `csrc/mnnvl_ar_fused/` differs from upstream only in its include
block (flashinfer-internal includes replaced by a local compat shim). Reuses
the existing FlashInfer MNNVL workspace (multicast/unicast pointers + Lamport
buffer flags) — this module swaps only the kernel dispatch.

Enabled at the callsite via SGLANG_JIT_MNNVL_AR=1 (default OFF); see
sglang/srt/layers/flashinfer_comm_fusion.py.
"""

from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch


@cache_once
def _jit_module():
    # sm_103a per the port contract (matches the flashinfer build on this
    # platform and the task harness build; plain 10.3 would drop the arch
    # suffix on B300).
    with override_jit_cuda_arch(10, 3, "a"):
        return load_jit(
            # "_fm" pins the fast-math build flavor in the cache key:
            # load_jit keys on module name + source hash, NOT compiler
            # flags, so without the token a stale IEEE .so built from the
            # same sources would be silently reused (mirrors the harness
            # runner's mnnvl_ar_fused_port_fm convention).
            "mnnvl_ar_fused_fm",
            cuda_files=["mnnvl_ar_fused/mnnvl_ar_fused.cu"],
            extra_dependencies=["flashinfer"],
            # Matches the deployed flashinfer-jit-cache binary's float codegen
            # (FTZ + approx div/sqrt); an IEEE build of the same source flips
            # output zero-signs on subnormal gammas and diverges greedy text.
            extra_cuda_cflags=["--use_fast_math"],
            header_only=False,
        )


def prebuild() -> None:
    """Build/load the module (used to warm the JIT cache before TP spawn)."""
    _jit_module()


def _ffi_fn():
    """The kernel entry: the jit port by default; the stock flashinfer module
    when SGLANG_JIT_MNNVL_AR_USE_STOCK_MODULE=1 (diagnostic bisection: same
    route mechanics, upstream-compiled kernel)."""
    from sglang.srt.environ import envs

    if envs.SGLANG_JIT_MNNVL_AR_USE_STOCK_MODULE.get():
        from flashinfer.comm.trtllm_mnnvl_ar import get_trtllm_mnnvl_comm_module

        return get_trtllm_mnnvl_comm_module().trtllm_mnnvl_allreduce_fusion
    if envs.SGLANG_JIT_MNNVL_AR_OPT.get():
        # bs=1 constant-specialized entry (frozen shapes; generic fallback
        # inside the same call for uncovered shapes)
        return _jit_module().mnnvl_ar_fused_opt
    return _jit_module().trtllm_mnnvl_allreduce_fusion


def allreduce_add_rmsnorm(
    input_tensor: torch.Tensor,
    workspace,
    norm_out: torch.Tensor,
    residual_out: torch.Tensor,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    launch_with_pdl: bool = True,
) -> None:
    """Oneshot fused AR+add+rmsnorm on an existing FlashInfer MNNVL workspace.

    Caller guarantees: mnnvl-backend workspace with sufficient oneshot buffer,
    2D contiguous bf16 tensors, outputs preallocated.
    """
    assert input_tensor.dtype == torch.bfloat16
    assert input_tensor.dim() == 2
    assert gamma.numel() == input_tensor.shape[1]

    _ffi_fn()(
        input_tensor,
        workspace.mc_ptr,
        workspace.uc_ptrs_dev,
        workspace.uc_ptr_local,
        workspace.buffer_flags,
        workspace.tp_size,
        workspace.rank,
        True,  # rmsnorm_fusion
        bool(launch_with_pdl),
        True,  # use_oneshot (callsite routing guarantees the oneshot regime)
        norm_out,
        residual_out,
        residual_in,
        gamma,
        float(eps),
        0.0,  # weight_bias: standard RMSNorm
    )
