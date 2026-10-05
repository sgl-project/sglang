"""Cake (FlashInfer) backends for the ``gemm`` operator group.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels`, which import FlashInfer only
when a kernel is actually called. Every explicit ``cake_*`` entry point below
resolves the FlashInfer backend of its op; callers gate on the adapter's
``supports_*`` check first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_CAKE = "sglang.kernels.cake_kernels"
_SM100_ONLY = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 0))})
_SM100_SM103 = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3))})
# sm_100a / sm_103a / sm_107a; the adapter's supports_* pins the exact set.
_SM100_SM107 = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 7))})


def _spec(
    name: str,
    module: str,
    function: str,
    capabilities: frozenset,
    dtypes: tuple,
    signature: str,
    description: str,
    *,
    in_place: bool = False,
) -> KernelSpec:
    return KernelSpec(
        op=f"gemm.{name}",
        backend=KernelBackend.FLASHINFER,
        target=f"{_CAKE}.{module}:{function}",
        capabilities=capabilities,
        format_signature=FormatSignature(
            supported_dtypes=dtypes, in_place=in_place, description=signature
        ),
        description=description,
    )


# --- contiguous grouped FP8 GEMM (SM100a only) -------------------------------
register_kernel(
    _spec(
        "prepare_group_gemm_fp8_nt_groupwise_contiguous",
        "gemm_grouped_fp8",
        "prepare_group_gemm_fp8_nt_groupwise_contiguous",
        _SM100_ONLY,
        ("float8_e4m3fn", "float32", "int32", "bfloat16"),
        "prepared runner: E4M3 a[M,K] x b[G,N,K] with (M,K/128)/(G,N/128,K/128) "
        "FP32 scales routed by sorted int32 m_indices[M] -> BF16 out[M,N]; "
        "launch() submits one kernel (first launch not graph-capturable)",
        "Cake contiguous grouped FP8 GEMM (prepare-once / launch-many) distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant",
        "gemm_grouped_fp8",
        "prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant",
        _SM100_ONLY,
        ("float8_e4m3fn", "float32", "int32"),
        "prepared runner: grouped FP8 gate_up GEMM b[G,2H,K] + SwiGLU + per-128-col "
        "FP8 quant -> (E4M3 out_q[M,H], FP32 out_s[M,H/128]); M <= 8192, K % 512 == 0, "
        "2H % 256 == 0, internal expert boundaries % 128 == 0",
        "Cake fused grouped FP8 gate_up GEMM + SwiGLU + FP8 quant distributed by FlashInfer.",
    )
)

# --- per-token NVFP4 GEMM (SM100a / SM103a) ----------------------------------
register_kernel(
    _spec(
        "mm_fp4_per_token",
        "gemm_nvfp4_per_token",
        "mm_fp4_per_token",
        _SM100_SM103,
        ("uint8", "float8_e4m3fn", "float32", "bfloat16", "float16"),
        "mm_fp4(backend='cake'): packed E2M1 a[M,K/2] x b=b_fp4.T with 128x4 E4M3 "
        "scales and per-token FP32 alpha[M] -> bf16/fp16 out[M,N]; N % 8, K % 256",
        "Cake per-token-alpha NVFP4 GEMM (mm_fp4 backend='cake') distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "prepare_mm_fp4_per_token",
        "gemm_nvfp4_per_token",
        "prepare_mm_fp4_per_token",
        _SM100_SM103,
        ("uint8", "float8_e4m3fn", "float32", "bfloat16", "float16"),
        "prepared runner of the per-token NVFP4 GEMM over a contiguous [N,K/2] weight; "
        "launch() allocation-free and graph-capturable",
        "Cake per-token NVFP4 GEMM runner (prepare-once / launch-many) distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "allocate_nvfp4_per_token_quantize_outputs",
        "gemm_nvfp4_per_token",
        "allocate_nvfp4_per_token_quantize_outputs",
        _SM100_SM103,
        ("uint8", "float32"),
        "(m, k, device) -> PerTokenQuantizeOutputs(fp4[M,K/2], sf[pad128(M),pad4(K/16)], scale[M]); "
        "workspace of the quantize + GEMM chain",
        "Workspace allocator of the Cake per-token NVFP4 chain distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "prepare_nvfp4_per_token_chain",
        "gemm_nvfp4_per_token",
        "prepare_nvfp4_per_token_chain",
        _SM100_SM103,
        ("bfloat16", "float16", "uint8", "float32"),
        "prepared runner: per-token NVFP4 quantize of x[M,K] into the workspace then the "
        "GEMM against b_fp4[N,K/2] -> out[M,N]; launch() graph-capturable",
        "Cake fused per-token NVFP4 linear (quantizer + GEMM chain) distributed by FlashInfer.",
    )
)

# --- Kimi-K3 FP8_PB_WO projections (SM100a / SM103a) --------------------------
register_kernel(
    _spec(
        "prepare_kimi_k3_fp8_projection_weights",
        "gemm_kimi_k3_fp8_projection",
        "prepare_kimi_k3_fp8_projection_weights",
        _SM100_SM103,
        ("float8_e4m3fn", "float32"),
        "E4M3 weight[N_pad128,K] + ModelOpt FP32 block scale [N/128,1,K/128,1] -> "
        "PreparedProjectionWeight (UE8M0 requantized, TMA-tiled); once per weight",
        "Weight preparation of the Cake Kimi-K3 FP8 projection distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "allocate_kimi_k3_fp8_projection_workspace",
        "gemm_kimi_k3_fp8_projection",
        "allocate_kimi_k3_fp8_projection_workspace",
        _SM100_SM103,
        ("float8_e4m3fn", "uint8"),
        "(prepared, M) -> ProjectionWorkspace(q[M,K] E4M3, sf bytes); once per M, no launch",
        "Workspace allocator of the Cake Kimi-K3 FP8 projection distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "prepare_kimi_k3_fp8_projection",
        "gemm_kimi_k3_fp8_projection",
        "prepare_kimi_k3_fp8_projection",
        _SM100_SM103,
        ("bfloat16", "float8_e4m3fn", "uint8"),
        "prepared runner: BF16 x[M,K] -> per-token 1x128 E4M3/UE8M0 quant -> block-scaled "
        "GEMM -> BF16 out[M,n_valid] view; launch() allocation-free, graph-capturable",
        "Cake Kimi-K3 FP8 projection runner (prepare-once / launch-many) distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "kimi_k3_fp8_projection_launcher",
        "gemm_kimi_k3_fp8_projection",
        "kimi_k3_fp8_projection_launcher",
        _SM100_SM103,
        ("bfloat16", "float8_e4m3fn", "uint8"),
        "(prepared) -> launcher(x[M,K] BF16, out=None) -> out[M,n_valid]; route plan cached per "
        "(M, out stride/alignment), workspace cached per M (LRU), call binds tensors only; graph-capturable",
        "Cake Kimi-K3 FP8 projection per-weight cached launcher distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "kimi_k3_fp8_projection",
        "gemm_kimi_k3_fp8_projection",
        "kimi_k3_fp8_projection",
        _SM100_SM103,
        ("bfloat16", "float8_e4m3fn"),
        "one-shot out[M,n_valid] = bf16(x @ dequant(weight).T); allocates out/workspace when omitted",
        "Cake Kimi-K3 FP8 projection (allocating one-shot form) distributed by FlashInfer.",
    )
)

# --- BF16 batched matmul (SM100a / SM103a) -----------------------------------
register_kernel(
    _spec(
        "bmm_bf16",
        "gemm_bmm_bf16",
        "bmm_bf16",
        _SM100_SM103,
        ("bfloat16",),
        "bmm_bf16(backend='cake'): BF16 A[B,M,K] row-major x B[B,K,N] exact column-major "
        "view -> bf16/fp16/fp32 out[B,M,N]; N % 8 == 0, K in {64, 256, 1024}",
        "Cake BF16 batched matmul (bmm_bf16 backend='cake') distributed by FlashInfer.",
    )
)

# --- MoE router GEMMs (Cake route on SM100a / SM103a) -------------------------
register_kernel(
    _spec(
        "mm_m1_16_k7168_n128",
        "gemm_router",
        "mm_m1_16_k7168_n128",
        _SM100_SM103,
        ("bfloat16",),
        "Mistral Large 3 router: bf16 mat_a[M,7168] (1 <= M <= 16) x bf16 column-major "
        "mat_b[7168,128] -> bf16 out[M,128] in place",
        "Cake router GEMM K7168/N128 distributed by FlashInfer.",
        in_place=True,
    )
)
register_kernel(
    _spec(
        "mm_m1_16_k7168_n256",
        "gemm_router",
        "mm_m1_16_k7168_n256",
        _SM100_SM103,
        ("bfloat16", "float32"),
        "DeepSeek-V3 router: bf16 mat_a[M,7168] (1 <= M <= 16) x bf16 column-major "
        "mat_b[7168,256] -> fp32 out[M,256] in place",
        "Cake router GEMM K7168/N256 distributed by FlashInfer.",
        in_place=True,
    )
)
register_kernel(
    _spec(
        "mm_m1_16_k6144_n256",
        "gemm_router",
        "mm_m1_16_k6144_n256",
        _SM100_SM103,
        ("bfloat16", "float32"),
        "GLM-MoE-DSA router: bf16 mat_a[M,6144] (1 <= M <= 16) x bf16 column-major "
        "mat_b[6144,256] -> fp32 out[M,256] in place",
        "Cake router GEMM K6144/N256 distributed by FlashInfer.",
        in_place=True,
    )
)

# --- SVDQuant NVFP4 GEMM (SM100a / SM103a, CUDA >= 13) -----------------------
register_kernel(
    _spec(
        "mm_nvfp4_svdquant",
        "gemm_svdquant",
        "mm_nvfp4_svdquant",
        _SM100_SM103,
        ("uint8", "bfloat16", "float32"),
        "mm_nvfp4_svdquant(backend='cake'): out = alpha * (a @ b^T + d @ l1^T) [+ bias]; "
        "packed E2M1 a[m,k/2], b[n,k/2], 128x4 UE4M3 scales, bf16 d[m,r]/l1[n,r], "
        "rank % 32 == 0; catalogued (M,N,K,rank,bias) routes only",
        "Cake SVDQuant NVFP4 GEMM (mm_nvfp4_svdquant backend='cake') distributed by FlashInfer.",
    )
)

# --- ragged BF16 grouped GEMM forward (sm_100a / sm_103a / sm_107a) ----------
register_kernel(
    _spec(
        "grouped_mm_bf16",
        "gemm_grouped_bf16",
        "grouped_mm_bf16",
        _SM100_SM107,
        ("bfloat16", "int32"),
        "grouped_mm_bf16(backend='cake'): out[m_indptr[e]:m_indptr[e+1]] = a[..] @ b[e].T; "
        "bf16 a[cum_m,K], b[E,N,K], device int32 m_indptr[E+1]; N % 256, K % 64; bf16 out only",
        "Cake ragged BF16 grouped GEMM (grouped_mm_bf16 backend='cake') distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "grouped_gemm_fwd",
        "gemm_grouped_bf16",
        "grouped_gemm_fwd",
        _SM100_SM107,
        ("bfloat16", "int32"),
        "Y[offs[e-1]:offs[e]] = X[offs[e-1]:offs[e]] @ W[e].T; bf16 x[sum_m,K], w[E,N,K], "
        "device int32 offs[E] (or m_indptr[E+1]); one-shot prepare + launch",
        "Cake ragged BF16 grouped GEMM forward (experimental one-shot) distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "prepare_grouped_gemm_fwd",
        "gemm_grouped_bf16",
        "prepare_grouped_gemm_fwd",
        _SM100_SM107,
        ("bfloat16", "int32"),
        "prepared GroupedGemmLaunch of the ragged BF16 forward; launch() allocation-free, "
        "graph-capturable, follows new offs written on device",
        "Cake ragged BF16 grouped GEMM forward runner (prepare-once / launch-many) distributed by FlashInfer.",
    )
)

# --- DeepGEMM-family prepared plans (batched / fp4: exported routes pinned to
# SM100a 148 SMs / SM103a 152 SMs; fp8 1d1d, k-grouped fp4, mixed fp8 x fp4:
# runtime shapes since FlashInfer e4f94f948) ------------------------------------
register_kernel(
    _spec(
        "prepare_fp8_batched_gemm",
        "gemm_deepgemm",
        "prepare_fp8_batched_gemm",
        _SM100_SM103,
        ("float8_e4m3fn", "float32", "bfloat16", "int32"),
        "prepared plan: E4M3 A[T,H,K] x B[H,N,K] with pow2 FP32 block scales -> BF16 "
        "(optional alpha) or dynamic FP8 (E4M3 + int32 UE8M0 words) [T,H,N]; exported "
        "H=8, K=4096, N=1024 routes only",
        "Cake/DeepGEMM per-head FP8 batched projection (prepared plan) distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "prepare_fp4_gemm",
        "gemm_deepgemm",
        "prepare_fp4_gemm",
        _SM100_SM103,
        ("uint8", "int8", "int32", "uint32", "bfloat16"),
        "prepared plan: packed E2M1 A[storage_M,K/2] x B[N,K/2] with packed UE8M0 scale "
        "words [K/128,MN] -> BF16 [m,N]; exported (M,N,K) routes only",
        "Cake/DeepGEMM native FP4 GEMM (prepared plan) distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "prepare_fp8_gemm_1d1d",
        "gemm_deepgemm",
        "prepare_fp8_gemm_1d1d",
        _SM100_SM103,
        ("uint8", "uint32", "bfloat16", "float32"),
        "prepared plan: any M/N, K % 128 == 0, E4M3 bytes with MN-major packed UE8M0 "
        "words -> BF16 out, or accumulate=True FP32 D += A @ B^T; JIT-built CUDA source",
        "Cake/DeepGEMM FP8 1D1D GEMM (prepared runtime-shape plan) distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "prepare_fp4_k_grouped_gemm",
        "gemm_deepgemm",
        "prepare_fp4_k_grouped_gemm",
        _SM100_SM103,
        ("uint8", "int8", "int32", "uint32", "bfloat16", "float32"),
        "prepared plan: independent K-group products A_g @ B_g.T over packed E2M1 with "
        "packed UE8M0 scales -> out[groups,physical_M,N] bf16/fp32 (optional in-place "
        "accumulate); any group layout, num_stages must be 7",
        "Cake/DeepGEMM K-grouped FP4 GEMM (prepared plan) distributed by FlashInfer.",
    )
)
register_kernel(
    _spec(
        "prepare_fp8_fp4_gemm",
        "gemm_deepgemm",
        "prepare_fp8_fp4_gemm",
        _SM100_SM103,
        ("float8_e4m3fn", "uint8", "int8", "int32", "uint32", "bfloat16"),
        "prepared plan: E4M3 A[storage_M,K] x packed E2M1 B[N,K/2] with packed UE8M0 "
        "scales -> BF16 [m,N]; runtime shapes, gran_k_a 32 or 128",
        "Cake/DeepGEMM mixed FP8 x FP4 GEMM (prepared plan) distributed by FlashInfer.",
    )
)


def _cake(name: str):
    return get_kernel(f"gemm.{name}", KernelBackend.FLASHINFER)


def cake_prepare_group_gemm_fp8_nt_groupwise_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    validate_indices: bool = False,
) -> Any:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _cake("prepare_group_gemm_fp8_nt_groupwise_contiguous")(
        a, b, a_scale, b_scale, m_indices, out, validate_indices=validate_indices
    )


def cake_prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indices: torch.Tensor,
    out_q: Optional[torch.Tensor] = None,
    out_s: Optional[torch.Tensor] = None,
    *,
    validate_indices: bool = False,
) -> Any:
    """Explicit Cake entry point; callers gate on the adapter's ``supports_*``."""
    return _cake("prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant")(
        a,
        b,
        a_scale,
        b_scale,
        m_indices,
        out_q,
        out_s,
        validate_indices=validate_indices,
    )


def cake_mm_fp4_per_token(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    use_8x4_sf_layout: bool = False,
    use_nvfp4: bool = True,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """Explicit Cake entry point (``mm_fp4`` backend ``"cake"``)."""
    return _cake("mm_fp4_per_token")(
        a,
        b,
        a_descale,
        b_descale,
        alpha,
        out_dtype,
        out,
        block_size,
        use_8x4_sf_layout,
        use_nvfp4,
        enable_pdl,
    )


def cake_prepare_mm_fp4_per_token(
    a_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    out: torch.Tensor,
    *,
    tactic: Optional[dict] = None,
) -> Any:
    """Explicit Cake entry point; returns an ``NVFP4PerTokenGemmRunner``."""
    return _cake("prepare_mm_fp4_per_token")(
        a_fp4, a_sf, b_fp4, b_sf, alpha, out, tactic=tactic
    )


def cake_allocate_nvfp4_per_token_quantize_outputs(
    m: int, k: int, device: torch.device
) -> Any:
    """Explicit Cake entry point; returns ``PerTokenQuantizeOutputs``."""
    return _cake("allocate_nvfp4_per_token_quantize_outputs")(m, k, device)


def cake_prepare_nvfp4_per_token_chain(
    x: torch.Tensor,
    global_scale_inv: torch.Tensor,
    b_fp4: torch.Tensor,
    b_sf: torch.Tensor,
    out: torch.Tensor,
    workspace: Any,
    *,
    out_scale: Optional[torch.Tensor] = None,
) -> Any:
    """Explicit Cake entry point; returns an ``NVFP4PerTokenChainRunner``."""
    return _cake("prepare_nvfp4_per_token_chain")(
        x, global_scale_inv, b_fp4, b_sf, out, workspace, out_scale=out_scale
    )


def cake_prepare_kimi_k3_fp8_projection_weights(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    n_valid: Optional[int] = None,
    *,
    splits: Optional[Sequence[int]] = None,
) -> Any:
    """Explicit Cake entry point; returns a ``PreparedProjectionWeight``."""
    return _cake("prepare_kimi_k3_fp8_projection_weights")(
        weight, weight_scale, n_valid, splits=splits
    )


def cake_allocate_kimi_k3_fp8_projection_workspace(prepared: Any, M: int) -> Any:
    """Explicit Cake entry point; returns a ``ProjectionWorkspace``."""
    return _cake("allocate_kimi_k3_fp8_projection_workspace")(prepared, M)


def cake_prepare_kimi_k3_fp8_projection(
    x: torch.Tensor,
    prepared: Any,
    out: torch.Tensor,
    workspace: Any,
) -> Any:
    """Explicit Cake entry point; returns a ``KimiK3Fp8ProjectionRunner``."""
    return _cake("prepare_kimi_k3_fp8_projection")(x, prepared, out, workspace)


def cake_kimi_k3_fp8_projection_launcher(prepared: Any, *, max_workspaces: int = 64) -> Any:
    """Explicit Cake entry point; returns a ``KimiK3Fp8ProjectionLauncher``."""
    return _cake("kimi_k3_fp8_projection_launcher")(prepared, max_workspaces=max_workspaces)


def cake_kimi_k3_fp8_projection(
    x: torch.Tensor,
    prepared: Any,
    out: Optional[torch.Tensor] = None,
    *,
    workspace: Any = None,
) -> torch.Tensor:
    """Explicit Cake entry point (allocating one-shot projection)."""
    return _cake("kimi_k3_fp8_projection")(x, prepared, out, workspace=workspace)


def cake_bmm_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Explicit Cake entry point (``bmm_bf16`` backend ``"cake"``)."""
    return _cake("bmm_bf16")(A, B, out, out_dtype)


def cake_mm_m1_16_k7168_n128(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    """Explicit Cake entry point; writes ``out`` in place."""
    _cake("mm_m1_16_k7168_n128")(mat_a, mat_b, out, launch_with_pdl)


def cake_mm_m1_16_k7168_n256(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    """Explicit Cake entry point; writes ``out`` in place."""
    _cake("mm_m1_16_k7168_n256")(mat_a, mat_b, out, launch_with_pdl)


def cake_mm_m1_16_k6144_n256(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    """Explicit Cake entry point; writes ``out`` in place."""
    _cake("mm_m1_16_k6144_n256")(mat_a, mat_b, out, launch_with_pdl)


def cake_mm_nvfp4_svdquant(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: torch.Tensor,
    l1: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Explicit Cake entry point (``mm_nvfp4_svdquant`` backend ``"cake"``)."""
    return _cake("mm_nvfp4_svdquant")(
        a, b, a_sf, b_sf, alpha, d, l1, bias=bias, out=out, enable_pdl=enable_pdl
    )


def cake_grouped_mm_bf16(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    *,
    tactic: int = -1,
) -> torch.Tensor:
    """Explicit Cake entry point (``grouped_mm_bf16`` backend ``"cake"``)."""
    return _cake("grouped_mm_bf16")(a, b, m_indptr, out, out_dtype, tactic=tactic)


def cake_grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit Cake entry point (one-shot ragged BF16 grouped GEMM forward)."""
    return _cake("grouped_gemm_fwd")(x, w, offs, out)


def cake_prepare_grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> Any:
    """Explicit Cake entry point; returns a ``GroupedGemmLaunch``."""
    return _cake("prepare_grouped_gemm_fwd")(x, w, offs, out)


def cake_prepare_fp8_batched_gemm(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    *,
    output_fp8: bool = True,
    alpha: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    output_scales: Optional[torch.Tensor] = None,
    descriptor_workspace: Optional[torch.Tensor] = None,
) -> Any:
    """Explicit Cake entry point; returns a ``BatchedGemmPlan``."""
    return _cake("prepare_fp8_batched_gemm")(
        a,
        b,
        output_fp8=output_fp8,
        alpha=alpha,
        out=out,
        output_scales=output_scales,
        descriptor_workspace=descriptor_workspace,
    )


def cake_prepare_fp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    m: int,
    alpha: float = 1.0,
    out: Optional[torch.Tensor] = None,
    num_stages: Optional[int] = None,
    block_n: int = 128,
    epilogue_store_n: int = 32,
    descriptor_workspace: Optional[torch.Tensor] = None,
) -> Any:
    """Explicit Cake entry point; returns an ``Fp4GemmPlan``."""
    return _cake("prepare_fp4_gemm")(
        a,
        b,
        a_scales,
        b_scales,
        m=m,
        alpha=alpha,
        out=out,
        num_stages=num_stages,
        block_n=block_n,
        epilogue_store_n=epilogue_store_n,
        descriptor_workspace=descriptor_workspace,
    )


def cake_prepare_fp8_gemm_1d1d(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    out: torch.Tensor,
    *,
    accumulate: bool = False,
) -> Any:
    """Explicit Cake entry point; returns an ``Fp8GemmPlan`` (``run()`` writes ``out``)."""
    return _cake("prepare_fp8_gemm_1d1d")(a, b, sfa, sfb, out, accumulate=accumulate)


def cake_prepare_fp4_k_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    m: int,
    group_ks: Sequence[int],
    k_alignment: int = 256,
    use_psum_layout: bool = True,
    output_dtype: str = "bf16",
    accumulate: bool = False,
    num_stages: int = 7,
    out: Optional[torch.Tensor] = None,
) -> Any:
    """Explicit Cake entry point; returns a ``GroupedFP4Plan`` (``run()`` -> ``plan.output``)."""
    return _cake("prepare_fp4_k_grouped_gemm")(
        a,
        b,
        a_scales,
        b_scales,
        m=m,
        group_ks=group_ks,
        k_alignment=k_alignment,
        use_psum_layout=use_psum_layout,
        output_dtype=output_dtype,
        accumulate=accumulate,
        num_stages=num_stages,
        out=out,
    )


def cake_prepare_fp8_fp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    m: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    gran_k_a: int = 32,
) -> Any:
    """Explicit Cake entry point; returns a ``MixedGemmPlan`` (``run()`` -> ``plan.output``)."""
    return _cake("prepare_fp8_fp4_gemm")(
        a, b, a_scales, b_scales, m=m, out=out, gran_k_a=gran_k_a
    )


__all__ = [
    "cake_allocate_kimi_k3_fp8_projection_workspace",
    "cake_allocate_nvfp4_per_token_quantize_outputs",
    "cake_bmm_bf16",
    "cake_grouped_gemm_fwd",
    "cake_grouped_mm_bf16",
    "cake_kimi_k3_fp8_projection",
    "cake_mm_fp4_per_token",
    "cake_mm_m1_16_k6144_n256",
    "cake_mm_m1_16_k7168_n128",
    "cake_mm_m1_16_k7168_n256",
    "cake_mm_nvfp4_svdquant",
    "cake_prepare_fp4_gemm",
    "cake_prepare_fp4_k_grouped_gemm",
    "cake_prepare_fp8_batched_gemm",
    "cake_prepare_fp8_fp4_gemm",
    "cake_prepare_fp8_gemm_1d1d",
    "cake_prepare_group_gemm_fp8_nt_groupwise_contiguous",
    "cake_prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant",
    "cake_prepare_grouped_gemm_fwd",
    "cake_prepare_kimi_k3_fp8_projection",
    "cake_prepare_kimi_k3_fp8_projection_weights",
    "cake_prepare_mm_fp4_per_token",
    "cake_prepare_nvfp4_per_token_chain",
]
