"""Op catalog and dispatch."""

from __future__ import annotations

import dataclasses
import functools
import importlib
from typing import Any, Callable, Mapping, Optional


class UnsupportedOp(RuntimeError):
    """No catalog entry for this op on the current target."""


class UnsupportedBackward(UnsupportedOp):
    """The selected op implementation has no autograd support."""


class InvalidInput(ValueError):
    """An entry exists but cannot handle these inputs."""


@dataclasses.dataclass(frozen=True)
class OpSpec:
    op: str
    arch: str  # must equal Target.key
    variant: str  # label only; not a dispatch input, not user visible
    impl: str  # "absolute.module:attr", imported on first use
    dtypes: frozenset = frozenset()  # bare torch names, so the table needs no torch
    accepts: Optional[Callable[[Mapping[str, Any]], bool]] = None
    requires: frozenset = frozenset()
    supports_backward: bool = False
    default_space: str = "heuristic"  # used when the caller passes space=None
    invariant_vocab: frozenset = frozenset()  # canonical dim names a BitwiseInvarianceSpec may use
    enforces_invariance: bool = False  # the impl prunes schedules per the resolved spec

    def __str__(self) -> str:
        return f"{self.op}/{self.arch} ({self.variant})"


_FP16 = frozenset({"float16", "bfloat16"})
_BF16 = frozenset({"bfloat16"})

# Per-op dim vocabularies for BitwiseInvarianceSpec. Parallel dims tile the
# output; reduction dims order the accumulation. TorchTLX catalog entries take
# no spec (they have no public wrapper) and keep the empty default.
_GEMM_DIMS = frozenset({"M", "N", "K"})
_GROUPED_GEMM_DIMS = frozenset({"G", "M", "N", "K"})
_ATTN_DIMS = frozenset({"B", "H", "SQ", "SKV", "D"})
_VARLEN_DIMS = frozenset({"TQ", "TKV", "H", "D"})
_HSTU_DIMS = frozenset({"B", "LQ", "LKV", "H", "D"})
_KDA_DIMS = frozenset({"T", "H", "D"})
_ROW_DIMS = frozenset({"T", "D"})


def _linear_rowscale_accepts(d):
    # TMA needs 16-byte rows (K, N multiples of 8); x and the weight ring share one CTA's shared memory.
    return 0 < d["K"] <= 512 and d["K"] % 8 == 0 and d["N"] > 0 and d["N"] % 8 == 0


_NORM_MODULES = {
    'sm100': ['layernorm', 'layernorm_add', 'rmsnorm', 'norm_fwd', 'norm_bwd', 'norm_backward_from_normalized'],
    '_softmax': ['softmax', 'softmax_fwd', 'softmax_bwd'], '_fp4': [
        'layernorm_mxfp4', 'rmsnorm_mxfp4', 'layernorm_nvfp4_quantize', 'rmsnorm_nvfp4_quantize',
        'layernorm_nvfp4_quantize_inplace', 'nvfp4_quantize_normalized', 'layernorm_nvfp4_quantize_q0_jagged_transpose',
        'layernorm_nvfp4_quantize_q0_jagged_transpose_inplace', 'compute_norm_nvfp4_global_scale'
    ], '_fp4_bwd': ['layernorm_nvfp4_backward_from_x_norm_padded_dy'], '_mxfp8': [
        'norm_quant_mxfp8_fwd', 'rmsnorm_mxfp8_quantize', 'layernorm_mxfp8_quantize',
        'layernorm_mxfp8_quantize_numerics_preserving', 'layernorm_mxfp8_with_backward',
        'layernorm_mxfp8_gdpa_with_backward', 'layernorm_mxfp8_gdpa_precomputed_no_normalized',
        'rmsnorm_mxfp8_quantize_3232_2pass'
    ], '_mxfp8_bwd': [
        'rmsnorm_backward', 'rmsnorm_backward_from_x_norm', 'layernorm_backward', 'layernorm_backward_from_x_norm',
        'rmsnorm_backward_quantize_dx_mxfp8', 'rmsnorm_backward_quantize_dx_mxfp8_precomputed',
        'rmsnorm_backward_quantize_dx_mxfp8_bf16_precomputed',
        'rmsnorm_backward_quantize_dx_mxfp8_from_output_recompute_q_precomputed'
    ]
}
_NORM_AUTOGRAD = frozenset([
    'layernorm', 'layernorm_add', 'rmsnorm', 'softmax', 'layernorm_mxfp8_with_backward',
    'layernorm_mxfp8_gdpa_with_backward'
])

# A static table, not decorator self-registration: `impl` stays a string so
# `import triton.tlx` never imports a kernel module or builds autotune configs.
# A missing (op, arch) is deferred, not an error -- the op raises on that arch.
CATALOG: tuple[OpSpec, ...] = (
    OpSpec(
        op="mm",
        arch="sm90",
        variant="ws_cooperative",
        impl="triton.tlx.ops.kernels.mm.sm90:mm",
        invariant_vocab=_GEMM_DIMS,
        enforces_invariance=True,
        dtypes=_FP16,
        accepts=lambda d: all(s * d["elem_bytes"] % 16 == 0 for s in d["row_strides"]),
        requires=frozenset({"tma"}),
        # The full space is only 8 configs (seconds to tune) and beats the heuristic on tall shapes.
        default_space="full",
    ),
    OpSpec(
        op="mm",
        arch="sm100",
        variant="ws",
        impl="triton.tlx.ops.kernels.mm.sm100:mm",
        invariant_vocab=_GEMM_DIMS,
        enforces_invariance=True,
        dtypes=_FP16,
        # TMA needs 16-byte-aligned descriptor row strides. Checked against the
        # real strides, not M/N/K: a column-major operand is fed to its
        # descriptor transposed, which moves the constraint to another dim.
        accepts=lambda d: all(s * d["elem_bytes"] % 16 == 0 for s in d["row_strides"]),
        requires=frozenset({"tma", "tmem"}),
        # The heuristic config trails the tuned space (up to ~4x on small shapes,
        # ~5% on 784x3272x147456), so default to the tuned configs; callers opt
        # into "heuristic" for a fast first call.
        default_space="full",
    ),
    OpSpec(
        op="mm",
        arch="gfx942",
        variant="direct_load",
        impl="triton.tlx.ops.kernels.mm.gfx942:mm",
        invariant_vocab=_GEMM_DIMS,
        enforces_invariance=True,
        dtypes=_FP16,
        # No `accepts`: operands are read through explicit strides rather than a
        # descriptor, so there is no alignment rule to fail. This arch therefore
        # admits shapes sm100 declines -- see kernels/mm/_shapes.py.
        requires=frozenset(),
    ),
    OpSpec(
        op="int4_embedding",
        arch="sm90",
        variant="gather_dequant",
        impl="triton.tlx.ops.kernels.int4_embedding.sm90:int4_embedding",
        dtypes=frozenset({"uint8"}),
        accepts=lambda d: d["D"] == 64 and all(p % 16 == 0 for p in d["base_ptrs"]),
        requires=frozenset(),
    ),
    OpSpec(
        op="mm_bias_grad",
        arch="sm100",
        variant="ws_wgrad_bias",
        impl="triton.tlx.ops.kernels.mm_bias_grad.sm100:mm_bias_grad",
        dtypes=frozenset({"bfloat16"}),
        accepts=lambda d: all(s * d["elem_bytes"] % 16 == 0 for s in d["row_strides"]) and all(p % 16 == 0
                                                                                               for p in d["base_ptrs"]),
        requires=frozenset({"tma", "tmem"}),
    ),
    OpSpec(
        op="addmm",
        arch="gfx942",
        variant="fused_gemm",
        impl="triton.tlx.ops.kernels.addmm.gfx942:addmm",
        invariant_vocab=_GEMM_DIMS,
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        op="addmm",
        arch="gfx950",
        variant="fused_gemm",
        impl="triton.tlx.ops.kernels.addmm.gfx950:addmm",
        invariant_vocab=_GEMM_DIMS,
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        op="addmm",
        arch="sm90",
        variant="tma_ws",
        impl="triton.tlx.ops.kernels.addmm.sm90:addmm",
        dtypes=_FP16,
        # Warp-specialized TMA kernel with a direct-load fallback, so no
        # alignment rule to fail and no TMA requirement.
        requires=frozenset(),
        default_space="full",
    ),
    OpSpec(
        op="addmm_mul_add",
        arch="sm90",
        variant="tma_ws",
        impl="triton.tlx.ops.kernels.fused.addmm_mul_add.sm90:addmm_mul_add",
        dtypes=_FP16,
        requires=frozenset({"tma"}),
    ),
    OpSpec(
        op="addmm_mul_add_launch",
        arch="sm90",
        variant="tma_ws",
        impl="triton.tlx.ops.kernels.fused.addmm_mul_add.sm90:addmm_mul_add_launch",
        dtypes=_FP16,
        requires=frozenset({"tma"}),
    ),
    OpSpec(
        op="addmm",
        arch="sm100",
        variant="tma_ws",
        impl="triton.tlx.ops.kernels.addmm.sm100:addmm",
        dtypes=_FP16,
        # Warp-specialized TMA kernel with a direct-load fallback, so no
        # alignment rule to fail and no TMA/TMEM requirement.
        requires=frozenset(),
    ),
    OpSpec(
        op="mm",
        arch="gfx950",
        variant="heuristic",
        impl="triton.tlx.ops.kernels.mm.gfx950:mm",
        invariant_vocab=_GEMM_DIMS,
        enforces_invariance=True,
        # LocalSplitU and persistent plans remain FP16-only; the register and
        # general LDS paths also support BF16 and validate their own plans.
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        op="grouped_gemm",
        arch="sm100",
        variant="ws",
        impl="triton.tlx.ops.kernels.grouped_gemm.sm100:grouped_gemm",
        invariant_vocab=_GROUPED_GEMM_DIMS,
        dtypes=frozenset({"float16"}),
        accepts=lambda d: all(stride * d["elem_bytes"] % 16 == 0
                              for stride in d["row_strides"]) and all(ptr % 16 == 0 for ptr in d["base_ptrs"]),
        requires=frozenset({"tma", "tmem"}),
    ),
    OpSpec(
        op="mm_mxfp8",
        arch="sm100",
        variant="ws_mxfp8",
        impl="triton.tlx.ops.kernels.mm_mxfp8.sm100:mm_mxfp8",
        invariant_vocab=_GEMM_DIMS,
        dtypes=frozenset({"float8_e4m3fn"}),
        # Row strides are 128-aligned by the M/N/K contract; only the bases vary.
        accepts=lambda d: all(ptr % 16 == 0 for ptr in d["base_ptrs"]),
        requires=frozenset({"tma", "tmem"}),
        default_space="full",
    ),
    OpSpec(
        op="mm_mxfp8",
        arch="gfx950",
        variant="scaled_mfma",
        impl="triton.tlx.ops.kernels.mm_mxfp8.gfx950:mm_mxfp8",
        invariant_vocab=_GEMM_DIMS,
        dtypes=frozenset({"float8_e4m3fn"}),
        # A generic load path handles unaligned base pointers. The fast LDS
        # path checks its descriptor span and pointer alignment at launch.
        requires=frozenset(),
    ),
    OpSpec(
        op="mm_mxfp6",
        arch="sm100",
        variant="ws_mxfp6",
        impl="triton.tlx.ops.kernels.mm_mxfp6.sm100:mm_mxfp6",
        dtypes=frozenset({"uint8"}),
        # Row strides are 128-aligned by the M/N/K contract; only the bases vary.
        accepts=lambda d: all(ptr % (32 if i in (0, 2) else 16) == 0 for i, ptr in enumerate(d["base_ptrs"])),
        requires=frozenset({"tma", "tmem"}),
    ),
    # No gfx950 mm_mxfp6 entry yet: the decompose backend in
    # kernels/mm_mxfp6/gfx950.py is not numerically correct at this
    # revision, so MI350 callers get UnsupportedOp instead of silent
    # wrong results. Register it once the fixing diff lands.
    OpSpec(
        op="grouped_gemm_mxfp8",
        arch="sm100",
        variant="ws_persistent_mxfp8",
        impl="triton.tlx.ops.kernels.grouped_gemm_mxfp8.sm100:grouped_gemm_mxfp8",
        invariant_vocab=_GROUPED_GEMM_DIMS,
        dtypes=frozenset({"float8_e4m3fn"}),
        accepts=lambda d: all(row_bytes % 16 == 0 for row_bytes in d["row_bytes"]) and all(ptr % 16 == 0
                                                                                           for ptr in d["base_ptrs"]),
        requires=frozenset({"tma", "tmem"}),
    ),
    OpSpec(
        op="grouped_gemm_mxfp8",
        arch="gfx950",
        variant="persistent_warp_pipeline_mxfp8",
        impl="triton.tlx.ops.kernels.grouped_gemm_mxfp8.gfx950:grouped_gemm_mxfp8",
        invariant_vocab=_GROUPED_GEMM_DIMS,
        dtypes=frozenset({"float8_e4m3fn"}),
        accepts=lambda d: all(row_bytes % 16 == 0 for row_bytes in d["row_bytes"]) and all(ptr % 16 == 0
                                                                                           for ptr in d["base_ptrs"]),
        requires=frozenset(),
    ),
    OpSpec(
        op="grouped_gemm",
        arch="gfx950",
        variant="heuristic",
        impl="triton.tlx.ops.kernels.grouped_gemm.gfx950:grouped_gemm",
        invariant_vocab=_GROUPED_GEMM_DIMS,
        dtypes=frozenset({"float16"}),
        requires=frozenset(),
    ),
    OpSpec(
        op="moe",
        arch="gfx950",
        variant="heuristic",
        impl="triton.tlx.ops.kernels.moe.gfx950:moe",
        dtypes=_BF16,
        # Tiles have no K/N tail; 256-wide N tiles fall back to 128 when they do not divide I or H.
        accepts=lambda d: all(d[dim] > 0 and d[dim] % 128 == 0 for dim in ("K", "I", "H")),
        requires=frozenset(),
    ),
    OpSpec(
        op="mm_torchtlx",
        arch="sm90",
        variant="inductor_hopper_gemm_ws",
        impl="triton.language.extra.tlx.inductor.sm100_torch:mm",
        dtypes=_FP16,
        accepts=lambda d: all(s * d["elem_bytes"] % 16 == 0 for s in d["row_strides"]),
        requires=frozenset({"tma"}),
    ),
    OpSpec(
        # torchTLX: the same mm through torch.compile. Benchmark-only, so it has
        # no `tlx.ops` wrapper; the entry exists so the perf suite can gate on it.
        op="mm_torchtlx",
        arch="sm100",
        variant="inductor_blackwell_gemm_ws",
        impl="triton.language.extra.tlx.inductor.sm100_torch:mm",
        dtypes=_FP16,
        accepts=lambda d: all(s * d["elem_bytes"] % 16 == 0 for s in d["row_strides"]),
        requires=frozenset({"tma", "tmem"}),
    ),
    OpSpec(
        op="mm_torchtlx",
        arch="gfx942",
        variant="inductor_gfx942_mm",
        impl="triton.language.extra.tlx.inductor.gfx942_torch:mm",
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        op="mm_torchtlx",
        arch="gfx950",
        variant="inductor_gfx950_mm",
        impl="triton.language.extra.tlx.inductor.gfx950_torch:mm",
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        # TorchTLX providers are benchmark/catalog entries rather than public
        # wrappers: their API is torch.addmm, with TLX selected by Inductor.
        op="addmm_torchtlx",
        arch="sm90",
        variant="inductor_hopper_gemm_ws",
        impl="triton.tlx.ops.kernels.addmm.sm90_torch:addmm",
        dtypes=_FP16,
        accepts=lambda d: all(s * d["elem_bytes"] % 16 == 0 for s in d["row_strides"]),
        requires=frozenset({"tma"}),
    ),
    OpSpec(
        # TorchTLX providers are benchmark/catalog entries rather than public
        # wrappers: their API is torch.addmm, with TLX selected by Inductor.
        op="addmm_torchtlx",
        arch="gfx942",
        variant="inductor_gfx942_addmm",
        impl="triton.tlx.ops.kernels.addmm.gfx942_torch:addmm",
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        # TorchTLX providers are benchmark/catalog entries rather than public
        # wrappers: their API is torch.addmm, with TLX selected by Inductor.
        op="addmm_torchtlx",
        arch="gfx950",
        variant="inductor_gfx950_addmm",
        impl="triton.tlx.ops.kernels.addmm.gfx950_torch:addmm",
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        op="bmm_torchtlx",
        arch="gfx950",
        variant="inductor_gfx950_bmm",
        impl="triton.tlx.ops.kernels.bmm.gfx950_torch:bmm",
        dtypes=_FP16,
        requires=frozenset(),
    ),
    OpSpec(
        op="flash_attn",
        arch="sm90",
        variant="ws_pipelined_pingpong",
        impl="triton.tlx.ops.kernels.flash_attn.sm90:flash_attn",
        invariant_vocab=_ATTN_DIMS,
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") in (64, 128),
        requires=frozenset({"tma"}),
        supports_backward=True,
    ),
    OpSpec(
        op="flash_attn",
        arch="sm100",
        variant="ws_pipelined_persistent",
        impl="triton.tlx.ops.kernels.flash_attn.sm100:flash_attn",
        invariant_vocab=_ATTN_DIMS,
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") in (64, 128),
        requires=frozenset({"tma", "tmem"}),
        supports_backward=True,
    ),
    OpSpec(
        op="flash_attn",
        arch="gfx950",
        variant="rotated_cluster_pipeline",
        impl="triton.tlx.ops.kernels.flash_attn.gfx950:flash_attn",
        invariant_vocab=_ATTN_DIMS,
        dtypes=_BF16,
        accepts=lambda d: d.get("HEAD_DIM") in (64, 128),
        requires=frozenset(),
        supports_backward=True,
    ),
    OpSpec(
        op="prepare_flash_attn_varlen_backward",
        arch="gfx950",
        variant="host_validated",
        impl="triton.tlx.ops.kernels.flash_attn_varlen.gfx950_bwd:prepare_varlen_backward",
        invariant_vocab=_VARLEN_DIMS,
        dtypes=frozenset({"int32"}),
    ),
    OpSpec(
        op="flash_attn_varlen_backward",
        arch="gfx950",
        variant="prepared",
        impl="triton.tlx.ops.kernels.flash_attn_varlen.gfx950_bwd:fa_varlen_backward",
        invariant_vocab=_VARLEN_DIMS,
        dtypes=_BF16,
        accepts=lambda d: d.get("HEAD_DIM") == 128,
        # Explicit first-order gradients; differentiating this op is unsupported.
        supports_backward=False,
    ),
    OpSpec(
        op="flash_attn_mxfp8",
        arch="sm100",
        variant="ws_pipelined_persistent_mxfp8",
        impl="triton.tlx.ops.kernels.flash_attn_mxfp8.sm100:flash_attn_mxfp8",
        invariant_vocab=_ATTN_DIMS,
        dtypes=_BF16,
        accepts=lambda d: d.get("HEAD_DIM") == 128 and d.get("N_CTX", 0) % 256 == 0,
        requires=frozenset({"tma", "tmem"}),
        supports_backward=True,
    ),
    OpSpec(
        op="flash_attn_mxfp8",
        arch="gfx950",
        variant="mfma_mxfp8",
        impl="triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950:flash_attn_mxfp8",
        invariant_vocab=_ATTN_DIMS,
        dtypes=_BF16,
        # Direct-LDS buffer loads use signed i32 offsets within each head.
        accepts=lambda d:
        (d.get("HEAD_DIM") == 128 and d.get("N_CTX", 0) > 0 and d["N_CTX"] % 256 == 0 and d["N_CTX"] * 128 < 2**31),
        requires=frozenset(),
        supports_backward=True,
    ),
    OpSpec(
        op="flash_attn_mxfp8",
        arch="gfx1250",
        variant="wmma_tdm_mxfp8",
        impl="triton.tlx.ops.kernels.flash_attn_mxfp8.gfx1250:flash_attn_mxfp8",
        invariant_vocab=_ATTN_DIMS,
        dtypes=_BF16,
        accepts=lambda d: d.get("HEAD_DIM") == 128 and d.get("N_CTX", 0) % 256 == 0,
        requires=frozenset(),
        supports_backward=False,
    ),
    OpSpec(
        op="flash_attn_varlen",
        arch="gfx950",
        variant="mfma_interleaved",
        impl="triton.tlx.ops.kernels.flash_attn_varlen.gfx950:flash_attn_varlen",
        invariant_vocab=_VARLEN_DIMS,
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") == 128,
        requires=frozenset(),
        supports_backward=False,
    ),
    OpSpec(
        op="flash_attn_varlen",
        arch="sm90",
        variant="ws_pingpong",
        impl="triton.tlx.ops.kernels.flash_attn_varlen.sm90:flash_attn_varlen",
        invariant_vocab=_VARLEN_DIMS,
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") in (64, 128),
        requires=frozenset({"tma"}),
        supports_backward=False,
    ),
    OpSpec(
        op="mla_window_attn",
        arch="sm90",
        variant="overlapped_row_blocks",
        impl="triton.tlx.ops.kernels.mla_window_attn.sm90:mla_window_attn",
        invariant_vocab=_ROW_DIMS,
        dtypes=frozenset({"float16"}),
        # The warp-specialized kernel reads the projection weights by TMA: their data must start on a 16-byte boundary.
        accepts=lambda d: d.get("D") in (128, 256) and d.get("HEAD_DIM") in
        (64, 128) and all(ptr % 16 == 0 for ptr in d.get("base_ptrs", ())),
        supports_backward=False,
    ),
    OpSpec(
        op="linear_rowscale",
        arch="sm90",
        variant="tlx_ws_pingpong_x_stationary",
        impl="triton.tlx.ops.kernels.fused.linear_rowscale.sm90:linear_rowscale",
        invariant_vocab=_GEMM_DIMS,
        dtypes=_FP16,
        accepts=_linear_rowscale_accepts,
        requires=frozenset({"tma"}),
        supports_backward=False,
    ),
    OpSpec(
        op="linear_rowscale_launch",
        arch="sm90",
        variant="tlx_ws_pingpong_x_stationary",
        impl="triton.tlx.ops.kernels.fused.linear_rowscale.sm90:linear_rowscale_launch",
        dtypes=_FP16,
        accepts=_linear_rowscale_accepts,
        requires=frozenset({"tma"}),
        supports_backward=False,
    ),
    OpSpec(
        op="swiglu_ffn",
        arch="sm90",
        variant="tlx_ws_persistent",
        impl="triton.tlx.ops.kernels.swiglu_ffn.sm90:swiglu_ffn",
        invariant_vocab=_ROW_DIMS,
        dtypes=frozenset({"float16"}),
        # TMA reads the weights: their data must start on a 16-byte boundary.
        accepts=lambda d: d.get("D") in (128, 256) and all(ptr % 16 == 0 for ptr in d.get("base_ptrs", ())),
        requires=frozenset({"tma"}),
        supports_backward=False,
    ),
    OpSpec(
        op="rmsnorm_cat",
        arch="sm90",
        variant="row_in_registers",
        impl="triton.tlx.ops.kernels.norm._cat_sm90:rmsnorm_cat",
        dtypes=_FP16,
        # The register kernel takes any strides; the TLX kernel for the over-arch layout loads rows with cp.async.bulk
        # and TMA, and runs only when `tlx_layout` proves 16-byte alignment.
        requires=frozenset({"tma"}),
        supports_backward=False,
    ),
    OpSpec(
        op="hstu_attn_dev",
        arch="sm100",
        variant="ws",
        impl="triton.tlx.ops.kernels.hstu_attn.sm100:hstu_attn",
        invariant_vocab=_HSTU_DIMS,
        dtypes=_FP16,
        # Causal-only, non-causal is not supported yet
        accepts=lambda d: bool(d.get("causal", True)),
        requires=frozenset({"tma", "tmem"}),
        supports_backward=True,
    ),
    OpSpec(
        op="hstu_attn_dev",
        arch="gfx950",
        variant="tlx",
        impl="triton.tlx.ops.kernels.hstu_attn.gfx950:hstu_attn",
        invariant_vocab=_HSTU_DIMS,
        dtypes=_FP16,
        # Causal-only, non-causal is not supported yet.
        accepts=lambda d: bool(d.get("causal", True)),
        supports_backward=True,
    ),
    OpSpec(
        op="hstu_cross_attn",
        arch="gfx950",
        variant="tlx",
        impl="triton.tlx.ops.kernels.hstu_cross_attn.gfx950:hstu_cross_attn",
        invariant_vocab=_HSTU_DIMS,
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") in (16, 32, 64, 128, 256),
        supports_backward=True,
    ),
    OpSpec(
        op="kimi_delta_attention",
        arch="sm100",
        variant="ws",
        impl="triton.tlx.ops.kernels.kda.sm100:kimi_delta_attention",
        invariant_vocab=_KDA_DIMS,
        dtypes=_FP16,
        accepts=lambda d: d.get("HEAD_DIM") == 128,
        requires=frozenset({"tma", "tmem"}),
        supports_backward=True,
    ),
    OpSpec(
        op="kda_paged_prefill",
        arch="gfx950",
        variant="tlx",
        impl="triton.tlx.ops.kernels.kda.gfx950_prefill:kda_paged_prefill",
        invariant_vocab=_KDA_DIMS,
        dtypes=_BF16,
        accepts=lambda d: d.get("KEY_DIM") == 128 and d.get("VALUE_DIM") == 128,
    ),
    OpSpec(
        op="kda_recurrent_decode",
        arch="gfx950",
        variant="tlx",
        impl="triton.tlx.ops.kernels.kda.gfx950_decode:kda_recurrent_decode",
        invariant_vocab=_KDA_DIMS,
        dtypes=_BF16,
        accepts=lambda d: 1 <= d.get("KEY_DIM", 0) <= 128 and 1 <= d.get("VALUE_DIM", 0) <= 128,
    ),
)

CATALOG += tuple(
    OpSpec(op=op, arch=arch, variant="row_norm", impl=f"triton.tlx.ops.kernels.norm.{module}:{op}", dtypes=frozenset(
        {"float16", "bfloat16", "float32"}), supports_backward=op in _NORM_AUTOGRAD)
    for module, names in _NORM_MODULES.items()
    for op in names
    for arch in (("sm90", "sm100") if module in ("sm100", "_softmax") else ("sm100", )))

_BY_KEY = {(s.op, s.arch): s for s in CATALOG}
assert len(_BY_KEY) == len(CATALOG), "duplicate (op, arch) in CATALOG"


def _target(device=None):
    # Lazy: hw.target imports torch.
    from triton.language.extra.tlx.hw.target import current_target, target_for_device

    return current_target() if device is None else target_for_device(device)


def _capabilities(target) -> frozenset:
    if target.spec is None:
        return frozenset()
    caps = set()
    if target.is_cuda and target.capability is not None and target.capability[0] >= 9:
        caps |= {"tma", "cluster"}
    if getattr(target, "has_tmem", False):
        caps.add("tmem")
    return frozenset(caps)


@functools.lru_cache(maxsize=None)
def _load(impl: str) -> Callable[..., Any]:
    mod, _, attr = impl.partition(":")
    return getattr(importlib.import_module(mod), attr)


@functools.lru_cache(maxsize=None)
def _arches_for(op: str) -> tuple[str, ...]:
    return tuple(sorted(s.arch for s in CATALOG if s.op == op))


def has_impl(op: str, arch: str) -> bool:
    """Is there a catalog entry for this pair, without importing the kernel?

    A table lookup, not a capability check: the benchmark suite uses it to skip
    an op cleanly on an arch it was never written for, rather than running every
    shape and reporting each one as an error.
    """
    return (op, arch) in _BY_KEY


def invariant_vocab_for(op: str) -> frozenset:
    """The dim vocabulary for `op`, shared by its arch entries; empty if unknown."""
    for spec in CATALOG:
        if spec.op == op:
            return spec.invariant_vocab
    return frozenset()


@functools.lru_cache(maxsize=None)
def any_enforces_invariance(op: str) -> bool:
    """Whether any arch entry for `op` enforces an invariance request yet."""
    return any(spec.op == op and spec.enforces_invariance for spec in CATALOG)


@functools.lru_cache(maxsize=None)
def _impl_for_arch(op: str, arch: str) -> tuple[Callable[..., Any], OpSpec]:
    spec = _BY_KEY.get((op, arch))
    if spec is None:
        available = ", ".join(_arches_for(op)) or "(nothing yet)"
        raise UnsupportedOp(f"tlx.ops.{op} has no implementation for arch={arch!r}. Available on: {available}")
    return _load(spec.impl), spec


def impl_for(op: str, arch: Optional[str] = None, *, device=None) -> tuple[Callable[..., Any], OpSpec]:
    """The blessed callable for `op`, plus its spec.

    Raises rather than falling back: a silent fallback turns "TLX is not
    running here" into an unexplained performance cliff.

    Public op wrappers pass the input tensor's `device`, so dispatch follows
    the device that will execute the kernel rather than an ambient current
    device. An explicit `arch` is retained for private catalog tests; it pins
    the entry and skips the capability check.
    """
    if arch is not None and device is not None:
        raise ValueError("impl_for accepts either arch or device, not both")
    if arch is None:
        target = _target(device)
        if not target.key:
            available = ", ".join(_arches_for(op)) or "(nothing yet)"
            location = f" for device={device}" if device is not None else ""
            raise UnsupportedOp(f"tlx.ops.{op}: could not determine a GPU architecture{location}. "
                                f"Available on: {available}")
        spec = _BY_KEY.get((op, target.key))
        if spec is None:
            available = ", ".join(_arches_for(op)) or "(nothing yet)"
            raise UnsupportedOp(f"tlx.ops.{op} has no implementation for {target.key}. "
                                f"Available on: {available}")
        missing = spec.requires - _capabilities(target)
        if missing:
            raise UnsupportedOp(f"{spec} needs {sorted(missing)}, which {target.key} does not report")
        return _load(spec.impl), spec

    return _impl_for_arch(op, arch)


def check_inputs(spec: OpSpec, dtype=None, **dims) -> None:
    if dtype is not None and spec.dtypes:
        name = str(dtype).removeprefix("torch.")
        if name not in spec.dtypes:
            raise InvalidInput(f"{spec} does not support {name}; supported: {sorted(spec.dtypes)}")
    if spec.accepts is not None and not spec.accepts(dims):
        raise InvalidInput(f"{spec} does not support these inputs: {dims}")


def check_backward(spec: OpSpec, *inputs) -> None:
    """Fail before launch when autograd is requested but unavailable."""
    if spec.supports_backward or not any(getattr(tensor, "requires_grad", False) for tensor in inputs):
        return

    import torch

    if torch.is_grad_enabled():
        raise UnsupportedBackward(f"tlx.ops.{spec.op} does not support backward on {spec.arch}; "
                                  "use tensors with requires_grad=False or call it under torch.no_grad()")
