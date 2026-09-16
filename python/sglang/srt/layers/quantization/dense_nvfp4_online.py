"""Load-time NVFP4 requantization for the DSV4.1-Flash *dense* Linear layers.

Why this exists instead of a serialized ``modelopt_fp4`` checkpoint: the fp4 MoE
path on this model is owned by ``Fp8Config`` itself (see ``fp8.py``, the
``is_fp4_experts`` branch that returns ``Mxfp4FlashinferCutlassMoEMethod``).
Installing an ``hf_quant_config.json`` would swap the whole quant config over to
``ModelOptMixedPrecisionConfig``, whose ``get_quant_method`` hands ``FusedMoE`` a
ModelOpt MoE method that cannot read this checkpoint's native I8-packed + E8M0
expert layout. So the experts must stay on ``Fp8Config`` and only the dense
LinearBase branch is redirected here.

The dense weights ship as 32x32 ue8m0 blockwise FP8. They are loaded exactly as
before, then dequantized to BF16 and requantized to NVFP4 once, at the end of
weight loading. Serving reuses ``ModelOptFp4LinearMethod`` unchanged for the
CUTLASS layout prep and the GEMM.

``wo_a`` is deliberately NOT eligible: its absorb GEMM reads ``.weight`` /
``.weight_scale_inv`` directly and never calls ``LinearMethod.apply``.

MERGED LINEARS
--------------
``shared_experts.gate_up_proj`` (logical_widths ``[576, 576]``) and
``self_attn.wqkv_a`` (wq_a + wkv fused, see ``deepseek_v4.py`` ``wqkv_a``) pack
several logical output slices into one tensor. Two things that must hold for
them, both of which the original narrow whitelist never exercised:

1. ``WEIGHT_LOADER_V2_SUPPORTED`` (``linear.py``) is keyed on the quant method's
   CLASS NAME. ``Fp8LinearMethod`` is in it, this subclass was not, so every
   parameter this method created was handed the LEGACY
   ``MergedColumnParallelLinear.weight_loader`` instead of ``weight_loader_v2``.
   The legacy path computes a merged shard offset in ROWS
   (``sum(output_sizes[:id]) // tp_size`` = 576) and applies it to the blockwise
   scale parameter, which only has ``N/block_n`` = 36 rows -- hence
   ``IndexError: start out of range (expected to be in range of [-36,36], but
   got 576)``. Only ``weight_loader_v2`` divides the offset by ``block_n``
   (``linear.py`` ``MergedColumnParallelLinear.weight_loader_v2``, the
   ``BlockQuantScaleParameter`` branch). Non-merged linears (``wq_b``/``wo_b``)
   never hit this because their single-slice offsets are trivially 0, which is
   why the shipped whitelist worked. ``_register_weight_loader_v2`` below fixes
   the dispatch rather than the symptom.

2. ``ModelOptFp4LinearMethod.process_weights_after_loading`` reduces
   ``weight_scale_2`` with ``.max()`` and folds it into ONE scalar ``alpha`` for
   the whole GEMM. So all logical slices of a merged linear MUST share a single
   ``weight_scale_2``; a per-slice scale_2 would be silently wrong for every
   slice but the largest. We therefore quantize against one common scale taken
   from the global amax, and ``_audit_slices`` verifies per logical slice that
   this common scale did not cost that slice any accuracy (the failure mode to
   look for is a small-range slice whose e4m3 block scales underflow).
"""

from __future__ import annotations

import logging
import os
from typing import List

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

logger = logging.getLogger(__name__)

# Which dense Linears to convert. Matching is on the MODULE prefix, which does
# not follow the checkpoint tensor names: the modules are
# ``model.layers.N.self_attn.*`` and ``model.layers.N.mlp.shared_experts.*``
# (the checkpoint calls them ``attn.*`` and ``ffn.*``), and wq_a/wkv do not exist
# as separate modules at all -- they are fused into ``self_attn.wqkv_a``. So this
# is an INCLUDE-by-family / EXCLUDE-by-name rule rather than a list of leaves.
#
# The first two are the originally shipped set. ``wqkv_a`` and the shared-expert
# pair are the merged/extra linears that item (1) in the module docstring
# unblocks.
_SHIPPED_FAMILIES = ("self_attn.wq_b", "self_attn.wo_b")
_INCLUDE_FAMILIES = (
    "self_attn.wq_b",
    "self_attn.wo_b",
    "self_attn.wqkv_a",
    "mlp.shared_experts.gate_up_proj",
    "mlp.shared_experts.down_proj",
)
# wo_a never goes through LinearMethod.apply: its absorb GEMM reads .weight /
# .weight_scale_inv directly and runs its own batched kernel.
_EXCLUDE_SUFFIXES = ("wo_a",)

_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_FP8_FP4_MAX = _E4M3_MAX * 6.0
# Smallest magnitude an e4m3 block scale can hold before it flushes to zero
# (subnormal min = 2^-9). A logical slice whose block scales land at or under
# this against the shared weight_scale_2 would be quantized to all-zero.
_E4M3_MIN_SUBNORMAL = 2.0**-9

_CONVERTED: List[str] = []
_AUDIT: List[str] = []


def dense_nvfp4_enabled() -> bool:
    return os.environ.get("SGLANG_DENSE_NVFP4", "0") == "1"


def _audit_enabled() -> bool:
    return os.environ.get("SGLANG_DENSE_NVFP4_AUDIT", "0") == "1"


def _families():
    raw = os.environ.get("SGLANG_DENSE_NVFP4_LAYERS", "")
    if raw == "shipped":
        return _SHIPPED_FAMILIES
    return tuple(s for s in raw.split(",") if s) if raw else _INCLUDE_FAMILIES


def _register_weight_loader_v2() -> None:
    """Opt this method into ``weight_loader_v2``.

    ``linear.py`` selects the loader by quant-method CLASS NAME, so a subclass of
    ``Fp8LinearMethod`` silently drops to the legacy loader. The legacy loader
    does not divide merged shard offsets by ``block_n``, which corrupts (in fact
    crashes) blockwise-FP8 scale loading on every MERGED linear. Registering here
    -- at import, which happens inside the first ``get_quant_method`` call and so
    strictly before any ``create_weights`` -- restores the same loader the
    unpatched ``Fp8LinearMethod`` would have used.
    """
    from sglang.srt.layers.linear import WEIGHT_LOADER_V2_SUPPORTED

    if "DenseNvfp4LinearMethod" not in WEIGHT_LOADER_V2_SUPPORTED:
        WEIGHT_LOADER_V2_SUPPORTED.append("DenseNvfp4LinearMethod")


_register_weight_loader_v2()


def dense_nvfp4_eligible(prefix: str) -> bool:
    """True when this dense Linear should be served as NVFP4."""
    if not dense_nvfp4_enabled():
        return False
    # Routed experts are already fp4 and belong to the MoE method, not here.
    if ".experts." in prefix and "shared_experts." not in prefix:
        return False
    if any(prefix.endswith(x) for x in _EXCLUDE_SUFFIXES):
        return False
    skip = os.environ.get("SGLANG_DENSE_NVFP4_SKIP", "")
    if skip and any(s and s in prefix for s in skip.split(",")):
        return False
    return any(f in prefix for f in _families())


def _dequantize_blockwise_fp8(weight, scale, block) -> torch.Tensor:
    """[N,K] fp8 + [ceil(N/bn), ceil(K/bk)] block scales -> BF16."""
    bn, bk = block
    w = weight.cuda().to(torch.float32)
    s = scale.cuda()
    if s.dtype == torch.uint8:
        # ue8m0 exponent byte: value = 2 ** (byte - 127)
        s = torch.pow(torch.tensor(2.0, device=s.device), s.float() - 127.0)
    else:
        s = s.float()
    n, k = w.shape
    sn, sk = s.shape
    assert sn == (n + bn - 1) // bn and sk == (k + bk - 1) // bk, (
        f"block scale shape {tuple(s.shape)} does not match weight {tuple(w.shape)} "
        f"with block {block}"
    )
    s = s.repeat_interleave(bn, 0)[:n].repeat_interleave(bk, 1)[:, :k]
    return (w * s).to(torch.bfloat16)


def _quantize_to_nvfp4(w_bf16: torch.Tensor, weight_scale_2: torch.Tensor):
    """BF16 [N,K] -> (packed uint8 [N,K/2], e4m3 block scales [N,K/16]).

    ``weight_scale_2`` is supplied by the caller rather than derived here so that
    every logical slice of a MERGED linear is quantized against the one common
    scale that ``ModelOptFp4LinearMethod`` will later fold into ``alpha``.
    """
    from flashinfer import SfLayout, nvfp4_quantize

    n, k = w_bf16.shape
    q, sf = nvfp4_quantize(
        w_bf16.contiguous(),
        1.0 / weight_scale_2,
        sfLayout=SfLayout.layout_linear,
        backend="cute-dsl",
    )
    q = q.reshape(n, k // 2).view(torch.uint8)
    sf = sf.view(torch.float8_e4m3fn).reshape(n, k // 16).contiguous()
    return q, sf


# e2m1 decode table, indexed by the 4-bit code (sign bit is bit 3).
_FP4_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize_nvfp4(q: torch.Tensor, sf: torch.Tensor, ws2: torch.Tensor):
    """Decode the ACTUAL stored NVFP4 bytes back to fp32.

    Deliberately reads ``q``/``sf`` rather than re-deriving what the quantizer
    "should" have produced, so the audit measures the bytes that will really be
    served.
    """
    n, k_half = q.shape
    table = torch.tensor(_FP4_E2M1, device=q.device, dtype=torch.float32)
    lo = (q & 0x0F).to(torch.long)
    hi = (q >> 4).to(torch.long)

    def decode(idx):
        mag = table[idx & 0x7]
        return torch.where((idx & 0x8) != 0, -mag, mag)

    vals = torch.stack((decode(lo), decode(hi)), dim=-1).reshape(n, k_half * 2)
    scales = sf.to(torch.float32).repeat_interleave(16, dim=1)
    return vals * scales * ws2.to(torch.float32)


def _audit_slices(prefix, w_bf16, q, sf, ws2, logical_widths):
    """Per logical slice: relative error of the round-trip, and scale headroom.

    A merged linear shares one ``weight_scale_2``. If one slice has a much
    smaller dynamic range than its sibling, its e4m3 block scales shrink toward
    the subnormal floor and that slice -- and only that slice -- loses accuracy,
    which an aggregate error over the fused tensor would hide.
    """
    deq = _dequantize_nvfp4(q, sf, ws2)
    ref = w_bf16.float()
    rows = 0
    for i, width in enumerate(logical_widths):
        r = ref[rows : rows + width]
        d = deq[rows : rows + width]
        s = sf[rows : rows + width].to(torch.float32)
        num = (d - r).norm().item()
        den = r.norm().item()
        rel = num / den if den > 0 else 0.0
        nz = s[s > 0]
        line = (
            f"{prefix} slice{i} rows[{rows}:{rows + width}] "
            f"rel_err={rel:.4%} amax={r.abs().amax().item():.6g} "
            f"sf_min={(nz.min().item() if nz.numel() else 0.0):.6g} "
            f"sf_max={s.max().item():.6g} "
            f"sf_zero_frac={(s == 0).float().mean().item():.3%}"
        )
        _AUDIT.append(line)
        logger.info("dense-nvfp4-audit: %s", line)
        if (s == 0).any():
            logger.error(
                "dense-nvfp4-audit: %s slice%d HAS ZERO BLOCK SCALES -- this slice "
                "is destroyed by the shared weight_scale_2", prefix, i,
            )
        rows += width


class DenseNvfp4LinearMethod(Fp8LinearMethod):
    """FP8 checkpoint loading + load-time NVFP4 requantization + NVFP4 serving."""

    def __init__(self, quant_config, prefix: str = ""):
        super().__init__(quant_config)
        self.prefix = prefix
        self._fp4_method = None

    def _make_fp4_method(self):
        if self._fp4_method is None:
            from sglang.srt.layers.quantization.modelopt_quant import (
                ModelOptFp4Config,
                ModelOptFp4LinearMethod,
            )

            cfg = ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                group_size=16,
                exclude_modules=[],
            )
            self._fp4_method = ModelOptFp4LinearMethod(cfg)
        return self._fp4_method

    # create_weights is inherited from Fp8LinearMethod unchanged, so the
    # checkpoint loads byte-for-byte as it does today.

    def process_weights_after_loading(self, layer) -> None:
        block = self.weight_block_size or [32, 32]
        w_fp8 = layer.weight.data
        scale = getattr(layer, "weight_scale_inv", None)
        if scale is None or w_fp8.dtype != torch.float8_e4m3fn or w_fp8.ndim != 2:
            # Not a blockwise-FP8 2D weight (bf16 gate/router, odd shapes): leave
            # it exactly as it was. The broad family rule can match these.
            logger.info(
                "dense-nvfp4: %s is not blockwise-FP8, leaving on the FP8 path",
                self.prefix,
            )
            return super().process_weights_after_loading(layer)
        n, k = w_fp8.shape
        if k % 64 != 0 or n % 32 != 0:
            logger.warning(
                "dense-nvfp4: %s shape (%d,%d) not NVFP4-tileable, leaving on FP8",
                self.prefix, n, k,
            )
            return super().process_weights_after_loading(layer)

        # The logical output slices of a merged linear, as create_weights
        # recorded them (fp8.py: layer.logical_widths = output_partition_sizes).
        # This must be PRESERVED, not flattened to [n]: it is how the rest of the
        # stack knows where gate ends and up begins.
        logical_widths = list(getattr(layer, "logical_widths", None) or [n])
        if sum(logical_widths) != n:
            logger.warning(
                "dense-nvfp4: %s logical_widths %s do not sum to %d; treating as "
                "a single slice", self.prefix, logical_widths, n,
            )
            logical_widths = [n]
        # Every slice of a merged linear must share ONE weight_scale_2, because
        # ModelOptFp4LinearMethod reduces weight_scale_2 with .max() into a single
        # scalar alpha. Taking the global amax is exactly that shared scale.
        if k % 16 != 0:
            logger.warning(
                "dense-nvfp4: %s K=%d not a multiple of 16, leaving on FP8",
                self.prefix, k,
            )
            return super().process_weights_after_loading(layer)

        w_bf16 = _dequantize_blockwise_fp8(w_fp8, scale.data, block)
        amax = w_bf16.abs().float().nan_to_num().amax()
        ws2 = torch.where(
            amax > 0, amax / _FP8_FP4_MAX, torch.ones_like(amax)
        ).to(torch.float32)
        q, sf = _quantize_to_nvfp4(w_bf16, ws2)

        if _audit_enabled() or len(logical_widths) > 1:
            # Merged linears are audited unconditionally: the shared-scale risk
            # only exists for them, and there are few enough to be free.
            _audit_slices(self.prefix, w_bf16, q, sf, ws2, logical_widths)
        del w_bf16

        # Drop the FP8 parameters and install the ModelOpt NVFP4 ones.
        for name in ("weight_scale_inv", "input_scale"):
            if hasattr(layer, name):
                try:
                    delattr(layer, name)
                except Exception:
                    setattr(layer, name, None)
        layer.weight = Parameter(q, requires_grad=False)
        layer.weight_scale = Parameter(sf, requires_grad=False)
        layer.weight_scale_2 = Parameter(ws2.reshape(1).cuda(), requires_grad=False)
        # Static per-tensor activation scale of 1.0: the NVFP4 per-16 block scale
        # is e4m3, whose range comfortably covers these activations, so the global
        # activation scale is not the accuracy-limiting term here.
        layer.input_scale = Parameter(
            torch.ones(1, dtype=torch.float32, device=q.device), requires_grad=False
        )
        layer.logical_widths = logical_widths
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        layer.params_dtype = torch.bfloat16

        # Reuse ModelOpt's CUTLASS pad+swizzle prep verbatim.
        self._make_fp4_method().process_weights_after_loading(layer)
        _CONVERTED.append(f"{self.prefix}:{n}x{k}")
        if len(_CONVERTED) <= 3 or len(_CONVERTED) % 100 == 0:
            logger.info(
                "dense-nvfp4: converted %s [%d,%d] (total %d)",
                self.prefix, n, k, len(_CONVERTED),
            )

    def apply(self, layer, x, bias=None):
        if not hasattr(layer, "weight_scale_interleaved"):
            # Fell back to FP8 for this layer (non-tileable shape).
            return super().apply(layer, x, bias)
        return self._make_fp4_method().apply(layer, x, bias)


def converted_layers() -> List[str]:
    return list(_CONVERTED)


def audit_lines() -> List[str]:
    return list(_AUDIT)
