"""Per-forward-mode FP8 -> MXFP4 swap for DeepSeek-V4 attention projections.

The DSv4 "fp4" checkpoint is fp4 only for the routed experts; every attention
projection is fp8 e4m3 with 128x128 ue8m0 block scales and runs through
``Fp8LinearMethod`` -> ``aiter_w8a8_block_fp8_linear`` (the aiter CK bpreshuffle
GEMM). This module requantizes a selected subset of those weights to OCP MXFP4
(e2m1 data, per-32 e8m0 scales) at load time and reroutes those layers onto
aiter's ``gemm_a4w4``.

The selection is made **per forward mode**, because the fp4 GEMM only wins at
some shapes. Measured on MI355X (gfx950, aiter CK/ASM a4w4 vs the fp8 CK
bpreshuffle GEMM, CUDA-graph timed, DP8/TP8 dp-attention per-rank shapes):

    projection                     N       K   M=8 (decode)  M=8192 (prefill)
    attn.q_b_proj   (wq_b)     65536    1536       x1.21           x1.63
    attn.q_a_proj   (wqkv_a)    2048    7168       x0.77           x2.27
    indexer.wq_b                8192    1536       x0.60           x2.45
    attn.o_proj_b   (wo_b)      7168   16384       x1.45           x1.95

so ``wqkv_a`` and ``indexer_wq_b`` are worth swapping for prefill but are a
regression for decode. A projection selected for only one mode keeps **both**
weights resident and dispatches on the mode; a projection selected for both
modes keeps only the fp4 weight and frees the fp8 one.

Flags
-----
``SGLANG_DSV4_FP4_PROJ``          master enable (0/1, default 0)
``SGLANG_DSV4_FP4_PROJ_PREFILL``  csv of keys used on prefill-like forwards
                                  (default ``wq_b,wqkv_a,indexer_wq_b,wo_b``)
``SGLANG_DSV4_FP4_PROJ_DECODE``   csv of keys used on decode-like forwards
                                  (default ``wq_b,wo_b``)

Each list also accepts ``all`` and ``none``/empty.

``wo_a`` is deliberately not selectable: with SGLANG_OPT_FP8_WO_A_GEMM (default
on) it never reaches ``quant_method.apply`` -- the absorb GEMM in
MqaAttention.forward reads ``.weight``/``.weight_scale_inv`` and runs its own
batched mxscale BMM -- and aiter has no batched a4w4 to replace it with.

Accuracy (GSM8k strict-match, n=1319, DeepSeek-V4-Pro FP4, MTP depth 3, real
target verification): baseline 96.82 +- 0.48; each projection swapped on its
own lands within 0.4 pt of that, all four together at 96.59 +- 0.50 -- inside
the +-0.70 pt stderr of the difference.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, FrozenSet, List

import torch

logger = logging.getLogger(__name__)

# Selector key -> predicate on the layer prefix, e.g. "model.layers.3.attn.wq_b"
# or "model.layers.7.attn.indexer.wq_b".
SELECTORS = {
    # attn.q_b_proj: [65536, 1536], every layer plus the MTP head.
    "wq_b": lambda p: p.endswith(".attn.wq_b"),
    # attn.q_a_proj: [2048, 7168] when SGLANG_OPT_FUSE_WQA_WKV fuses wq_a+wkv,
    # otherwise the separate [1536, 7168] / [512, 7168] projections.
    "wqkv_a": lambda p: (
        p.endswith(".attn.wqkv_a")
        or p.endswith(".attn.wq_a")
        or p.endswith(".attn.wkv")
    ),
    # C4 indexer query projection: [8192, 1536], C4 layers only.
    "indexer_wq_b": lambda p: p.endswith(".indexer.wq_b"),
    # attn.o_proj_b: [7168, 16384], every layer.
    "wo_b": lambda p: p.endswith(".attn.wo_b"),
}

PREFILL = "prefill"
DECODE = "decode"

_DEFAULT_PREFILL = "wq_b,wqkv_a,indexer_wq_b,wo_b"
_DEFAULT_DECODE = "wq_b,wo_b"


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def _parse_keys(env_name: str, default: str) -> FrozenSet[str]:
    raw = os.environ.get(env_name, default).strip()
    if not raw or raw.lower() in ("none", "0", "off"):
        return frozenset()
    if raw.lower() == "all":
        return frozenset(SELECTORS)
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SELECTORS]
    if unknown:
        raise ValueError(
            f"{env_name}: unknown selector(s) {unknown}; "
            f"valid keys are {sorted(SELECTORS)} (or 'all'/'none')"
        )
    return frozenset(keys)


def is_enabled() -> bool:
    return os.environ.get("SGLANG_DSV4_FP4_PROJ", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def selected_masks() -> Dict[str, FrozenSet[str]]:
    """key -> the set of forward modes it should run in fp4 for."""
    if not is_enabled():
        return {}
    prefill = _parse_keys("SGLANG_DSV4_FP4_PROJ_PREFILL", _DEFAULT_PREFILL)
    decode = _parse_keys("SGLANG_DSV4_FP4_PROJ_DECODE", _DEFAULT_DECODE)
    masks = {}
    for key in SELECTORS:
        modes = set()
        if key in prefill:
            modes.add(PREFILL)
        if key in decode:
            modes.add(DECODE)
        if modes:
            masks[key] = frozenset(modes)
    return masks


def _match(prefix: str, masks: Dict[str, FrozenSet[str]]):
    for key, modes in masks.items():
        if SELECTORS[key](prefix):
            return key, modes
    return None, None


# --------------------------------------------------------------------------
# forward mode, published by the model's forward
# --------------------------------------------------------------------------

# Modes whose token count is decode-shaped. TARGET_VERIFY is the MTP
# verification step (bs * num_draft_tokens rows) and shares the decode CUDA
# graph; IDLE is a zero-token DP filler that is captured the same way.
# ForwardMode.is_extend() cannot be used here -- it returns True for
# TARGET_VERIFY.
_DECODE_MODE_NAMES = ("DECODE", "TARGET_VERIFY", "IDLE")

# Default to prefill: it is the only value that is always serviceable, because
# a projection selected for prefill only keeps its fp8 weight too, whereas a
# projection selected for both modes has had its fp8 weight freed.
_CURRENT_MODE = PREFILL


def current_mode() -> str:
    return _CURRENT_MODE


def _classify(forward_batch) -> str:
    mode = getattr(forward_batch, "forward_mode", None)
    return DECODE if getattr(mode, "name", "") in _DECODE_MODE_NAMES else PREFILL


def _wrap_model_forward(cls) -> None:
    """Publish the forward mode around ``cls.forward``.

    This has to sit on the model class rather than on ``ModelRunner.forward``:
    CUDA-graph capture calls ``model_runner.model.forward(...)`` directly, and
    the fp8/fp4 branch is a Python-level ``if`` that gets baked into the
    captured graph. Wrapping here means capture sees the mode it is capturing
    for. The branch is a pure function of ``forward_mode``, so every graph
    captured for a given mode bakes the same kernels.
    """
    orig = cls.forward
    if getattr(orig, "_dsv4_fp4_wrapped", False):
        return

    def forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        global _CURRENT_MODE
        prev = _CURRENT_MODE
        _CURRENT_MODE = _classify(forward_batch)
        try:
            return orig(self, input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            _CURRENT_MODE = prev

    forward._dsv4_fp4_wrapped = True
    forward.__name__ = getattr(orig, "__name__", "forward")
    forward.__doc__ = getattr(orig, "__doc__", None)
    cls.forward = forward


def install_model_hook(cls) -> None:
    """Called from the DSv4 model modules; no-op unless the swap is enabled."""
    if not is_enabled():
        return
    _wrap_model_forward(cls)


# --------------------------------------------------------------------------
# MXFP4 linear method
# --------------------------------------------------------------------------

_MXQ = None


def _mx_quant():
    global _MXQ
    if _MXQ is None:
        import aiter

        _MXQ = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    return _MXQ


class Fp4ProjLinearMethod:
    """Mode-dispatching wrapper around the layer's original Fp8LinearMethod.

    On a forward mode the layer was selected for, runs aiter's a4w4 GEMM over
    the MXFP4 copy of the weight; otherwise delegates to the untouched fp8
    method, so the non-selected mode is bit-identical to the baseline build.
    """

    def __init__(self, fp8_method, key: str, modes: FrozenSet[str]):
        self.fp8_method = fp8_method
        self.key = key
        self.modes = modes
        # Some callers introspect the wrapped method's config.
        self.quant_config = getattr(fp8_method, "quant_config", None)

    def __getattr__(self, name):
        # Anything we do not override (block_quant, weight_block_size, ...)
        # comes from the real fp8 method.
        try:
            fp8_method = self.__dict__["fp8_method"]
        except KeyError:  # pragma: no cover - only during partial construction
            raise AttributeError(name) from None
        return getattr(fp8_method, name)

    def create_weights(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "Fp4ProjLinearMethod is installed after the weights are loaded"
        )

    def process_weights_after_loading(self, layer):  # pragma: no cover
        # Conversion already happened; the fp8 method's hook must not run twice.
        return

    def apply(self, layer, x, bias=None):
        if _CURRENT_MODE not in self.modes:
            return self.fp8_method.apply(layer, x, bias)

        if isinstance(x, tuple):
            raise RuntimeError(
                f"DSV4_FP4_PROJ: layer {layer._dsv4_fp4_prefix} ({self.key}) was "
                "handed a pre-quantized fp8 activation, which the a4w4 path "
                f"cannot consume. Drop '{self.key}' from "
                "SGLANG_DSV4_FP4_PROJ_PREFILL/_DECODE."
            )

        import aiter

        x2d = x.view(-1, x.shape[-1])
        m = x2d.shape[0]
        qx, sx = _mx_quant()(x2d, shuffle=True)
        out = aiter.gemm_a4w4(
            qx,
            layer._dsv4_fp4_w,
            sx,
            layer._dsv4_fp4_s,
            dtype=torch.bfloat16,
            bpreshuffle=True,
        )
        # gemm_a4w4 pads M up to the kernel tile; drop the padding.
        out = out[:m]
        if bias is not None:
            out = out + bias
        return out.view(*x.shape[:-1], out.shape[-1])


# --------------------------------------------------------------------------
# weight conversion
# --------------------------------------------------------------------------


def _fp8_needs_bpreshuffle(fp8_method, layer) -> bool:
    """Whether the fp8 loader would have bpreshuffled this weight.

    Mirrors the gate in Fp8LinearMethod.process_weights_after_loading. We set
    skip_aiter_bpreshuffle on every selected layer so the fp4 conversion sees a
    row-major weight, then re-apply the shuffle here for the layers that keep
    an fp8 weight.
    """
    from sglang.srt.layers.quantization.fp8_utils import (
        _use_aiter_bpreshuffle_gfx95,
        aiter_w8a8_block_fp8_linear,
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )

    if not _use_aiter_bpreshuffle_gfx95:
        return False
    if fp8_method.w8a8_block_fp8_linear is not aiter_w8a8_block_fp8_linear:
        return False
    n, k = layer.weight.shape
    return not use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k)


def _convert_layer(layer, fp8_method, keep_fp8: bool) -> None:
    """Build the MXFP4 copy from the row-major fp8 block-scale weight."""
    from aiter.ops.shuffle import shuffle_weight

    from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

    assert not getattr(layer, "aiter_bpreshuffled", False), (
        f"{layer._dsv4_fp4_prefix}: weight was already bpreshuffled for the fp8 "
        "kernel; the fp4 conversion needs the row-major layout"
    )

    w_bf16 = block_quant_dequant(
        layer.weight.data,
        layer.weight_scale_inv.data.to(torch.float32),
        [128, 128],
        torch.bfloat16,
    )
    qw, sw = _mx_quant()(w_bf16, shuffle=True)
    del w_bf16
    qw = shuffle_weight(qw, layout=(16, 16))

    if keep_fp8:
        layer._dsv4_fp4_w = qw
        layer._dsv4_fp4_s = sw
        # Restore what the fp8 loader would have done had we not skipped it.
        if _fp8_needs_bpreshuffle(fp8_method, layer):
            layer.weight.copy_(shuffle_weight(layer.weight, (16, 16)))
            layer.aiter_bpreshuffled = True
    else:
        # fp4 in every mode: the fp8 weight is dead, hand its storage back.
        layer.weight = torch.nn.Parameter(qw, requires_grad=False)
        layer._dsv4_fp4_w = layer.weight
        layer._dsv4_fp4_s = sw
        if hasattr(layer, "weight_scale_inv"):
            delattr(layer, "weight_scale_inv")
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------

_INSTALLED = False
_CONVERTED: List[str] = []


def converted_layers() -> List[str]:
    return list(_CONVERTED)


def install() -> None:
    """Patch the fp8 quant method so selected DSv4 projections gain an fp4 copy.

    Must run before the model is constructed: ``Fp8Config.get_quant_method`` is
    what tags the layers.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    masks = selected_masks()
    if not masks:
        return

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    orig_get = Fp8Config.get_quant_method
    orig_proc = Fp8LinearMethod.process_weights_after_loading

    def get_quant_method(self, layer, prefix):
        method = orig_get(self, layer, prefix)
        if isinstance(layer, LinearBase) and isinstance(method, Fp8LinearMethod):
            key, modes = _match(prefix, masks)
            if key is not None:
                layer._dsv4_fp4_prefix = prefix
                layer._dsv4_fp4_key = key
                layer._dsv4_fp4_modes = modes
                # Keep the fp8 loader from rewriting the weight into the aiter
                # bpreshuffle layout; the fp4 conversion needs it row-major.
                layer.skip_aiter_bpreshuffle = True
        return method

    def process_weights_after_loading(self, layer):
        orig_proc(self, layer)
        key = getattr(layer, "_dsv4_fp4_key", None)
        if key is None:
            return
        modes = layer._dsv4_fp4_modes
        _convert_layer(layer, self, keep_fp8=len(modes) < 2)
        layer.quant_method = Fp4ProjLinearMethod(self, key, modes)
        _CONVERTED.append(layer._dsv4_fp4_prefix)

    Fp8Config.get_quant_method = get_quant_method
    Fp8LinearMethod.process_weights_after_loading = process_weights_after_loading
    _INSTALLED = True

    both = sorted(k for k, m in masks.items() if len(m) == 2)
    prefill_only = sorted(k for k, m in masks.items() if m == frozenset({PREFILL}))
    decode_only = sorted(k for k, m in masks.items() if m == frozenset({DECODE}))
    logger.info(
        "DSV4_FP4_PROJ: mxfp4 for prefill+decode=[%s] prefill-only=[%s] "
        "decode-only=[%s] (the last two keep both weights resident)",
        ",".join(both),
        ",".join(prefill_only),
        ",".join(decode_only),
    )
