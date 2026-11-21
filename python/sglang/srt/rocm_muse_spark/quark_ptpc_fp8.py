"""Checkpoint contract of AMD Quark "PTPC" FP8 Muse Spark exports on ROCm.

PTPC is Quark's W8A8 FP8 e4m3 scheme with one static scale per weight output
channel and one dynamic scale per activation token. The ROCm Muse model serves
it with the generic Quark W8A8 FP8 linears for attention and the generic Triton
FP8 MoE GEMMs for the routed experts (``muse_moe_fp8``). Everything here is
pure Python + torch, so the contract can be checked without a GPU.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.layers.quantization.quark.quark import QuarkConfig

FP8_DTYPE = torch.float8_e4m3fn

# Quantizer fields that must be unset: anything here turns per-channel /
# per-token FP8 into a group, block or microscaling format the kernels do not run.
_UNSET_FIELDS = (
    "group_size",
    "block_size",
    "scale_format",
    "scale_calculation_mode",
    "mx_element_dtype",
)
# Excluded projections of a decoder layer that this model builds in BF16 anyway.
_BF16_MLP_MODULES = ("gate", "router", "pre_expert_proj", "post_expert_proj")
_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LAYER = r"(?:model\.)?(?:language_model\.)?layers\.(\d+)"
_BF16_MLP_RE = re.compile(_LAYER + r"\.mlp\.(?:" + "|".join(_BF16_MLP_MODULES) + ")")
_EXPERTS_RE = re.compile(_LAYER + r"\.mlp\.experts")
_EXPERT_LEAF_RE = re.compile(
    _LAYER + r"\.mlp\.experts\.(\d+)\.(" + "|".join(_EXPERT_PROJECTIONS) + ")"
)
_DECODER_RE = re.compile(r"(?:model\.)?(?:language_model\.)?layers(?:\.|$)")
_DECODER_ANCESTORS = ("model", "language_model", "model.language_model")


def _as_quark(quant_config: QuantizationConfig | None) -> QuarkConfig | None:
    from sglang.srt.layers.quantization.quark.quark import QuarkConfig

    return quant_config if isinstance(quant_config, QuarkConfig) else None


def _quark_dict(quant_config: QuantizationConfig | None) -> dict:
    quark = _as_quark(quant_config)
    return quark.quant_config if quark is not None else {}


def _exclude_entries(quant_config: QuantizationConfig | None) -> list:
    """The export's exclude list; any other container (a string, a mapping) would be read as characters or keys."""
    entries = _quark_dict(quant_config).get("exclude")
    if entries is None:
        return []
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"exclude is a {type(entries).__name__}, expected a list")
    return list(entries)


def _check_quantizer(
    *, spec: object, role: str, ch_axis: int, dynamic: bool
) -> str | None:
    if not isinstance(spec, Mapping):
        return f"{role} quantizer missing"
    expected = {
        "dtype": "fp8_e4m3",
        "qscheme": "per_channel",
        "ch_axis": ch_axis,
        "is_dynamic": dynamic,
        "symmetric": True,
    }
    for key, value in expected.items():
        # Absent keys fail too: Quark writes all of them, and a missing axis or
        # dynamic flag is exactly the ambiguity this contract must not guess at.
        if key not in spec or spec[key] != value or type(spec[key]) is not type(value):
            return f"{role}.{key}={spec.get(key)!r}, expected {value!r}"
    for key in _UNSET_FIELDS:
        if spec.get(key) is not None:
            return f"{role}.{key}={spec[key]!r}, expected unset"
    if spec.get("scale_type") not in (None, "float"):
        return f"{role}.scale_type={spec['scale_type']!r}, expected 'float'"
    if spec.get("is_scale_quant"):
        return f"{role}.is_scale_quant is set"
    return None


def excluded_expert_layers(
    *, quant_config: QuantizationConfig | None, num_experts: int
) -> frozenset[int]:
    """Decoder layers whose routed experts the export keeps in BF16.

    Quark names excluded experts leaf by leaf (``layers.N.mlp.experts.E.gate_proj``
    for every expert and projection) or as the whole ``layers.N.mlp.experts``
    module. A layer excluding only some of its experts or projections cannot be
    served (one expert bank holds one dtype) and raises.
    """
    leaves: dict[int, set[tuple[int, str]]] = {}
    whole: set[int] = set()
    for entry in _exclude_entries(quant_config):
        if not isinstance(entry, str):
            continue
        if match := _EXPERTS_RE.fullmatch(entry):
            whole.add(int(match.group(1)))
        elif match := _EXPERT_LEAF_RE.fullmatch(entry):
            layer, expert, projection = match.groups()
            leaves.setdefault(int(layer), set()).add((int(expert), projection))
    full = {(e, p) for e in range(num_experts) for p in _EXPERT_PROJECTIONS}
    for layer, names in leaves.items():
        if layer in whole:
            continue
        if names != full:
            raise ValueError(
                f"layer {layer} excludes {len(names)} of {len(full)} expert "
                "projections; only whole-layer expert exclusion is supported"
            )
    return frozenset(whole | set(leaves))


def _exclude_rejection(entries: Iterable) -> str | None:
    for entry in entries:
        if not isinstance(entry, str) or entry.startswith("re:"):
            return f"exclude entry {entry!r} is not a literal module name"
        if any(ch in entry for ch in "*?[]"):
            return f"exclude entry {entry!r} is not a literal module name"
        if entry in _DECODER_ANCESTORS:
            return f"exclude entry {entry!r} names the whole text model"
        if not _DECODER_RE.match(entry):
            continue  # lm_head, embeddings, norms, vision / audio towers
        if (
            _BF16_MLP_RE.fullmatch(entry)
            or _EXPERTS_RE.fullmatch(entry)
            or _EXPERT_LEAF_RE.fullmatch(entry)
        ):
            continue
        # Attention exclusions are refused: the fused q/k/v/o-gate projection
        # has one dtype, and the packed-module mapping does not see the gate.
        return f"exclude entry {entry!r} is not a supported BF16 module"
    return None


def ptpc_fp8_rejection(
    *, quant_config: QuantizationConfig | None, num_experts: int
) -> str | None:
    """Why ``quant_config`` is not the supported Quark PTPC-FP8 export, or None."""
    quark = _as_quark(quant_config)
    if quark is None:
        return "not a Quark config"
    for attr, value in (
        ("online_scheme", quark.online_scheme),
        ("dequantization_config", quark.dequantization_config),
        ("excluded_fp8_config", quark.excluded_fp8_config),
    ):
        if value is not None:
            return f"{attr} is set (online or requantized Quark config)"
    config = quark.quant_config
    global_config = config.get("global_quant_config")
    if not isinstance(global_config, Mapping):
        return "global_quant_config missing"
    reason = _check_quantizer(
        spec=global_config.get("weight"), role="weight", ch_axis=0, dynamic=False
    ) or _check_quantizer(
        spec=global_config.get("input_tensors"),
        role="input_tensors",
        ch_axis=1,
        dynamic=True,
    )
    if reason:
        return reason
    for key in ("output_tensors", "bias"):
        if global_config.get(key) is not None:
            return f"global_quant_config.{key} is quantized"
    for key in (
        "layer_quant_config",
        "layer_type_quant_config",
        "kv_cache_quant_config",
        "softmax_quant_spec",
        "algo_config",
    ):
        if config.get(key):
            return f"{key} is not empty"
    export = config.get("export")
    if not isinstance(export, Mapping) or export.get("weight_format") != (
        "real_quantized"
    ):
        return "export.weight_format is not 'real_quantized'"
    if export.get("kv_cache_group") or export.get("weight_merge_groups"):
        return "export merges KV-cache or weight groups"
    try:
        reason = _exclude_rejection(_exclude_entries(quant_config))
        excluded_expert_layers(quant_config=quant_config, num_experts=num_experts)
    except ValueError as error:
        return str(error)
    return reason


def is_quark_ptpc_fp8(
    *, quant_config: QuantizationConfig | None, num_experts: int
) -> bool:
    return (
        ptpc_fp8_rejection(quant_config=quant_config, num_experts=num_experts) is None
    )


def check_rope_rows(
    *, loaded: torch.Tensor, heads: int, head_dim: int, name: str
) -> None:
    """A q/k weight or per-row scale must carry one row per head channel before the
    rope de-interleave permutes them; a scalar or short scale would otherwise skip
    or misapply the permutation."""
    rows = heads * head_dim
    want_dim = 1 if name.endswith("weight_scale") else 2
    if loaded.dim() != want_dim or loaded.shape[0] != rows or head_dim % 2:
        raise ValueError(
            f"{name}: expected {want_dim}-D tensor with {rows} rows "
            f"({heads} heads x {head_dim}), got shape {tuple(loaded.shape)}"
        )


def check_serialized_dtype(
    *, param: torch.Tensor, loaded: torch.Tensor, name: str
) -> None:
    """``copy_`` casts silently; a PTPC export must never land BF16 values in an
    FP8 parameter, FP8 codes in a BF16 one, or non-fp32 per-channel scales."""
    if (param.dtype == FP8_DTYPE or loaded.dtype == FP8_DTYPE) and (
        param.dtype != loaded.dtype
    ):
        raise ValueError(
            f"{name}: checkpoint dtype {loaded.dtype} does not match parameter "
            f"dtype {param.dtype}"
        )
    if name.endswith("weight_scale") and param.dtype == torch.float32:
        if loaded.dtype != torch.float32:
            raise ValueError(f"{name}: scale dtype {loaded.dtype}, expected float32")


def check_channel_scales(*, name: str, scale: torch.Tensor, rows: int) -> None:
    """Per-channel scales after loading: one per output row, finite and positive.
    Unloaded scales keep their sentinel (negative or NaN) and fail here."""
    if scale.dtype != torch.float32 or scale.shape[-1] != rows:
        raise RuntimeError(
            f"{name}: scale {tuple(scale.shape)} {scale.dtype} does not cover "
            f"{rows} rows in float32"
        )
    bad = ~(torch.isfinite(scale) & (scale > 0))
    if bool(bad.any()):
        raise RuntimeError(
            f"{name}: {int(bad.sum())} of {scale.numel()} scales missing, "
            "non-finite or non-positive"
        )


class LoadCoverage:
    """Pairs each FP8 weight piece loaded in one ``load_weights`` call with its scale.

    A checkpoint shard that carries a scale without its weight (or the reverse)
    would leave uninitialised FP8 codes behind a valid-looking scale, and a piece
    that arrives twice would silently keep whichever copy came last.
    """

    def __init__(self):
        self.weights: set = set()
        self.scales: set = set()

    def add(self, *, key: tuple, suffix: str) -> None:
        seen = {"weight": self.weights, "weight_scale": self.scales}.get(suffix)
        if seen is None:
            return
        if key in seen:
            raise RuntimeError(f"Quark PTPC-FP8 {suffix} {key} loaded twice")
        seen.add(key)

    def check(self) -> None:
        unpaired = sorted(map(str, self.weights ^ self.scales))
        if unpaired:
            raise RuntimeError(
                f"Quark PTPC-FP8 weights and scales unpaired for {len(unpaired)} "
                f"pieces, e.g. {unpaired[:4]}"
            )
