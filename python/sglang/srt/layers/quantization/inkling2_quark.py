"""Layout adapter between AMD Quark ``real_quantized`` exports and Inkling-2's loader.
Validates the export and resolves its fnmatch ``exclude`` and ``layer_quant_config``
globs itself: the generic Quark layer matches only exact and ``re:`` names."""

import copy
import enum
import fnmatch
import logging
import re
from typing import Any

import msgspec
import torch

from sglang.srt.layers import linear
from sglang.srt.layers.quantization.fp8_utils import use_aiter_bpreshuffle_gemm
from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8
from sglang.srt.utils import print_info_once

logger = logging.getLogger(__name__)

ATTENTION_FP8_LINEARS = ("wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn")
# AttnFP8 splits the merged qkvr projection: q, k, v, gate in FP8, r and sinks in BF16.
FP8_QKVR_PROJECTIONS = ("wq_du", "wk_dv", "wv_dv", "gate_attn")
BF16_QKVR_PROJECTIONS = ("wr_du", "attn_sink_proj")

UINT4_GROUP_SIZE = 128
MXFP4_BLOCK_SIZE = 32

# Only the fields that define the stored format are compared.
_UINT4_WEIGHT_SPEC = dict(
    dtype="uint4",
    qscheme="per_group",
    group_size=UINT4_GROUP_SIZE,
    ch_axis=-1,
    symmetric=False,
    zero_point_type="int32",
    scale_type="float",
    is_dynamic=False,
)
_MXFP4_WEIGHT_SPEC = dict(
    dtype="fp4",
    qscheme="per_group",
    group_size=MXFP4_BLOCK_SIZE,
    ch_axis=-1,
    scale_format="e8m0",
    scale_calculation_mode="even",
    round_method="half_even",
    is_dynamic=False,
)
_MXFP4_INPUT_SPEC = {**_MXFP4_WEIGHT_SPEC, "is_dynamic": True}
_FP8_WEIGHT_SPEC = dict(
    dtype="fp8_e4m3",
    qscheme="per_channel",
    ch_axis=0,
    symmetric=True,
    is_dynamic=False,
)
_FP8_INPUT_SPEC = dict(
    dtype="fp8_e4m3",
    qscheme="per_channel",
    ch_axis=1,
    per_token=True,
    is_dynamic=True,
    symmetric=True,
)

# Checkpoint names, ``model.`` prefix included.
_EXPERT_TENSOR = re.compile(
    r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(?:w13|w2)_weight(?:_(scale|zero_point))?"
)
_ATTENTION_TENSOR = re.compile(
    r"model\.llm\.layers\.(\d+)\.attn\.(?:"
    + "|".join(ATTENTION_FP8_LINEARS)
    + r")\.(weight|weight_scale)"
)

_BF16_OR_F32 = (torch.bfloat16, torch.float32)
# gfx94x converts the e4m3fn weights to fnuz after loading.
_FP8_E4M3 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


class Inkling2Fp8Attention(msgspec.Struct, frozen=True, kw_only=True):
    """FP8 weight and input specs shared by the five attention linears."""

    weight_config: dict[str, Any]
    input_config: dict[str, Any]


def _check_spec(what: str, spec: Any, expected: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"{what} must be a dict, got {spec!r}")
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f"{what}: {key} is {spec.get(key)!r}, expected {value!r}")


def _check_none(what: str, value: Any) -> None:
    if value is not None:
        raise ValueError(f"{what} must be null, got {value!r}")


def _validate_export(quantization_config: dict[str, Any]) -> None:
    if quantization_config.get("quant_method") != "quark":
        raise ValueError(
            f"quant_method is {quantization_config.get('quant_method')!r}, "
            "expected 'quark'"
        )
    export = quantization_config.get("export")
    if not isinstance(export, dict):
        raise ValueError(
            "The Quark 'export' section is missing: Inkling2 serves real_quantized "
            "Quark exports only (online requantization is not supported)"
        )
    for key, supported in (
        ("weight_format", "real_quantized"),
        ("pack_method", "reorder"),
    ):
        if export.get(key) != supported:
            raise ValueError(
                f"export.{key} is {export.get(key)!r}, expected {supported!r}"
            )
    if export.get("kv_cache_group"):
        raise ValueError(
            "export.kv_cache_group must be empty (quantized KV cache is not "
            f"supported), got {export['kv_cache_group']!r}"
        )
    if export.get("weight_merge_groups"):
        raise ValueError(
            "export.weight_merge_groups must be null, got "
            f"{export['weight_merge_groups']!r}"
        )
    for key in ("kv_cache_quant_config", "layer_type_quant_config"):
        if quantization_config.get(key):
            raise ValueError(f"{key} must be empty, got {quantization_config[key]!r}")
    _check_none("softmax_quant_spec", quantization_config.get("softmax_quant_spec"))
    exclude = quantization_config.get("exclude")
    if exclude and not (
        isinstance(exclude, list)
        and all(isinstance(pattern, str) for pattern in exclude)
    ):
        raise ValueError(f"exclude must be a list of glob strings, got {exclude!r}")


def _expert_format(global_config: Any) -> str:
    if not isinstance(global_config, dict):
        raise ValueError(f"global_quant_config must be a dict, got {global_config!r}")
    _check_none("global_quant_config.bias", global_config.get("bias"))
    _check_none(
        "global_quant_config.output_tensors", global_config.get("output_tensors")
    )
    weight = global_config.get("weight")
    dtype = weight.get("dtype") if isinstance(weight, dict) else None
    if dtype == "uint4":
        _check_spec("global_quant_config.weight", weight, _UINT4_WEIGHT_SPEC)
        if global_config.get("input_tensors") is not None:
            raise ValueError(
                "global_quant_config.input_tensors must be null for uint4 (W4A16)"
            )
        return "uint4_awq"
    if dtype == "fp4":
        _check_spec("global_quant_config.weight", weight, _MXFP4_WEIGHT_SPEC)
        if global_config.get("input_tensors") is None:
            raise ValueError(
                "global_quant_config.input_tensors is null: fp4 exports need dynamic "
                "MXFP4 inputs (W4A4)"
            )
        _check_spec(
            "global_quant_config.input_tensors",
            global_config["input_tensors"],
            _MXFP4_INPUT_SPEC,
        )
        return "mxfp4"
    raise ValueError(
        f"global_quant_config.weight.dtype is {dtype!r}, expected 'uint4' or 'fp4'"
    )


def _attention_fp8(layer_quant_config: Any) -> Inkling2Fp8Attention | None:
    if layer_quant_config is None or layer_quant_config == {}:
        return None
    if not isinstance(layer_quant_config, dict):
        raise ValueError(
            f"layer_quant_config must be a dict, got {layer_quant_config!r}"
        )
    supported = {f"model.llm.layers.*.attn.{name}" for name in ATTENTION_FP8_LINEARS}
    if set(layer_quant_config) != supported:
        raise ValueError(
            f"layer_quant_config keys {sorted(layer_quant_config)} differ from the "
            f"supported FP8 attention set {sorted(supported)}"
        )
    specs = [layer_quant_config[key] for key in sorted(supported)]
    for key, spec in layer_quant_config.items():
        what = f"layer_quant_config[{key!r}]"
        if not isinstance(spec, dict):
            raise ValueError(f"{what} must be a dict, got {spec!r}")
        _check_none(f"{what}.bias", spec.get("bias"))
        _check_none(f"{what}.output_tensors", spec.get("output_tensors"))
        _check_spec(f"{what}.weight", spec.get("weight"), _FP8_WEIGHT_SPEC)
        _check_spec(f"{what}.input_tensors", spec.get("input_tensors"), _FP8_INPUT_SPEC)
    if any(spec != specs[0] for spec in specs):
        raise ValueError(
            "the five FP8 attention specs in layer_quant_config must be identical"
        )
    return Inkling2Fp8Attention(
        weight_config=copy.deepcopy(specs[0]["weight"]),
        input_config=copy.deepcopy(specs[0]["input_tensors"]),
    )


class _Resolution(enum.Enum):
    EXCLUDED = "excluded"
    LAYER = "layer"
    GLOBAL = "global"


def _resolve(name: str, exclude, layer_patterns) -> _Resolution:
    """Quark's matching order, with fnmatch globs."""
    if any(fnmatch.fnmatchcase(name, pattern) for pattern in exclude):
        return _Resolution.EXCLUDED
    if any(fnmatch.fnmatchcase(name, pattern) for pattern in layer_patterns):
        return _Resolution.LAYER
    return _Resolution.GLOBAL


def _check_resolution(num_layers, exclude, layer_patterns, fp8_attention) -> None:
    """Experts must take the global spec, the five attention linears the FP8 spec (or
    be excluded for BF16 attention), and everything else must be excluded."""
    attention = _Resolution.LAYER if fp8_attention is not None else _Resolution.EXCLUDED
    kept_bf16 = ["model.llm.unembed", "model.visual.x", "mtp.x"]
    for layer in range(num_layers):
        base = f"model.llm.layers.{layer}"
        outcome = _resolve(f"{base}.mlp.experts", exclude, layer_patterns)
        if outcome != _Resolution.GLOBAL:
            raise ValueError(
                f"{base}.mlp.experts resolves to {outcome.value!r} instead of the "
                "global spec: BF16 or per-layer experts are not supported"
            )
        for name in ATTENTION_FP8_LINEARS:
            outcome = _resolve(f"{base}.attn.{name}", exclude, layer_patterns)
            if outcome == attention:
                continue
            if fp8_attention is not None:
                raise ValueError(
                    f"{base}.attn.{name} resolves to {outcome.value!r} instead of the "
                    "FP8 layer_quant_config spec"
                )
            raise ValueError(
                f"export quantizes {base}.attn.{name} ({outcome.value!r}), but "
                "without an FP8 layer_quant_config this loader keeps attention in "
                "BF16: exclude model.llm.layers.*.attn.*"
            )
        kept_bf16 += [f"{base}.attn.{name}" for name in BF16_QKVR_PROJECTIONS]
        kept_bf16.append(f"{base}.mlp.gate")
    for module in kept_bf16:
        if _resolve(module, exclude, layer_patterns) != _Resolution.EXCLUDED:
            raise ValueError(
                f"export quantizes {module}, which this loader keeps in BF16"
            )


class _Fp8AttentionScheme(QuarkW8A8Fp8):
    """QuarkW8A8Fp8 that refuses to lose the FP8 weight: for N % 16 != 0 on ROCm the
    base scheme silently dequantizes it to BF16 and drops the scale."""

    def process_weights_after_loading(self, layer) -> None:
        super().process_weights_after_loading(layer)
        if layer.weight_scale is None or layer.weight.dtype not in _FP8_E4M3:
            raise ValueError(
                f"The FP8 attention linear with {layer.weight.shape[1]} outputs lost "
                f"its FP8 weight while loading (weight {layer.weight.dtype}, "
                f"weight_scale {layer.weight_scale}): it would run dequantized"
            )
        backend = (
            "aiter bpreshuffle"
            if use_aiter_bpreshuffle_gemm(layer.weight.shape[1])
            else "torch._scaled_mm"
        )
        print_info_once(f"Inkling2 Quark attention FP8 GEMM: {backend}")


class Inkling2Fp8Qkvr(torch.nn.ModuleDict):
    """AttnFP8 qkvr: an FP8 linear for q, k, v and gate and a BF16 one for r and sinks.
    The outputs are joined back in merged order: the prologue kernels read q, k, v and r
    with one row stride.
    Called like MergedColumnParallelLinear: forward(x) -> (out, None)."""

    # Inkling2Attention reads qkvr.bias; FP8 attention refuses biases.
    bias = None

    def __init__(self, input_size, output_sizes, *, quant_config, prefix, **kwargs):
        super().__init__(
            {
                "fp8": linear.MergedColumnParallelLinear(
                    input_size,
                    [*output_sizes[:3], output_sizes[4]],
                    quant_config=quant_config,
                    prefix=f"{prefix}.fp8",
                    **kwargs,
                ),
                "bf16": linear.MergedColumnParallelLinear(
                    input_size,
                    [output_sizes[3], *output_sizes[5:]],
                    prefix=f"{prefix}.bf16",
                    **kwargs,
                ),
            }
        )
        self.qkv = sum(self["fp8"].output_partition_sizes[:3])
        self.r = self["bf16"].output_partition_sizes[0]

    def forward(self, x):
        fp8, bf16 = self["fp8"](x)[0], self["bf16"](x)[0]
        qkv, r = self.qkv, self.r
        out = torch.cat([fp8[:, :qkv], bf16[:, :r], fp8[:, qkv:], bf16[:, r:]], -1)
        return out, None


class Inkling2QuarkLayout(Inkling2QuantConfig):
    """Validated Quark export: expert format, attention precision and tensor dtype
    policy; built from config.json, not a registered method."""

    def __init__(
        self,
        *,
        expert_format: str,
        attention_fp8: Inkling2Fp8Attention | None,
        num_layers: int,
    ):
        # No parameter groups: the Quark path never builds NVFP4/MXFP8 Inkling2Experts.
        super().__init__({"model": {"parameter_groups": []}})
        self.expert_format = expert_format
        self.attention_fp8 = attention_fp8
        self.num_layers = num_layers
        # FP32 logits like the native release; the base class default is BF16.
        self.unembed_output_dtype = torch.float32

    @classmethod
    def from_quark_config(cls, quark_config, text_config) -> "Inkling2QuarkLayout":
        if not quark_config.is_prequantized:
            raise ValueError(
                "Inkling2 serves prequantized Quark exports; online requantization "
                "(--quantization quark_mxfp4) is not supported"
            )
        return cls.from_quantization_dict(
            quark_config.quant_config, text_config.num_hidden_layers
        )

    @classmethod
    def from_quantization_dict(
        cls, quantization_config: dict[str, Any], num_layers: int
    ) -> "Inkling2QuarkLayout":
        """Validates the ``quantization_config`` section of config.json."""
        _validate_export(quantization_config)
        expert_format = _expert_format(quantization_config.get("global_quant_config"))
        layer_quant_config = quantization_config.get("layer_quant_config")
        fp8_attention = _attention_fp8(layer_quant_config)
        exclude = tuple(quantization_config.get("exclude") or ())
        layer_patterns = tuple(layer_quant_config or ())
        _check_resolution(num_layers, exclude, layer_patterns, fp8_attention)
        layout = cls(
            expert_format=expert_format,
            attention_fp8=fp8_attention,
            num_layers=num_layers,
        )
        layout._check_linear_quant_environment()
        logger.info("Inkling2 Quark export: %s", layout.describe())
        return layout

    def _check_linear_quant_environment(self) -> None:
        if self.attention_fp8 is None:
            return
        # RowParallelLinear would silently build wo_ud in BF16. linear.py reads the
        # variable once, at import, on ROCm only.
        if linear._disable_hip_linear_quant:
            raise ValueError(
                "SGLANG_ROCM_DISABLE_LINEARQUANT=1 would build the FP8 attention "
                "output projection (wo_ud) unquantized; unset it to serve this export"
            )

    def describe(self) -> str:
        scheme = "W4A4" if self.expert_format == "mxfp4" else "W4A16"
        attention = "fp8 (W8A8)" if self.attention_fp8 is not None else "bf16"
        return f"experts={self.expert_format} ({scheme}), attention={attention}"

    def _expert_field(self, checkpoint_name: str) -> str | None:
        match = _EXPERT_TENSOR.fullmatch(checkpoint_name)
        if match is None or int(match[1]) >= self.num_layers:
            return None
        return match[2] or "weight"

    def _attention_field(self, checkpoint_name: str) -> str | None:
        match = _ATTENTION_TENSOR.fullmatch(checkpoint_name)
        if self.attention_fp8 is None or match is None:
            return None
        return match[2] if int(match[1]) < self.num_layers else None

    def checkpoint_tensor(
        self, checkpoint_name: str, tensor: torch.Tensor
    ) -> torch.Tensor:
        """Checks the tensor dtype and upcasts BF16 scales to FP32 (lossless)."""
        accepted, upcast = self._dtype_policy(checkpoint_name)
        if accepted is not None and tensor.dtype not in accepted:
            raise ValueError(
                f"{checkpoint_name}: expected "
                f"{' or '.join(str(dtype) for dtype in accepted)}, got {tensor.dtype}"
            )
        if upcast and tensor.dtype != torch.float32:
            return tensor.to(torch.float32)
        return tensor

    def _dtype_policy(
        self, checkpoint_name: str
    ) -> tuple[tuple[torch.dtype, ...] | None, bool]:
        """(accepted dtypes, upcast to FP32). Other tensors get (None, False): the
        loader checks their dtype against the parameter."""
        if (field := self._expert_field(checkpoint_name)) is not None:
            if self.expert_format == "mxfp4":
                if field == "zero_point":
                    raise ValueError(
                        f"{checkpoint_name}: MXFP4 exports have no zero points"
                    )
                return (torch.uint8,), False
            if field == "scale":
                return _BF16_OR_F32, True
            return (torch.int32,), False
        if (field := self._attention_field(checkpoint_name)) is not None:
            if field == "weight_scale":
                return _BF16_OR_F32, True
            # OCP e4m3fn only: byte 0x80 is -0.0 here and NaN in e4m3fnuz.
            return (torch.float8_e4m3fn,), False
        return None, False

    def attention_slot(self, name: str, projection: str, index: int) -> tuple[str, int]:
        """(name, shard id) of a qkvr projection. With AttnFP8, q, k, v and gate go to
        the FP8 module and r and sinks to the BF16 one, each numbered from 0."""
        if self.attention_fp8 is None:
            return name, index
        if projection in FP8_QKVR_PROJECTIONS:
            return (
                name.replace(".qkvr.", ".qkvr.fp8."),
                FP8_QKVR_PROJECTIONS.index(projection),
            )
        return (
            name.replace(".qkvr.", ".qkvr.bf16."),
            BF16_QKVR_PROJECTIONS.index(projection),
        )

    def expected_shards(self, param_name: str, default: set[int]) -> set[int]:
        """The shards ``load_weights`` must have seen for a packed parameter."""
        if self.attention_fp8 is None:
            return default
        if param_name.endswith((".qkvr.fp8.weight", ".qkvr.fp8.weight_scale")):
            return set(range(len(FP8_QKVR_PROJECTIONS)))
        if param_name.endswith(".qkvr.bf16.weight"):
            return set(range(len(BF16_QKVR_PROJECTIONS)))
        return default

    def get_quant_method(self, layer, prefix: str):
        """AttnFP8: the W8A8 FP8 method of the q/k/v/gate module and wo_ud."""
        if self.attention_fp8 is None or not prefix.endswith((".qkvr.fp8", ".wo_ud")):
            return None
        layer.scheme = _Fp8AttentionScheme(
            self.attention_fp8.weight_config, self.attention_fp8.input_config
        )
        return QuarkLinearMethod(self)
