"""HY4 DFly: a block-parallel draft bundled in the target checkpoint.

The draft layers are stored under ``model.dfly_layers.*`` and described by the
``dfly_*`` keys of the target text config. One draft forward runs the bonus
token at slot 0 and ``N`` mask tokens; each draft layer keeps its KV in the
pool of a target layer, and its sparse attention reuses a target indexer's
top-k.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import msgspec

from sglang.srt.configs.hy_v4 import HYV4Config
from sglang.srt.model_executor.cuda_graph_config import Backend

DFLY_CHECKPOINT_PREFIX = "model.dfly_layers."
DFLY_ARCHITECTURE = "HYV4ForCausalLMDFly"

# Key spellings follow the reference checkpoint config.
_DFLY_KEYS = (
    "dfly_num_hidden_layers",
    "dfly_block_size",
    "dfly_mask_token_id",
    "dfly_target_index_layer",
    "dfly_target_kv_layers",
)
_FIXED_DFLY_FIELDS = {
    "enable_ihc": False,
    "mlp_type": "dense",
    "attn_causal": True,
    "mtp_fusion": "slot0",
    "enable_hidden_correction": True,
}
_MAX_NUM_SPECULATIVE_TOKENS = 16
# The sparse MLA kernels take top-k widths in 128-index blocks.
_TOPK_ALIGNMENT = 128
# Draft layers write BF16 MLA KV into target layers; FP8 KV has no draft path yet.
_SUPPORTED_KV_CACHE_DTYPES = ("auto", "bf16", "bfloat16")
# The only DSA kernel whose external-row metadata the draft overrides.
_SUPPORTED_DSA_DECODE_BACKEND = "flashmla_sparse"
_FP8_FORMAT = "float-quantized"
_MXFP4_FORMAT = "mxfp4-pack-quantized"
_CHECKPOINT_FORMATS = (_FP8_FORMAT, _MXFP4_FORMAT, "mixed-precision")
_FP8_BLOCK_WEIGHTS = dict(
    num_bits=8,
    type="float",
    strategy="block",
    block_structure=[128, 128],
    symmetric=True,
    dynamic=False,
)
_FP8_GROUP_ACTIVATIONS = dict(
    num_bits=8,
    type="float",
    strategy="group",
    group_size=128,
    symmetric=True,
    dynamic=True,
)
_MXFP4_WEIGHTS = dict(
    num_bits=4,
    type="float",
    strategy="group",
    group_size=32,
    symmetric=True,
    dynamic=False,
)


class HYV4DFlySpec(msgspec.Struct, frozen=True, kw_only=True):
    num_hidden_layers: int
    num_speculative_tokens: int
    mask_token_id: int
    target_index_layers: tuple[int, ...]
    target_kv_layers: tuple[int, ...]
    target_index_topk: int
    context_local_window: int
    intermediate_size: int
    hidden_correction_intermediate_size: int

    @property
    def query_width(self) -> int:
        return self.num_speculative_tokens + 1

    @property
    def draft_index_topk(self) -> int:
        width = self.target_index_topk + self.context_local_window + self.query_width
        return -(-width // _TOPK_ALIGNMENT) * _TOPK_ALIGNMENT


def checkpoint_bundles_dfly_draft(config: Any) -> bool:
    return any(getattr(config, key, None) is not None for key in _DFLY_KEYS)


def resolve_hyv4_dfly_spec(
    text_config: Any, *, speculative_num_draft_tokens: Optional[int] = None
) -> HYV4DFlySpec:
    """Validate the bundled draft against the reference contract.

    ``speculative_num_draft_tokens`` is the server's per-step block width, the
    bonus anchor plus the drafts; None takes ``dfly_block_size``.
    """
    if not isinstance(text_config, HYV4Config):
        raise ValueError(
            "HYV4 DFly drafts are bundled only with HYV4 checkpoints; got text "
            f"config {type(text_config).__name__}"
        )
    num_layers = _dfly_key(text_config, "num_hidden_layers")
    if not _is_int(num_layers) or num_layers < 1:
        raise ValueError(
            f"HYV4 DFly requires a positive dfly_num_hidden_layers, got {num_layers!r}"
        )
    for name, required in _FIXED_DFLY_FIELDS.items():
        value = _dfly_key(text_config, name, required)
        if type(value) is not type(required) or value != required:
            raise ValueError(
                f"HYV4 DFly supports only dfly_{name}={required!r}, got {value!r}"
            )
    mask_token_id = _dfly_key(text_config, "mask_token_id")
    if not _is_int(mask_token_id) or not 0 <= mask_token_id < text_config.vocab_size:
        raise ValueError(
            f"HYV4 DFly dfly_mask_token_id={mask_token_id!r} is outside the "
            f"vocabulary of size {text_config.vocab_size}"
        )
    target_index_topk = _dfly_key(
        text_config, "target_index_topk", text_config.index_topk
    )
    if (
        not _is_int(target_index_topk)
        or target_index_topk <= 0
        or target_index_topk != text_config.index_topk
    ):
        raise ValueError(
            f"HYV4 DFly dfly_target_index_topk={target_index_topk!r} must be a "
            f"positive int equal to the target index_topk={text_config.index_topk!r}"
        )
    window = _dfly_key(text_config, "context_local_window")
    if window is None:
        window = 0
    if not _is_int(window) or window < 0:
        raise ValueError(
            f"HYV4 DFly dfly_context_local_window must be a non-negative int, got {window!r}"
        )
    ffn_size = _dfly_key(text_config, "intermediate_size")
    if ffn_size is None:
        ffn_size = text_config.intermediate_size
    correction_size = _dfly_key(
        text_config, "hidden_correction_intermediate_size", text_config.hidden_size
    )
    for name, size in (
        ("intermediate_size", ffn_size),
        ("hidden_correction_intermediate_size", correction_size),
    ):
        if not _is_int(size) or size < 1:
            raise ValueError(
                f"HYV4 DFly dfly_{name} must be a positive int, got {size!r}"
            )
    post_proj_norm = _dfly_key(text_config, "attn_post_proj_norm")
    if post_proj_norm is not None and type(post_proj_norm) is not bool:
        raise ValueError(
            "HYV4 DFly dfly_attn_post_proj_norm must be boolean or None, "
            f"got {post_proj_norm!r}"
        )
    index_layers = _target_layers(
        text_config, "target_index_layer", num_layers, scalar_ok=True
    )
    kv_layers = _target_layers(text_config, "target_kv_layers", num_layers)
    if len(set(kv_layers)) != num_layers:
        raise ValueError(
            f"HYV4 DFly needs a distinct target KV layer per draft layer, got {kv_layers}"
        )
    shared = [i for i in index_layers if text_config.indexer_types[i] != "full"]
    if shared:
        raise ValueError(
            "HYV4 DFly dfly_target_index_layer must name layers whose indexer_types "
            f"entry is 'full'; layers {shared} reuse a previous layer's top-k"
        )
    return HYV4DFlySpec(
        num_hidden_layers=num_layers,
        num_speculative_tokens=_resolve_num_speculative_tokens(
            text_config, speculative_num_draft_tokens
        ),
        mask_token_id=mask_token_id,
        target_index_layers=index_layers,
        target_kv_layers=kv_layers,
        target_index_topk=target_index_topk,
        context_local_window=window,
        intermediate_size=ffn_size,
        hidden_correction_intermediate_size=correction_size,
    )


def build_hyv4_dfly_draft_config(text_config: Any, spec: HYV4DFlySpec) -> Any:
    """The draft's text config: the target's attention geometry with dense,
    single-residual layers that consume external top-k."""
    config = copy.deepcopy(text_config)
    num_layers = spec.num_hidden_layers
    config.num_hidden_layers = num_layers
    config.enable_ihc = False
    config.n_routed_experts = 0
    config.n_shared_experts = 0
    config.num_nextn_predict_layers = 0
    config.attn_post_proj_norm = _dfly_key(text_config, "attn_post_proj_norm")
    config.intermediate_size = spec.intermediate_size
    config.mlp_layer_types = ["dense"] * num_layers
    config.layer_types = [text_config.layer_types[i] for i in spec.target_kv_layers]
    # Every draft layer takes its top-k from the target, so none builds an indexer.
    config.indexer_types = ["shared"] * num_layers
    config.index_topk = spec.draft_index_topk
    config.hidden_correction_intermediate_size = (
        spec.hidden_correction_intermediate_size
    )
    config.architectures = [DFLY_ARCHITECTURE]
    return config


def check_hyv4_dfly_quant_config(quantization_config: Optional[dict]) -> None:
    """Accept an unquantized checkpoint or compressed-tensors groups that are
    either W8A8 FP8 (static 128x128 weight blocks, dynamic group-128
    activations) or weight-only MXFP4 (group 32, uint8 scales)."""
    if quantization_config is None:
        return
    error = (
        "HYV4 DFly supports BF16 checkpoints or compressed-tensors W8A8 FP8 "
        "(static 128x128 weight blocks, dynamic group-128 activations) and/or "
        "weight-only MXFP4 (group 32, uint8 scales) groups with no "
        f"kv_cache_scheme or transform_config; got {quantization_config!r}"
    )
    checkpoint_format = quantization_config.get("format")
    if (
        quantization_config.get("quant_method") != "compressed-tensors"
        or checkpoint_format not in _CHECKPOINT_FORMATS
        or quantization_config.get("kv_cache_scheme") is not None
        or quantization_config.get("transform_config") is not None
    ):
        raise ValueError(error)
    groups = quantization_config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(error)
    for group in groups.values():
        if not isinstance(group, dict) or group.get("output_activations") is not None:
            raise ValueError(error)
        if not (
            _is_block_fp8_group(group, checkpoint_format)
            or _is_mxfp4_group(group, checkpoint_format)
        ):
            raise ValueError(error)


def has_hyv4_dfly_mxfp4_groups(quantization_config: Optional[dict]) -> bool:
    """Whether a checkpoint that passed check_hyv4_dfly_quant_config holds a
    weight-only MXFP4 group."""
    if quantization_config is None:
        return False
    checkpoint_format = quantization_config.get("format")
    return any(
        _is_mxfp4_group(group, checkpoint_format)
        for group in quantization_config["config_groups"].values()
    )


def hyv4_dfly_cuda_graph_conflicts(
    *, decode_backend: str, prefill_backend: str
) -> list[str]:
    """The CUDA graph settings DFly cannot serve. A full decode graph captures
    the draft together with the target verify; a breakable prefill graph
    records the target top-k at eager breaks, so replay keeps live rows."""
    conflicts = []
    if decode_backend not in (Backend.FULL, Backend.DISABLED):
        conflicts.append(
            f"the {decode_backend!r} decode CUDA graph backend (use full, or "
            "pass --disable-decode-cuda-graph)"
        )
    if prefill_backend not in (Backend.BREAKABLE, Backend.DISABLED):
        conflicts.append(
            f"the {prefill_backend!r} prefill CUDA graph backend (use breakable, "
            "or pass --disable-prefill-cuda-graph)"
        )
    return conflicts


def check_hyv4_dfly_serving(
    *,
    spec: HYV4DFlySpec,
    draft_config: Any,
    dtype: str,
    tp_size: int,
    attn_dp_size: int,
    pp_size: int,
    attn_cp_size: int,
    dcp_size: int,
    disaggregation_mode: str,
    enable_lora: bool,
    enable_two_batch_overlap: bool,
    kv_cache_dtype: str,
    draft_kv_cache_dtype: Optional[str],
    draft_quantization: Optional[str],
    draft_attention_backend: Optional[str],
    speculative_eagle_topk: Optional[int],
    decode_cuda_graph_backend: str,
    prefill_cuda_graph_backend: str,
    enable_mixed_chunk: bool,
) -> None:
    """Fail at startup on every serving combination the DFly draft cannot run."""
    unsupported = hyv4_dfly_cuda_graph_conflicts(
        decode_backend=decode_cuda_graph_backend,
        prefill_backend=prefill_cuda_graph_backend,
    )
    if enable_mixed_chunk:
        unsupported.append("mixed chunked prefill")
    if dtype not in ("auto", "bfloat16"):
        unsupported.append(f"dtype={dtype!r} (BF16 only)")
    for name, size in (
        ("attention DP size", attn_dp_size),
        ("pp_size", pp_size),
        ("attn_cp_size", attn_cp_size),
        ("dcp_size", dcp_size),
    ):
        if size != 1:
            unsupported.append(f"{name}={size}")
    if disaggregation_mode != "null":
        unsupported.append(f"disaggregation_mode={disaggregation_mode!r}")
    if enable_lora:
        unsupported.append("LoRA")
    if enable_two_batch_overlap:
        unsupported.append("two-batch overlap")
    if kv_cache_dtype not in _SUPPORTED_KV_CACHE_DTYPES:
        unsupported.append(f"kv_cache_dtype={kv_cache_dtype!r}")
    # Draft layers store KV in target layers, so the draft cannot pick its own dtype.
    if draft_kv_cache_dtype is not None and _kv_dtype(draft_kv_cache_dtype) != (
        _kv_dtype(kv_cache_dtype)
    ):
        unsupported.append(f"draft kv_cache_dtype={draft_kv_cache_dtype!r}")
    if draft_quantization is not None:
        unsupported.append(
            f"draft quantization={draft_quantization!r} (the draft follows the checkpoint)"
        )
    if draft_attention_backend is not None:
        unsupported.append(f"draft attention backend={draft_attention_backend!r}")
    if speculative_eagle_topk not in (None, 1):
        unsupported.append(f"speculative_eagle_topk={speculative_eagle_topk}")
    for name, size in (
        ("num_attention_heads", draft_config.num_attention_heads),
        ("intermediate_size", draft_config.intermediate_size),
        (
            "dfly_hidden_correction_intermediate_size",
            spec.hidden_correction_intermediate_size,
        ),
    ):
        if size % tp_size:
            unsupported.append(f"{name}={size} not divisible by TP {tp_size}")
    if unsupported:
        raise ValueError(
            "HYV4 DFly serves BF16 or compressed-tensors FP8/MXFP4 checkpoints "
            "with TP only, breakable or eager prefill, a BF16 KV cache and a "
            "draft that inherits the target's KV dtype and quantization. "
            "Unsupported: " + ", ".join(unsupported) + "."
        )


def _is_block_fp8_group(group: dict, checkpoint_format: Optional[str]) -> bool:
    # A group's own format wins; a mixed-precision checkpoint states it per group.
    return (
        (group.get("format") or checkpoint_format) == _FP8_FORMAT
        and _matches(group.get("weights"), _FP8_BLOCK_WEIGHTS)
        and _matches(group.get("input_activations"), _FP8_GROUP_ACTIVATIONS)
    )


def _is_mxfp4_group(group: dict, checkpoint_format: Optional[str]) -> bool:
    weights = group.get("weights")
    return (
        (group.get("format") or checkpoint_format) == _MXFP4_FORMAT
        and group.get("input_activations") is None
        and _matches(weights, _MXFP4_WEIGHTS)
        and weights.get("scale_dtype") in (None, "uint8", "torch.uint8")
    )


def _matches(args: Any, expected: dict) -> bool:
    return isinstance(args, dict) and all(args.get(k) == v for k, v in expected.items())


def _kv_dtype(name: str) -> str:
    return "bfloat16" if name in _SUPPORTED_KV_CACHE_DTYPES else name


def _dfly_key(config: Any, name: str, default: Any = None) -> Any:
    # dfly_* keys are optional checkpoint fields, absent on non-DFly checkpoints.
    return copy.deepcopy(getattr(config, f"dfly_{name}", default))


def _is_int(value: Any) -> bool:
    return type(value) is int


def _resolve_num_speculative_tokens(
    text_config: Any, speculative_num_draft_tokens: Optional[int]
) -> int:
    block_size = speculative_num_draft_tokens
    name = "speculative_num_draft_tokens"
    if block_size is None:
        block_size = _dfly_key(text_config, "block_size")
        name = "dfly_block_size"
    # The block counts the bonus anchor in addition to the drafts.
    if not _is_int(block_size) or not 2 <= block_size <= (
        _MAX_NUM_SPECULATIVE_TOKENS + 1
    ):
        raise ValueError(
            f"HYV4 DFly requires {name} (anchor plus drafts) in "
            f"[2, {_MAX_NUM_SPECULATIVE_TOKENS + 1}], got {block_size!r}"
        )
    return block_size - 1


def _target_layers(
    text_config: Any, name: str, num_layers: int, *, scalar_ok: bool = False
) -> tuple[int, ...]:
    layers = _dfly_key(text_config, name)
    if scalar_ok and _is_int(layers):
        layers = [layers] * num_layers
    num_target_layers = text_config.num_hidden_layers
    if (
        not isinstance(layers, (list, tuple))
        or len(layers) != num_layers
        or any(not _is_int(i) or not 1 <= i < num_target_layers for i in layers)
    ):
        raise ValueError(
            f"HYV4 DFly dfly_{name} needs one target layer in [1, "
            f"{num_target_layers}) per draft layer ({num_layers}), got {layers!r}"
        )
    return tuple(layers)
