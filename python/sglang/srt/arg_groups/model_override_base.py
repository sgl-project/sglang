"""What the per-model override declarations are written against.

The declarations themselves live one directory down, in
``arg_groups/model_overrides/``: one module per model family, mirroring the
``models/`` naming. This module is what they all import -- the registry they
register into, the read-only views they are handed, and the few accessors that
answer questions about the model. It deliberately depends on nothing in
``overrides.py``, so a family module never has to import its way back up.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import is_mps, is_no_spec_infer_or_topk_one

logger = logging.getLogger(__name__)


# Constant per-architecture overrides (populated by the migration sweeps).
MODEL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # These models run in bfloat16 regardless of the requested dtype
    # (faithful port of the legacy unconditional arch branch).
    "MistralLarge3ForCausalLM": {"dtype": "bfloat16"},
    "PixtralForConditionalGeneration": {"dtype": "bfloat16"},
}


# Derived per-architecture override providers, in registration order.
_MODEL_OVERRIDE_FNS: Dict[str, List[Callable[..., dict]]] = {}


# Predicate-keyed providers, in registration order — for legacy branches
# matched by substring/predicate on the architecture string rather than an
# exact name (e.g. '"Step3p5ForCausalLM" in model_arch').
_PREDICATE_OVERRIDE_FNS: List[Tuple[Callable[[str], bool], Callable[..., dict]]] = []


def register_model_override(architecture: str):
    """Register a derived-override provider for ``architecture``.

    The decorated callable receives ``(server_args, hf_config)``, must not
    mutate either, and returns a ``{field: resolved_value}`` dict (possibly
    empty when nothing applies). Providers needing derived model data beyond
    the HF config go through ``model_config_of(server_args)`` (cached,
    read-only) — never anything mutating.
    """

    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        _MODEL_OVERRIDE_FNS.setdefault(architecture, []).append(fn)
        return fn

    return decorator


def register_model_override_predicate(predicate: Callable[[str], bool]):
    """Register a derived-override provider keyed by an architecture
    predicate. Same callable contract as ``register_model_override``."""

    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        _PREDICATE_OVERRIDE_FNS.append((predicate, fn))
        return fn

    return decorator


def _invoke_provider(
    fn: Callable[..., dict], server_args: Any, hf_config: Any
) -> Dict[str, Any]:
    declared = fn(server_args, hf_config)
    if not isinstance(declared, dict):
        raise TypeError(
            f"model override provider {fn.__qualname__} must return a dict, "
            f"got {type(declared).__name__}"
        )
    return declared


class ResolvedView:
    """Read-only view of the resolving configuration handed to post-process
    passes: the accumulated declarations overlaid on the pristine
    ``server_args`` (residual imperative writes of non-resolved fields show
    through the fallthrough) — exactly the state the legacy handler at the
    same slot observed. Writes are rejected: passes return declarations.
    """

    __slots__ = ("_server_args", "_overlay")

    def __init__(self, server_args: Any, overlay: Optional[Dict[str, Any]] = None):
        object.__setattr__(self, "_server_args", server_args)
        object.__setattr__(self, "_overlay", overlay or {})

    def __getattr__(self, name: str) -> Any:
        overlay = object.__getattribute__(self, "_overlay")
        if name in overlay:
            return overlay[name]
        return getattr(object.__getattribute__(self, "_server_args"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "ResolvedView is read-only; post-process passes return declarations"
        )


class ResolvingConfig:
    """Live read view of the resolution result: the declaration stash over the
    record's fields, looked up per read.

    ``ResolvedView`` snapshots the overlay when it is built, which is what a
    post-process pass wants -- it reads the state at its slot. A resolver that
    reads *after* declaring, or after calling something that declares, needs the
    current answer instead, so this one walks the stash on every read. It falls
    through to the field, which is where the raw input lives.
    """

    __slots__ = ("_server_args",)

    def __init__(self, server_args: Any):
        object.__setattr__(self, "_server_args", server_args)

    def __getattr__(self, name: str) -> Any:
        server_args = object.__getattribute__(self, "_server_args")
        for _source, declared in reversed(
            getattr(server_args, "_resolved_overrides", None) or ()
        ):
            if name in declared:
                return declared[name]
        return getattr(server_args, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "ResolvingConfig is read-only; resolution writes through declarations"
        )


def resolving_view(server_args: Any) -> ResolvingConfig:
    """A live read view of what resolution has decided so far."""
    return ResolvingConfig(server_args)


def _declaration_overlay(server_args: Any) -> Dict[str, Any]:
    """What the declarations say so far, last writer wins.

    Nothing writes the fields, so a mid-resolution reader needs this to see a
    decision at all; the fields keep what the caller supplied."""
    overlay: Dict[str, Any] = {}
    for _source, declared in getattr(server_args, "_resolved_overrides", None) or ():
        overlay.update(declared)
    return overlay


def resolved_view(server_args: Any) -> ResolvedView:
    """Read-only view of the resolving configuration: the declarations
    overlaid on the fields, snapshotted per call.

    For mid-resolution code that is not a pass (``__post_init__`` handlers and
    hooks) that must answer with what resolution decided -- a declaration-only resolver (a model-specific
    override, a registry entry) never writes the field, so a field read there
    answers with the raw input."""
    return ResolvedView(server_args, overlay=_declaration_overlay(server_args))


def attention_backends_of(cfg: Any) -> tuple:
    """(prefill, decode) attention backends of a config-shaped object (a
    ResolvedView mid-resolution, or pristine server_args at dispatch time):
    split fields fall back to the base backend."""
    prefill = (
        cfg.prefill_attention_backend
        if cfg.prefill_attention_backend
        else cfg.attention_backend
    )
    decode = (
        cfg.decode_attention_backend
        if cfg.decode_attention_backend
        else cfg.attention_backend
    )
    return prefill, decode


def _register_for(*architectures: str):
    """Register one provider for several architectures (family lists)."""

    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        for architecture in architectures:
            register_model_override(architecture)(fn)
        return fn

    return decorator


def record_of(view: Any) -> Any:
    """The record a view reads through.

    For the few helpers a view cannot serve: `get_default_attn_backend` reads
    through *both* overlays, so it needs the record the two views are built
    from rather than either one of them.
    """
    return object.__getattribute__(view, "_server_args")


def is_attention_backend_not_set(cfg: Any):
    """None of the three attention backends has been decided yet.

    Takes the view rather than the record: every read is a view read, and the
    callers that hold a view (the override providers) would otherwise have to
    reach back through it for a record.
    """
    return (
        cfg.attention_backend is None
        and cfg.prefill_attention_backend is None
        and cfg.decode_attention_backend is None
    )


def use_mla_backend(server_args: Any):
    from sglang.srt.configs.model_config import AttentionArch

    model_config = model_config_of(server_args)
    return model_config.attention_arch == AttentionArch.MLA


def model_config_of(server_args: Any):
    """The model configuration this record describes, built once and memoised.

    Takes a view as readily as the record: a view is a read overlay of one
    record, the memo has to live on that record either way, and the callers
    that hold a view would otherwise all have to unwrap it themselves.
    """
    if isinstance(server_args, (ResolvedView, ResolvingConfig)):
        server_args = record_of(server_args)
    # Lazy init to avoid circular import
    cfg = resolving_view(server_args)
    from sglang.srt.configs.model_config import ModelConfig

    memo = getattr(server_args, "_model_config", None)
    if memo is not None:
        # The key is the path this record carried when the cache was
        # filled. The GGUF and ModelScope handlers declare a different
        # `model_path`, and a configuration built before them describes
        # another checkpoint. `ModelConfig` re-points its own `model_path`
        # at the local pull directory when the weights sit behind an
        # object-store URI, so its field is not the key. A configuration a
        # fixture supplied carries no key and is handed back as it is.
        built_from = getattr(server_args, "_model_config_built_from", None)
        if built_from is None or built_from == cfg.model_path:
            return memo

    model_config = ModelConfig.from_server_args(server_args)
    server_args._model_config = model_config
    server_args._model_config_built_from = cfg.model_path
    if model_config.is_hybrid_swa:
        logger.info(
            "Hybrid SWA model detected. architectures=%s",
            model_config.hf_config.architectures,
        )
    return model_config


def mamba_extra_buffer_of(cfg: Any) -> bool:
    """Mid-resolution equivalent of runtime_context.mamba_extra_buffer_enabled:
    reads the (possibly overlaid) strategy from a config-shaped object.

    This is the one definition of the predicate: ``ServerArgs`` delegates its
    member to it, and the runtime_context accessor is its post-publish sibling
    (which cannot reuse it, because the two leaves land in different bags)."""
    return cfg.disable_radix_cache is False and cfg.mamba_radix_cache_strategy in (
        "extra_buffer",
        "extra_buffer_lazy",
    )


def get_default_attn_backend(server_args: Any, use_mla_backend: bool, model_config):
    """
    Auto select the fastest attention backend.

    1. Models with MHA Architecture (e.g: Llama, QWen)
        1.1 We will turn on FA3 on hopper unless user use spec decode with topk > 1 or page_size > 1.
        1.2 Use trtllm_mha for SM100/SM103 (Blackwell B200/GB200/B300) excluding spec with topk > 1.
           Note: trtllm_mha does not support SM120, which will fall back to flashinfer.
        1.3 In other cases, we will use flashinfer if available, otherwise use triton.
    2. Models with MLA Architecture and using FA3
        2.1 We will use FA3 backend on hopper.
        2.2 We will use Flashinfer backend on blackwell.
        2.3 Otherwise, we will use triton backend.
    """
    cfg = resolving_view(server_args)
    # OOT platforms provide their own default attention backend.
    if current_platform.is_out_of_tree():
        return current_platform.get_default_attention_backend()

    # Whisper requires flashinfer for cross-attention CUDA graph support.
    if "WhisperForConditionalGeneration" in (
        model_config.hf_config.architectures or []
    ):
        return "flashinfer"

    if not use_mla_backend:
        # MHA architecture

        if get_platform().is_hopper_with_cuda_12_3 and is_no_spec_infer_or_topk_one(
            resolved_view(server_args)
        ):
            # Note: flashinfer 0.6.1 caused performance regression on Hopper attention kernel
            # Before the kernel is fixed, we choose fa3 as the default backend on Hopper MHA
            # ref: https://github.com/sgl-project/sglang/issues/17411
            return "fa3"
        elif (
            get_platform().is_sm100
            and is_no_spec_infer_or_topk_one(resolved_view(server_args))
            and (
                cfg.speculative_algorithm is None
                or cfg.speculative_eagle_topk is not None
            )
        ):
            # trtllm_mha requires equal K/V row widths; fa4 carries
            # v_head_dim through.
            if model_config.has_asymmetric_kv:
                return "fa4"
            return "trtllm_mha"
        elif get_platform().is_hip:
            return "aiter"
        elif is_mps():
            return "torch_native"
        else:
            # FlashInfer does not support attention sinks.
            if get_platform().has_flashinfer and not model_config.has_attention_sinks:
                return "flashinfer"
            return "triton"
    else:
        # MLA architecture
        if get_platform().is_hopper_with_cuda_12_3:
            return "fa3"
        elif get_platform().is_sm100:
            return "flashinfer"
        elif get_platform().is_hip:
            head_num = model_config.get_num_kv_heads(cfg.tp_size)
            # TODO current aiter only support head number 16 or 128 head number
            if head_num == 128 or head_num == 16:
                return "aiter"
            else:
                return "triton"
        elif is_mps():
            return "torch_native"
        else:
            return "triton"


def _dspark_verify_on_decode_backend(
    backend: Optional[str], q_len: int, kv_cache_dtype: Optional[str]
) -> bool:
    """Whether the MLA decode backend can serve a q_len-wide target verify."""
    if backend == "trtllm_mla":
        return True
    if backend == "tokenspeed_mla":
        return kv_cache_dtype == "fp8_e4m3" and q_len <= 8
    if backend == "cutedsl_mla":
        # cute-dsl monolithic MLA decode folds the verify tokens into the head
        # dim (fold_sq), so it serves any DSPARK verify width. Needs flashinfer
        # >= 0.6.15 (older builds reject q_len >= 5).
        return True
    return False


def _is_mxfp4_pack_quantized(hf_config: Any) -> bool:
    qc = getattr(
        getattr(hf_config, "text_config", hf_config), "quantization_config", None
    )
    if not isinstance(qc, dict):
        return False
    groups = qc.get("config_groups") or {}
    return any(
        "mxfp4" in str(g.get("format", ""))
        for g in groups.values()
        if isinstance(g, dict)
    )
