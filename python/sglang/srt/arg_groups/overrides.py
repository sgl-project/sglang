# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Declarative model-override registry.

Model-identity adjustments to the server configuration are DECLARED here and
materialized onto ``server_args`` at the end of ``__post_init__`` (gate
order, last writer wins) — model code never mutates ``ServerArgs`` fields
imperatively.

Two declaration forms, keyed on ``hf_config.architectures[0]``:

- ``MODEL_OVERRIDES``: pure-constant cases — ``arch -> {field: value}``.
- ``@register_model_override(arch)``: derived cases — a callable
  ``fn(server_args, hf_config) -> dict`` that faithfully carries today's
  conditional logic. ``server_args`` is pristine and must be treated
  read-only: the callable returns declarations, it never writes.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.arg_groups.arg_utils import resolvable_fields
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.utils.common import (
    cpu_has_amx_support,
    get_device_capability,
    get_device_sm,
    get_nvidia_driver_version,
    get_quantization_config,
    is_blackwell_supported,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_musa,
    is_npu,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
    is_triton_kernels_available,
    is_xpu,
    xpu_has_xmx_support,
)

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
    the HF config go through ``server_args.get_model_config()`` (cached,
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


# Ordered post-process passes (the normalization stage). List order is the
# end-state execution order and mirrors today's handler call sequence in
# __post_init__; during the transition each pass is invoked from its legacy
# slot via run_post_process_pass, so ordering is preserved byte-for-byte.
POST_PROCESS_PASSES: List[Callable[..., dict]] = []


def register_post_process(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Register a post-process pass: ``fn(view) -> {field: resolved_value}``.

    The pass reads a :class:`ResolvedView` (post-model-override state) and
    must not mutate anything; validations may live in a pass (read + raise).
    """
    POST_PROCESS_PASSES.append(fn)
    return fn


def _declaration_overlay(server_args: Any) -> Dict[str, Any]:
    """Accumulated declared values: declarations never mutate
    ``server_args``, so mid-resolution readers overlay them from the
    declaration stash (last writer wins, like the gate)."""
    overlay: Dict[str, Any] = {}
    for _source, declared in getattr(server_args, "_resolved_overrides", None) or ():
        overlay.update(declared)
    return overlay


def run_post_process_pass(server_args: Any, fn: Callable[..., dict]) -> None:
    """Invoke one pass at its legacy handler slot.

    Evaluates the pass on the resolving state (a read-only view with the
    accumulated declarations overlaid from the stash) and appends its
    declaration to the stash. During ``__post_init__`` the fields stay
    untouched — ``materialize_declarations`` applies the whole stash once at
    the end of resolution; a pass invoked after materialization (a post-init
    slot) writes through immediately.
    """
    declared = fn(ResolvedView(server_args, overlay=_declaration_overlay(server_args)))
    if not isinstance(declared, dict):
        raise TypeError(
            f"post-process pass {fn.__qualname__} must return a dict, "
            f"got {type(declared).__name__}"
        )
    if declared:
        entry = (fn.__qualname__, dict(declared))
        stash = getattr(server_args, "_resolved_overrides", None)
        if stash is None:
            # Handlers hosting pass slots may be invoked directly on fixtures
            # that never ran the monolith dispatch (which owns the stash);
            # create it lazily. Real publishes always pass through the
            # dispatch first — the dispatch ASSIGNS the stash, so pass slots
            # must sit at or after it in __post_init__ order.
            stash = server_args._resolved_overrides = []
        stash.append(entry)
        validate_declarations(server_args, [entry])
        if getattr(server_args, "_declarations_materialized", False):
            _apply_fields(server_args, declared)


def _apply_fields(server_args: Any, fields: Dict[str, Any]) -> None:
    """Write fields on behalf of the pipeline (bypasses the strict bare-
    assignment guard that protects post-resolution mutation)."""
    object.__setattr__(server_args, "_in_override", True)
    try:
        for field, value in fields.items():
            setattr(server_args, field, value)
    finally:
        object.__setattr__(server_args, "_in_override", False)


def materialize_declarations(server_args: Any) -> None:
    """Apply the accumulated declarations onto ``server_args`` once, at the
    end of ``__post_init__`` (gate order: last writer wins). After this the
    fields carry the resolved configuration — every post-init reader, in any
    process, reads them directly; ``resolved_view`` remains an internal
    helper for mid-resolution code only."""
    for _source, declared in getattr(server_args, "_resolved_overrides", None) or ():
        for field, value in declared.items():
            setattr(server_args, field, value)
    server_args._declarations_materialized = True


def resolved_view(server_args: Any) -> ResolvedView:
    """Read-only view of the resolving configuration for mid-resolution code
    that is not a pass (``__post_init__`` handlers and hooks). Internal to
    the resolution pipeline: after ``materialize_declarations`` runs, the
    fields themselves carry the resolved values — read them directly."""
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


def mamba_extra_buffer_of(cfg: Any) -> bool:
    """Mid-resolution equivalent of runtime_context.mamba_extra_buffer_enabled:
    reads the (possibly overlaid) strategy from a config-shaped object."""
    return cfg.disable_radix_cache is False and cfg.mamba_radix_cache_strategy in (
        "extra_buffer",
        "extra_buffer_lazy",
    )


def declare_load_time_override(source: str, declared: Dict[str, Any]) -> None:
    """Declare a load-time resolved field (model-file config overrides,
    weight-resolved dtypes) on the published ``server_args``: resolution has
    already materialized, so the declaration writes through, joining the
    declaration stash for provenance and republish consistency."""
    from sglang.srt.runtime_context import get_context

    server_args = get_context().server_args
    validate_declarations(server_args, [(source, dict(declared))])
    override = getattr(server_args, "override", None)
    if override is not None:
        override(source, **declared)
    else:
        # Config-shaped fixtures without the mutation entry point.
        _apply_fields(server_args, declared)


def collect_model_override_declarations(
    architecture: str, server_args: Any, hf_config: Any
) -> List[Tuple[str, Dict[str, Any]]]:
    """Collect ``(source, declaration)`` pairs for one architecture.

    Application order (last writer wins downstream in the gate): the constant
    ``MODEL_OVERRIDES`` entry first, then exact-keyed callables in
    registration order, then matching predicate-keyed callables in
    registration order. Empty declarations are dropped.
    """
    declarations: List[Tuple[str, Dict[str, Any]]] = []
    const = MODEL_OVERRIDES.get(architecture)
    if const:
        declarations.append((f"MODEL_OVERRIDES[{architecture!r}]", dict(const)))
    for fn in _MODEL_OVERRIDE_FNS.get(architecture, ()):
        declared = _invoke_provider(fn, server_args, hf_config)
        if declared:
            declarations.append((fn.__qualname__, dict(declared)))
    for predicate, fn in _PREDICATE_OVERRIDE_FNS:
        if predicate(architecture):
            declared = _invoke_provider(fn, server_args, hf_config)
            if declared:
                declarations.append((fn.__qualname__, dict(declared)))
    return declarations


# ---------------------------------------------------------------------------
# Derived per-family declarations (faithful ports of legacy arch branches).
# Callables read the PRISTINE server_args, never write; logging is kept
# verbatim from the legacy branch for operator-visible fidelity.
# ---------------------------------------------------------------------------


def _register_for(*architectures: str):
    """Register one provider for several architectures (family lists)."""

    def decorator(fn: Callable[..., dict]) -> Callable[..., dict]:
        for architecture in architectures:
            register_model_override(architecture)(fn)
        return fn

    return decorator


@_register_for(
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "KimiK25ForConditionalGeneration",
    "MistralLarge3ForCausalLM",
    "PixtralForConditionalGeneration",
    "GlmMoeDsaForCausalLM",
    "LongcatFlashForCausalLM",
    "LongcatFlashForCausalLMNextN",
)
def _deepseek_family_overrides(server_args: Any, hf_config: Any) -> dict:
    """Order-safe declarations of the DeepSeek/DSA branch. The CP parallel
    writes (enable_dp_attention/ep_size/moe_a2a_backend have post-monolith
    writers), the kv-cache/split-backend defaults, the quant/moe block (read
    before it by _set_default_dsa_kv_cache_dtype) and the env writes stay in
    the branch."""
    from sglang.srt.configs.model_config import is_deepseek_dsa

    overrides: Dict[str, Any] = {}
    if is_deepseek_dsa(hf_config):  # DeepSeek 3.2/GLM 5
        # Set attention backend for DeepSeek
        if server_args.is_attention_backend_not_set():
            overrides["attention_backend"] = "dsa"
            logger.info("Use dsa attention backend for DeepSeek with DSA.")
        if not is_npu() and not is_xpu():  # CUDA or ROCm GPU
            if server_args.enable_prefill_cp:
                logger.warning(
                    "Context parallel feature is still under experiment. It has only been verified on Hopper platform."
                )
                overrides["enable_dp_attention"] = True
                overrides["moe_dense_tp_size"] = 1
                if server_args.cp_strategy == "zigzag":
                    overrides["moe_a2a_backend"] = "deepep"
                    overrides["ep_size"] = server_args.tp_size
                    logger.warning(
                        "zigzag DSA CP requires moe_dense_tp_size=1, "
                        "moe_a2a_backend=deepep, ep_size=tp_size, batch_size=1."
                    )
                else:
                    assert (
                        server_args.dp_size == 1
                    ), "interleave DSA CP does not support DP attention."
                assert (
                    server_args.tp_size <= 8
                ), "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
                # Note(kpham-sgl): Keep attn_tp_size == 1 under DSA CP.
                # DSACPLayerCommunicator does not all-reduce attention-TP
                # partial o_proj outputs before replicated dense FFNs.
                attn_cp_size = server_args.tp_size // server_args.dp_size
                overrides["attn_cp_size"] = attn_cp_size
                logger.warning(
                    "Enabled DSA context parallel: "
                    f"strategy={server_args.cp_strategy}, dp_size={server_args.dp_size}, "
                    f"moe_dense_tp_size={overrides['moe_dense_tp_size']}, "
                    f"ep_size={overrides.get('ep_size', server_args.ep_size)}, tp_size={server_args.tp_size}, "
                    f"attn_cp_size={attn_cp_size}, "
                    f"kv_cache_dtype={server_args.kv_cache_dtype}, "
                    f"moe_a2a_backend={overrides.get('moe_a2a_backend', server_args.moe_a2a_backend)}, "
                    f"cuda_graph_config[prefill].backend=disabled"
                )

            # Deferred import to avoid a circular import at module-load
            # time (dsa.utils imports the runtime-context accessors).
            from sglang.srt.layers.attention.dsa.utils import (
                aiter_can_use_preshuffle_paged_mqa,
            )

            if is_hip() and not aiter_can_use_preshuffle_paged_mqa():
                # Legacy ROCm DSA path: aiter's gluon paged-MQA kernel is
                # unavailable (Triton<3.5 and AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS
                # not set, or SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1 / SGLANG_USE_AITER=0).
                overrides["page_size"] = 1
                logger.warning(
                    "Setting page size to 1 for DeepSeek DSA on ROCm "
                    "(aiter preshuffle paged-MQA path unavailable: "
                    "needs Triton>=3.5.0 or AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1)."
                )
            else:
                overrides["page_size"] = 64
                logger.warning("Setting page size to 64 for DeepSeek DSA.")
    else:
        # DeepSeek V3/R1/V3.1
        if is_sm100_supported():
            if (
                server_args.attention_backend is None
                and server_args.prefill_attention_backend is None
                and server_args.decode_attention_backend is None
            ):
                overrides["attention_backend"] = "trtllm_mla"
                logger.info(
                    "Use trtllm_mla as attention backend on sm100 for DeepseekV3ForCausalLM"
                )
        # MLA prefill CP auto-config. Mirrors the NSA CP block above
        # (minus the in-seq/round-robin mode split, which MLA CP does not support)
        if server_args.enable_prefill_cp and server_args.use_mla_backend():
            logger.warning(
                "MLA prefill context parallel is still experimental. "
                "Verified on Hopper with the fa3 backend."
            )
            overrides["enable_dp_attention"] = True
            # TODO(kpham-sgl) Supports moe_dense_tp_size != 1.
            overrides["moe_dense_tp_size"] = 1
            overrides["moe_a2a_backend"] = "deepep"
            overrides["ep_size"] = server_args.tp_size
            logger.warning(
                "For MLA CP, we have the following restrictions: moe_dense_tp_size == 1, moe_a2a_backend == deepep, ep_size == tp_size, batch_size == 1"
            )
            # FIXME(kpham-sgl): Keep attn_tp_size == 1 under MLA CP.
            # DSACPLayerCommunicator does not all-reduce attention-TP
            # partial o_proj outputs before replicated dense FFNs.
            attn_cp_size = server_args.tp_size // server_args.dp_size
            overrides["attn_cp_size"] = attn_cp_size
            logger.warning(
                f"Enable Context Parallel opt for MLA, "
                f"Setting dp_size == {server_args.dp_size} and "
                f"attn_cp_size == {attn_cp_size}, "
                f"moe_dense_tp_size == {overrides['moe_dense_tp_size']}, "
                f"ep_size == {overrides['ep_size']}, "
                f"tp_size == {server_args.tp_size}, "
                f"moe_a2a_backend {overrides['moe_a2a_backend']}, "
                f"cuda_graph_config[prefill].backend=disabled"
            )
    return overrides


# Keep in sync with MIMO_V2_MODEL_ARCHS (server_args.py / configs/hf_config.py).
@_register_for("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM")
def _mimo_v2_overrides(server_args: Any, hf_config: Any) -> dict:
    if server_args.speculative_algorithm == "EAGLE":
        logger.info("Enable multi-layer EAGLE speculative decoding for MiMoV2 model.")
        return {"enable_multi_layer_eagle": True}
    return {}


@_register_for("MiniMaxM2ForCausalLM")
def _minimax_m2_overrides(server_args: Any, hf_config: Any) -> dict:
    logger.info(
        "Enable TF32 matmul for MiniMaxM2ForCausalLM model to improve gate gemm performance."
    )
    return {"enable_tf32_matmul": True}


@_register_for(
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
)
def _gemma2_gemma3_overrides(server_args: Any, hf_config: Any) -> dict:
    # FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with gemma2 model.
    # It failed at this test: https://github.com/sgl-project/sglang/actions/runs/16255155597/job/45890331952#step:4:736
    logger.warning(
        f"Disable hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
    )
    return {"disable_hybrid_swa_memory": True}


@_register_for("Exaone4ForCausalLM", "ExaoneMoEForCausalLM")
def _exaone_overrides(server_args: Any, hf_config: Any) -> dict:
    if hf_config.sliding_window_pattern is not None:
        logger.warning(
            f"Disabling hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
        )
        return {"disable_hybrid_swa_memory": True}
    return {}


@_register_for("GptOssForCausalLM")
def _gpt_oss_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    # Set attention backend for GPT-OSS
    if server_args.is_attention_backend_not_set():
        if is_sm100_supported():
            overrides["attention_backend"] = "trtllm_mha"
        elif is_sm90_supported():
            overrides["attention_backend"] = "fa3"
        elif is_cpu() and cpu_has_amx_support():
            overrides["attention_backend"] = "intel_amx"
        elif is_xpu():
            overrides["attention_backend"] = "intel_xpu"
        elif is_hip():
            overrides["attention_backend"] = "aiter"
        else:
            overrides["attention_backend"] = "triton"
    if is_xpu():
        # Check for bf16 dtype on Intel XPU. Reads the pristine dtype request,
        # which equals the legacy mid-branch read: dtype had no earlier writer
        # for this arch.
        if server_args.dtype == "auto":
            logger.warning(
                "GptOssForCausalLM on Intel XPU currently supports bfloat16 dtype only"
            )
        elif server_args.dtype not in ["bfloat16"]:
            raise NotImplementedError(
                f"GptOssForCausalLM on Intel XPU only supports bfloat16 dtype, "
                f"but got '{server_args.dtype}'. Please use --dtype bfloat16 or remove --dtype to use auto."
            )
    quantization_config = getattr(hf_config, "quantization_config", None)
    is_mxfp4_quant_format = (
        quantization_config is not None
        and quantization_config.get("quant_method") == "mxfp4"
    )
    if is_mxfp4_quant_format:
        # use bf16 for mxfp4 triton kernels
        overrides["dtype"] = "bfloat16"
    if server_args.moe_runner_backend == "auto":
        from sglang.srt.environ import envs

        if is_sm100_supported() and is_mxfp4_quant_format:
            overrides["moe_runner_backend"] = "flashinfer_mxfp4"
            logger.warning(
                "Detected SM100 and MXFP4 quantization format for GPT-OSS model, enabling FlashInfer MXFP4 MOE kernel."
            )
        elif is_sm120_supported() and is_mxfp4_quant_format:
            # trtllm-gen only supports SM100
            overrides["moe_runner_backend"] = "marlin"
            logger.warning(
                "Detected SM120 and MXFP4 quantization format for GPT-OSS model, enabling Marlin MOE kernel."
            )
        elif (is_hip() and envs.SGLANG_USE_AITER.get()) and is_mxfp4_quant_format:
            overrides["moe_runner_backend"] = "auto"
            logger.warning(
                "Detected ROCm and MXFP4 quantization format for GPT-OSS model, enabling aiter MXFP4 MOE kernel."
            )
            ## The AITER MXFP4 fused-MoE path for GPT-OSS expects the
            ## SEPARATED gate/up tile layout (matches the
            ## `gptoss_fp4_tuned_fmoe.csv` flydsl entries and the
            ## Mxfp4MoEMethod weight shuffle). Other AITER MXFP4
            ## callers default to INTERLEAVE; opt this path out
            ## unless the user explicitly overrode it.
            # envs.SGLANG_USE_AITER_MOE_GU_ITLV.set(False)
        elif is_hip() and envs.SGLANG_USE_AITER.get():
            # For GPT-OSS bf16 on ROCm with aiter, use triton backend
            # because aiter CK kernel doesn't support all GEMM dimensions
            overrides["moe_runner_backend"] = "triton"
            logger.warning(
                "Detected ROCm with SGLANG_USE_AITER for GPT-OSS bf16 model, using triton MOE kernel."
            )
        elif is_musa() and envs.SGLANG_DEEPEP_BF16_DISPATCH.get():
            overrides["moe_runner_backend"] = "deep_gemm"
            logger.warning(
                "Detected MUSA with SGLANG_DEEPEP_BF16_DISPATCH for bf16 model, using deep_gemm kernel."
            )
        elif (
            server_args.ep_size == 1
            and is_triton_kernels_available()
            and server_args.quantization is None
            and not (is_cpu() and cpu_has_amx_support())
        ):
            # The triton_kernels package segfaults on Blackwell (B200)
            # with NVIDIA driver >= 595. Fall back to triton backend.
            if is_blackwell_supported() and get_nvidia_driver_version() >= (595,):
                overrides["moe_runner_backend"] = "triton"
                logger.warning(
                    "Detected GPT-OSS model on Blackwell with driver >= 595, "
                    "using triton MOE kernel to avoid triton_kernels SIGSEGV."
                )
            else:
                overrides["moe_runner_backend"] = "triton_kernel"
                logger.warning(
                    "Detected GPT-OSS model, enabling triton_kernels MOE kernel."
                )
    return overrides


# Keep in sync with LLAMA4_MODEL_ARCHS (server_args.py).
@_register_for("Llama4ForConditionalGeneration", "Llama4ForCausalLM")
def _llama4_overrides(server_args: Any, hf_config: Any) -> dict:
    if server_args.device == "cpu":
        return {}
    overrides: Dict[str, Any] = {}
    # Auto-select attention backend for Llama4 if not specified
    if server_args.attention_backend is None:
        if is_sm100_supported():
            backend, platform = "trtllm_mha", "sm100"
        elif is_sm90_supported():
            backend, platform = "fa3", "sm90"
        elif is_hip():
            backend, platform = "aiter", "hip"
        elif server_args.device == "xpu":
            backend, platform = "intel_xpu", "xpu"
        else:
            backend, platform = "triton", "other platforms"
        logger.warning(
            f"Use {backend} as attention backend on {platform} for Llama4 model"
        )
        overrides["attention_backend"] = backend
    if is_sm100_supported() and server_args.moe_runner_backend == "auto":
        if server_args.quantization in {"fp8", "modelopt_fp8"}:
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on SM100 for Llama4"
            )
    return overrides


@_register_for(
    "Gemma4ForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4UnifiedForConditionalGeneration",
)
def _gemma4_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    default_attention_backend = "trtllm_mha" if is_sm100_supported() else "triton"
    if server_args.is_attention_backend_not_set():
        logger.info(
            f"Use {default_attention_backend} as default attention backend for Gemma4"
        )
        overrides["attention_backend"] = default_attention_backend
    # If only one split backend is set, keep the other side on a
    # Gemma4-compatible fallback instead of letting generic backend selection
    # choose an unsupported backend later.
    elif server_args.attention_backend is None:
        overrides["attention_backend"] = default_attention_backend
    if is_sm100_supported() and server_args.moe_runner_backend == "auto":
        if server_args.get_model_config().quantization == "modelopt_fp4":
            overrides["quantization"] = "modelopt_fp4"
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on "
                "SM100 for Gemma-4 (modelopt_fp4)"
            )
    return overrides


@_register_for("MossVLForConditionalGeneration")
def _moss_vl_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    if server_args.is_attention_backend_not_set():
        overrides["prefill_attention_backend"] = "flashinfer"
        logger.info("Use flashinfer as default prefill attention backend for Moss-VL")
    prefill_backend = (
        overrides.get("prefill_attention_backend")
        or server_args.get_attention_backends()[0]
    )
    assert prefill_backend == "flashinfer", (
        "MossVLForConditionalGeneration requires flashinfer prefill "
        "attention backend for cross-attention custom mask support."
    )
    return overrides


@_register_for("MiniCPMV4_6ForConditionalGeneration")
def _minicpm_v4_6_overrides(server_args: Any, hf_config: Any) -> dict:
    if is_sm100_supported() and server_args.attention_backend is None:
        return {"attention_backend": "triton"}
    return {}


@_register_for(
    "FalconH1ForCausalLM", "JetNemotronForCausalLM", "JetVLMForConditionalGeneration"
)
def _falcon_h1_jet_overrides(server_args: Any, hf_config: Any) -> dict:
    if is_sm100_supported() and server_args.attention_backend is None:
        return {"attention_backend": "triton"}
    return {}


@_register_for("GraniteMoeHybridForCausalLM")
def _granite_moe_hybrid_overrides(server_args: Any, hf_config: Any) -> dict:
    has_mamba = any(
        layer_type == "mamba" for layer_type in getattr(hf_config, "layer_types", [])
    )
    if has_mamba and is_sm100_supported() and server_args.attention_backend is None:
        return {"attention_backend": "flashinfer"}
    return {}


@_register_for("Lfm2ForCausalLM")
def _lfm2_overrides(server_args: Any, hf_config: Any) -> dict:
    if is_sm100_supported() and server_args.attention_backend is None:
        return {"attention_backend": "flashinfer"}
    return {}


@_register_for("DeepseekV4ForCausalLM")
def _deepseek_v4_overrides(server_args: Any, hf_config: Any) -> dict:
    """DeepSeek V4 attention/page/window/MoE-runner defaults (from
    arg_groups/deepseek_v4_hook.py). The kv-cache dtype and NPU split-backend
    writes, the max_running_requests fill and the validations stay in the
    hook at its legacy slot."""
    from sglang.srt.environ import envs
    from sglang.srt.server_args import ServerArgs

    model_arch = hf_config.architectures[0]
    overrides: Dict[str, Any] = {"attention_backend": "dsv4"}

    page_size = 256
    if server_args.device == "npu":
        # NPU keeps the device-aware "dsv4" backend (the registry routes it to
        # the Ascend V4 subclass); only the pool geometry / dtype differ.
        # set_default_server_args() pins all three backends to "ascend" for
        # generic NPU models; override that here so V4 stays consistently on
        # dsv4.
        page_size = 128
        overrides["prefill_attention_backend"] = "dsv4"
        overrides["decode_attention_backend"] = "dsv4"
    overrides["page_size"] = page_size
    logger.info(
        f"Use dsv4 attention backend for {model_arch}, setting page_size to {page_size}."
    )

    if server_args.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio:
        overrides["swa_full_tokens_ratio"] = 0.1
        logger.info(f"Setting swa_full_tokens_ratio to 0.1 for {model_arch}.")

    if server_args.moe_runner_backend == "auto":
        model_config = server_args.get_model_config()
        # nvidia/DeepSeek-V4-Pro-NVFP4 uses flashinfer_trtllm_routed MoE runner backend.
        if model_config.nvfp4_moe_meta is not None:
            overrides["moe_runner_backend"] = "flashinfer_trtllm_routed"
            logger.info(
                "Use flashinfer_trtllm_routed as MoE runner backend for "
                f"{model_arch} hybrid FP8+NVFP4 checkpoint."
            )
        elif model_config.is_fp4_experts and not envs.SGLANG_DSV4_FP4_DEQUANT.get():
            if is_sm100_supported():
                overrides["moe_runner_backend"] = "flashinfer_mxfp4"
                logger.info(
                    "Use flashinfer_mxfp4 as MoE runner backend for "
                    f"{model_arch} packed-FP4 checkpoint on SM100."
                )
            elif is_sm90_supported() or is_sm120_supported():
                overrides["moe_runner_backend"] = "marlin"
                logger.info(
                    "Use marlin as MoE runner backend for "
                    f"{model_arch} packed-FP4 checkpoint on SM90/SM120."
                )
    return overrides


@_register_for("NemotronHForCausalLM", "NemotronHPuzzleForCausalLM")
def _nemotron_h_overrides(server_args: Any, hf_config: Any) -> dict:
    """NemotronH quantization / MoE runner / attention backend defaults
    (absorbed from the retired arg_groups/nemotron_h_hook.py; the mamba radix
    cache handling and the triton-backend assert stay in the arch branch)."""
    model_arch = hf_config.architectures[0]
    model_config = server_args.get_model_config()
    overrides: Dict[str, Any] = {}

    is_modelopt = model_config.quantization in [
        "modelopt",
        "modelopt_fp8",
        "modelopt_fp4",
        "modelopt_mixed",
    ]
    quantization = server_args.quantization
    if is_modelopt:
        assert model_config.hf_config.mlp_hidden_act == "relu2"
        if model_config.quantization == "modelopt":
            quant_algo = model_config.hf_config.quantization_config["quant_algo"]
            if quant_algo == "MIXED_PRECISION":
                quantization = "modelopt_mixed"
            else:
                quantization = (
                    "modelopt_fp4" if quant_algo == "NVFP4" else "modelopt_fp8"
                )
        else:
            quantization = model_config.quantization
        overrides["quantization"] = quantization

    if (is_modelopt or model_config.quantization is None) and (
        server_args.moe_runner_backend == "auto"
    ):
        if is_sm100_supported() and server_args.moe_a2a_backend == "none":
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                f"Use flashinfer_trtllm as MoE runner backend on sm100 for {model_arch}"
            )
        elif (
            (
                model_config.quantization in ("modelopt_fp4", "modelopt_mixed")
                or quantization == "modelopt_fp4"
            )
            and is_cuda()
            and (8, 0) <= get_device_capability() < (10, 0)
        ):
            overrides["moe_runner_backend"] = "marlin"
            logger.info(
                "Use marlin as MoE runner backend on SM80-SM90 for "
                f"{model_arch} {model_config.quantization}"
            )
        else:
            overrides["moe_runner_backend"] = "flashinfer_cutlass"

    if is_sm100_supported() and server_args.attention_backend is None:
        overrides["attention_backend"] = "flashinfer"
    return overrides


@_register_for(
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "InternS2PreviewForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
)
def _qwen3_5_hybrid_overrides(server_args: Any, hf_config: Any) -> dict:
    if not is_sm100_supported() or server_args.attention_backend is not None:
        return {}
    sm100_default_attn_backend = "triton"
    # trtllm_mha requires speculative_eagle_topk == 1 and page_size > 1.
    # _get_default_attn_backend handles the eagle_topk check.
    # There is only one case where page_size=1 is required,
    # which is when radix cache is enabled and both extra_buffer
    # and spec decoding are disabled.
    default_attn_backend = server_args._get_default_attn_backend(
        use_mla_backend=server_args.use_mla_backend(),
        model_config=server_args.get_model_config(),
    )
    # The mamba radix-cache pass runs before this dispatch: read the
    # declared strategy through the view (the legacy branch observed the
    # already-written field here).
    if default_attn_backend == "trtllm_mha" and not (
        not mamba_extra_buffer_of(resolved_view(server_args))
        and not server_args.disable_radix_cache
        and server_args.speculative_algorithm is None
    ):
        sm100_default_attn_backend = "trtllm_mha"
    return {
        "attention_backend": sm100_default_attn_backend,
        "page_size": 64 if sm100_default_attn_backend == "trtllm_mha" else 1,
    }


@_register_for("Qwen3VLForConditionalGeneration")
def _qwen3vl_overrides(server_args: Any, hf_config: Any) -> dict:
    from sglang.srt.environ import envs

    if (
        is_hip()
        and envs.SGLANG_USE_AITER_UNIFIED_ATTN.get()
        and server_args.page_size is None
    ):
        logger.info(
            "Setting page_size=16 for aiter unified attention on Qwen3VLForConditionalGeneration."
        )
        return {"page_size": 16}
    return {}


@_register_for(
    "Qwen3MoeForCausalLM",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3NextForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "InternS2PreviewForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
)
def _qwen3_moe_family_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    if is_sm100_supported():
        quant_method = get_quantization_config(hf_config)
        quantization = server_args.quantization
        if (
            quantization is None
            and not server_args._quantization_explicitly_unset
            and quant_method is not None
        ):
            overrides["quantization"] = quant_method
            quantization = quant_method
        if (
            (quantization in ("fp8", "modelopt_fp4") or quantization is None)
            and server_args.moe_a2a_backend == "none"
            and server_args.moe_runner_backend == "auto"
        ):
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on sm100 for "
                f"{hf_config.architectures[0]}"
            )
    return overrides


@_register_for("Glm4MoeForCausalLM")
def _glm4_moe_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    if is_sm100_supported():
        quantization_config = getattr(hf_config, "quantization_config", None)
        quant_method = (
            quantization_config.get("quant_method")
            if quantization_config is not None
            else None
        )
        quantization = server_args.quantization
        if (
            quantization is None
            and not server_args._quantization_explicitly_unset
            and quant_method is not None
        ):
            overrides["quantization"] = quant_method
            quantization = quant_method
        if (
            quantization in {"modelopt_fp4", None}
            and server_args.moe_a2a_backend == "none"
            and server_args.moe_runner_backend == "auto"
        ):
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on sm100 for Glm4MoeForCausalLM"
            )
    logger.info(
        "Enable TF32 matmul for Glm4MoeForCausalLM model to improve gate gemm performance."
    )
    overrides["enable_tf32_matmul"] = True
    return overrides


@_register_for("Olmo2ForCausalLM")
def _olmo2_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    # FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with Olmo3 model.
    logger.warning(
        f"Disabling hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
    )
    overrides["disable_hybrid_swa_memory"] = True
    if server_args.attention_backend is None:
        if is_cuda() and is_sm100_supported():
            overrides["attention_backend"] = "trtllm_mha"
        elif is_cuda() and get_device_sm() >= 80:
            overrides["attention_backend"] = "fa3"
        else:
            overrides["attention_backend"] = "triton"
    return overrides


@register_model_override_predicate(
    lambda arch: "Step3p5ForCausalLM" in arch
    or "Step3p7ForConditionalGeneration" in arch
)
def _step3p_overrides(server_args: Any, hf_config: Any) -> dict:
    overrides: Dict[str, Any] = {}
    if server_args.is_attention_backend_not_set():
        if is_blackwell_supported():
            logger.info("Auto-select fa4 attention backend for Step3p7 on Blackwell.")
            overrides["attention_backend"] = "fa4"
        elif is_sm90_supported():
            logger.info("Auto-select fa3 attention backend for Step3p7 on Hopper.")
            overrides["attention_backend"] = "fa3"
    if server_args.speculative_algorithm == "EAGLE":
        logger.info(
            "Enable multi-layer EAGLE speculative decoding for Step3p5ForCausalLM model."
        )
        overrides["enable_multi_layer_eagle"] = True
    if server_args.enable_hierarchical_cache:
        logger.warning(
            "Reset swa_full_tokens_ratio to 1.0 for Step3p5ForCausalLM model with hierarchical cache"
        )
        overrides["swa_full_tokens_ratio"] = 1.0
        logger.warning(
            "Disable hybrid SWA memory for Step3p5ForCausalLM model with hierarchical cache"
        )
        overrides["disable_hybrid_swa_memory"] = True
    return overrides


# ---------------------------------------------------------------------------
# Post-process passes (normalization stage), in end-state execution order.
# Faithful ports of the legacy __post_init__ handlers; each is invoked from
# its legacy slot via run_post_process_pass during the transition.
# ---------------------------------------------------------------------------


# Architectures whose monolith branch routes through the mamba radix cache
# handling (hybrid linear-attention models). Keep in sync with the branch
# guards in _handle_model_specific_adjustments.
_MAMBA_RADIX_CACHE_ARCHS = frozenset(
    {
        "KimiLinearForCausalLM",
        "BailingMoeV2_5ForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "MiniCPMV4_6ForConditionalGeneration",
        "NemotronHForCausalLM",
        "NemotronHPuzzleForCausalLM",
        "FalconH1ForCausalLM",
        "JetNemotronForCausalLM",
        "JetVLMForConditionalGeneration",
        "Lfm2ForCausalLM",
        "ZayaForCausalLM",
    }
)

# Architectures that support the extra_buffer mamba radix cache strategy.
# Single source of truth: ServerArgs._support_mamba_cache_extra_buffer
# delegates here.
_MAMBA_EXTRA_BUFFER_ARCHS = frozenset(
    {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3NextForCausalLM",
        "InternS2PreviewForConditionalGeneration",
        "MiniCPMV4_6ForConditionalGeneration",
        "BailingMoeV2_5ForCausalLM",
        "FalconH1ForCausalLM",
        "GraniteMoeHybridForCausalLM",
        "NemotronHForCausalLM",
        "NemotronHPuzzleForCausalLM",
    }
)


def supports_mamba_cache_extra_buffer(view: Any, model_arch: str) -> bool:
    """Whether ``model_arch`` supports the extra_buffer strategy on the
    configured linear-attention backend (pure read)."""
    if model_arch in _MAMBA_EXTRA_BUFFER_ARCHS:
        return view.linear_attn_backend == "triton"
    return False


@register_post_process
def _mamba_radix_cache_resolution(view: Any) -> dict:
    """Resolve the hybrid-mamba radix cache fields (pure).

    Slot pass: invoked at each legacy ``_handle_mamba_radix_cache`` slot —
    the hybrid-spec call at the head of the monolith and the per-arch branch
    calls — where it reads the mid-resolution ``page_size`` /
    ``disable_overlap_schedule`` exactly as the legacy helper did. The arch
    guard replicates the union of the legacy call-site guards so the pass is
    self-sufficient in the end-state pass list.
    """
    from sglang.srt.configs.linear_attn_model_registry import (
        get_linear_attn_spec_by_arch,
    )

    hf_config = view.get_model_config().hf_config
    model_arch = hf_config.architectures[0]

    in_branch = model_arch in _MAMBA_RADIX_CACHE_ARCHS
    if model_arch == "GraniteMoeHybridForCausalLM":
        in_branch = any(
            layer_type == "mamba"
            for layer_type in getattr(hf_config, "layer_types", [])
        )
    spec = get_linear_attn_spec_by_arch(model_arch)
    if not ((spec is not None and spec.uses_mamba_radix_cache) or in_branch):
        return {}

    if view.disable_radix_cache:
        return {}

    declared: Dict[str, Any] = {"uses_mamba_radix_cache": True}
    if view.mamba_radix_cache_strategy == "auto":
        wants_overlap = not view.disable_overlap_schedule
        wants_paging = view.page_size is not None and view.page_size > 1
        if (wants_overlap or wants_paging) and supports_mamba_cache_extra_buffer(
            view, model_arch
        ):
            declared["mamba_radix_cache_strategy"] = "extra_buffer"
        else:
            declared["mamba_radix_cache_strategy"] = "no_buffer"
            declared["disable_overlap_schedule"] = True
    return declared


@register_post_process
def _dsa_kv_cache_dtype_default(view: Any) -> dict:
    """Slot pass in the DSA arm, ordered before the split-backend
    resolution: default the kv-cache dtype from the device capability
    (Blackwell FP8, Hopper bf16) and normalize the bf16 alias. Reads the
    PRISTINE dsa split backends (their resolution runs after this pass)."""
    from sglang.srt.configs.model_config import is_deepseek_dsa

    hf_config = view.get_model_config().hf_config
    if hf_config.architectures[0] not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    if not is_deepseek_dsa(hf_config):
        return {}
    if is_npu() or is_xpu():
        return {}

    import torch

    major, _ = torch.cuda.get_device_capability()

    # If user specified a backend but didn't explicitly set kv_cache_dtype,
    # suggest them to be explicit about kv_cache_dtype to avoid surprises
    if (
        view.dsa_prefill_backend is not None or view.dsa_decode_backend is not None
    ) and view.kv_cache_dtype == "auto":
        logger.warning(
            "When specifying --dsa-prefill-backend or --dsa-decode-backend, "
            "you should also explicitly set --kv-cache-dtype (e.g., 'fp8_e4m3' or 'bfloat16'). "
            "DeepSeek V3.2 defaults to FP8 KV cache which may not be compatible with all backends."
        )

    kv_cache_dtype = view.kv_cache_dtype
    if kv_cache_dtype == "auto":
        kv_cache_dtype = "fp8_e4m3" if major >= 10 else "bfloat16"
        logger.warning(
            f"Setting KV cache dtype to {kv_cache_dtype} for DeepSeek DSA on SM{major} device."
        )
    if kv_cache_dtype == "bf16":
        kv_cache_dtype = "bfloat16"
    assert kv_cache_dtype in [
        "bfloat16",
        "fp8_e4m3",
    ], "DeepSeek DSA only supports bf16/bfloat16 or fp8_e4m3 kv_cache_dtype"
    if kv_cache_dtype != view.kv_cache_dtype:
        return {"kv_cache_dtype": kv_cache_dtype}
    return {}


@register_post_process
def _dsa_split_backend_resolution(view: Any) -> dict:
    """Slot pass in the DSA arm: default the DSA prefill/decode split
    backends from the mid-resolution kv-cache dtype and the device
    capability. The hisparse arm takes precedence under --enable-hisparse."""
    from sglang.srt.configs.model_config import is_deepseek_dsa

    hf_config = view.get_model_config().hf_config
    if hf_config.architectures[0] not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    if not is_deepseek_dsa(hf_config):
        return {}
    if is_npu() or is_xpu():
        return {}

    import torch

    major, _ = torch.cuda.get_device_capability()
    kv_cache_dtype = view.kv_cache_dtype
    user_set_prefill = view.dsa_prefill_backend is not None
    user_set_decode = view.dsa_decode_backend is not None
    declared: Dict[str, Any] = {}

    if view.enable_hisparse:
        from sglang.srt.arg_groups.hisparse_hook import _hisparse_default_backend

        backend = _hisparse_default_backend(kv_cache_dtype)
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = backend
        if not user_set_decode:
            declared["dsa_decode_backend"] = backend
        prefill = declared.get("dsa_prefill_backend", view.dsa_prefill_backend)
        decode = declared.get("dsa_decode_backend", view.dsa_decode_backend)
        logger.warning(
            f"HiSparse enabled ({kv_cache_dtype}): using DSA backends "
            f"prefill={prefill}, decode={decode}."
        )
        return declared

    if not user_set_prefill and not user_set_decode and is_hip():
        declared["dsa_prefill_backend"] = "tilelang"
        declared["dsa_decode_backend"] = "tilelang"
    elif kv_cache_dtype == "fp8_e4m3":
        # Blackwell FP8 defaults to trtllm; Hopper FP8 to flashmla_kv.
        default = "trtllm" if major >= 10 else "flashmla_kv"
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = default
        if not user_set_decode:
            declared["dsa_decode_backend"] = default
    else:
        # Set prefill/decode backends based on hardware architecture.
        if not user_set_prefill:
            declared["dsa_prefill_backend"] = "flashmla_sparse"
        if not user_set_decode:
            declared["dsa_decode_backend"] = "trtllm" if major >= 10 else "fa3"

    prefill = declared.get("dsa_prefill_backend", view.dsa_prefill_backend)
    decode = declared.get("dsa_decode_backend", view.dsa_decode_backend)
    logger.warning(
        f"Set DSA backends for {kv_cache_dtype} KV Cache: "
        f"prefill={prefill}, decode={decode}."
    )
    return declared


# Keep in sync with the DeepSeek family list on _deepseek_family_overrides.
_DEEPSEEK_FAMILY_ARCHS = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "KimiK25ForConditionalGeneration",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "GlmMoeDsaForCausalLM",
        "LongcatFlashForCausalLM",
        "LongcatFlashForCausalLMNextN",
    }
)


@register_post_process
def _deepseek_moe_quant_resolution(view: Any) -> dict:
    """Slot pass invoked from inside the DeepSeek arch branch ("Set moe
    backend for DeepSeek"), NOT a dispatch-time declaration: the DSA
    kv-cache-dtype default earlier in the branch must read the PRISTINE
    quantization, so this resolution has to stay at its legacy slot."""
    hf_config = view.get_model_config().hf_config
    model_arch = hf_config.architectures[0]
    if model_arch not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    overrides: Dict[str, Any] = {}
    if is_sm100_supported():
        quant_method = get_quantization_config(hf_config)
        quant_cfg = getattr(hf_config, "quantization_config", None) or {}
        config_groups = quant_cfg.get("config_groups", {})
        group0 = config_groups.get("group_0", {})
        weights_cfg = group0.get("weights", {})
        # this also apply to kimi k2.5
        # since it follow the compressed tensor int4 recipe
        # but not kimi k2 instruct or 0905 instruct.
        is_kimi_k2_k25_thinking_int4 = (
            quant_method == "compressed-tensors"
            and weights_cfg.get("num_bits") == 4
            and weights_cfg.get("group_size") == 32
            and weights_cfg.get("strategy") == "group"
            and weights_cfg.get("type") == "int"
        )
        quantization = view.quantization
        if quantization is None and not view._quantization_explicitly_unset:
            # DeepSeek V3/R1 uses native FP8 MoE experts without
            # declaring it in quantization_config.  However, other
            # models that share the same architecture class (e.g.
            # Moonlight-16B-A3B) are purely BF16.  Check the actual
            # safetensors header instead of assuming FP8 by arch name.
            if quant_method is None and model_arch in ["DeepseekV3ForCausalLM"]:
                from sglang.srt.utils.common import has_fp8_weights_in_checkpoint

                if has_fp8_weights_in_checkpoint(view.model_path):
                    overrides["quantization"] = quantization = "fp8"
                    logger.info(
                        "Detected FP8 expert weights in checkpoint, "
                        "default to fp8 for DeepSeek on sm100"
                    )
                else:
                    logger.info(
                        "No FP8 expert weights found in checkpoint, "
                        "keeping bf16 for DeepSeek-arch model on sm100"
                    )
            else:
                overrides["quantization"] = quantization = quant_method
        if (
            view.moe_a2a_backend == "none"
            and view.moe_runner_backend == "auto"
            and (
                quantization
                in ["fp8", "modelopt_fp8", "modelopt_fp4", "modelopt_mixed"]
                or is_kimi_k2_k25_thinking_int4
                or quantization is None
            )
        ):
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            if is_kimi_k2_k25_thinking_int4:
                logger.info(
                    "Use flashinfer_trtllm as MoE runner backend on Blackwell for Kimi K2 / K2.5 thinking int4"
                )
            else:
                logger.info(
                    "Use flashinfer_trtllm as MoE runner backend on sm100 for DeepseekV3ForCausalLM"
                )
        if (
            model_arch in ["LongcatFlashForCausalLM", "LongcatFlashForCausalLMNextN"]
            and view.fp8_gemm_runner_backend == "auto"
            and quantization in ["fp8", "modelopt_fp8"]
            and quant_cfg.get("scale_fmt", None) != "ue8m0"
        ):
            overrides["fp8_gemm_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as FP8 GEMM backend on Blackwell for LongCat FP8 "
                "checkpoint with non-ue8m0 scales"
            )
    return overrides


@register_post_process
def _deepseek_spec_moe_resolution(view: Any) -> dict:
    """Slot pass at the DeepSeek branch's HIP arm: draft (nextn) spec-MoE
    backends for the DeepSeek fp4 checkpoint. Reads the mid-resolution
    quantization (after _deepseek_moe_quant_resolution) and the pre-a2a
    ep_size, exactly like the legacy in-branch writes."""
    from sglang.srt.environ import envs

    hf_config = view.get_model_config().hf_config
    model_arch = hf_config.architectures[0]
    if model_arch not in _DEEPSEEK_FAMILY_ARCHS:
        return {}
    if not is_hip():
        return {}
    if not (
        view.quantization == "modelopt_fp4"
        and view.speculative_algorithm == "EAGLE"
        and (
            view.speculative_moe_runner_backend is None
            or view.speculative_moe_a2a_backend is None
        )
    ):
        return {}
    if envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get():
        logger.info(
            "Use deep_gemm moe runner and deepep a2a backend for bf16 nextn layer in deepseek fp4 checkpoint."
        )
        # Validate usage of ep
        if view.ep_size == 1:
            raise ValueError(
                "Invalid configuration: 'deep_gemm' speculative MoE runner backend with "
                "'deepep' a2a backend requires expert parallelism (ep_size > 1). "
                f"Current ep_size is {view.ep_size}. "
                "Please set --ep-size > 1 (e.g., --ep-size 8) to use this configuration, "
                "or change --speculative-moe-a2a-backend to 'none' if expert parallelism is not available."
            )
        return {
            "speculative_moe_runner_backend": "deep_gemm",
            "speculative_moe_a2a_backend": "deepep",
        }
    logger.info(
        "Use triton fused moe by default for bf16 nextn layer in deepseek fp4 checkpoint."
    )
    return {
        "speculative_moe_runner_backend": "triton",
        "speculative_moe_a2a_backend": "none",
    }


@register_post_process
def _deepseek_v4_kv_cache_dtype(view: Any) -> dict:
    """Slot pass in the DeepSeek V4 hook: default the kv-cache dtype to FP8
    (bfloat16 on NPU, where the pool geometry differs) and validate the
    result. The NPU split-backend writes stay in the hook."""
    hf_config = view.get_model_config().hf_config
    model_arch = hf_config.architectures[0]
    if model_arch != "DeepseekV4ForCausalLM":
        return {}

    kv_cache_dtype = view.kv_cache_dtype
    if kv_cache_dtype == "auto":
        kv_cache_dtype = "fp8_e4m3"
        logger.warning(f"Setting KV cache dtype to {kv_cache_dtype} for {model_arch}.")
    if view.device == "npu":
        kv_cache_dtype = "bfloat16"
    assert kv_cache_dtype in [
        "fp8_e4m3",
        "bfloat16",
    ], f"{kv_cache_dtype} is not supported for {model_arch}"
    if kv_cache_dtype != view.kv_cache_dtype:
        return {"kv_cache_dtype": kv_cache_dtype}
    return {}


@register_post_process
def _deepseek_v4_sm120_moe(view: Any) -> dict:
    """Slot pass in the DeepSeek V4 validation branch: SM120 lacks
    tcgen05/TMEM, fall back to the marlin MoE runner (reads the
    mid-resolution moe_runner_backend, after the dispatch-time nvfp4
    default)."""
    from sglang.srt.environ import envs

    model_config = view.get_model_config()
    hf_config = model_config.hf_config
    if hf_config.architectures[0] != "DeepseekV4ForCausalLM":
        return {}
    if (
        is_sm120_supported()
        and view.moe_runner_backend == "auto"
        and not (model_config.is_fp4_experts and envs.SGLANG_DSV4_FP4_DEQUANT.get())
    ):
        logger.info("Use marlin as MoE runner backend on SM120 for DeepseekV4")
        return {"moe_runner_backend": "marlin"}
    return {}


@register_post_process
def _sparse_head_overlap_disable(view: Any) -> dict:
    from sglang.srt.environ import envs

    if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set():
        logger.warning(
            "Overlap scheduler is disabled when using sparse head for embedding model."
        )
        return {"disable_overlap_schedule": True}
    return {}


# Architectures with explicit FlashInfer AllReduce Fusion support. Keep in
# sync with the model-side fusion implementations.
_FLASHINFER_ALLREDUCE_FUSION_ARCHS = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GptOssForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "MistralLarge3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3NextForCausalLM",
        "KimiK25ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "NemotronHForCausalLM",
        "NemotronHPuzzleForCausalLM",
    }
)


@register_post_process
def _flashinfer_allreduce_fusion_auto_enable(view: Any) -> dict:
    """Slot pass at the monolith tail: auto-enable FlashInfer AllReduce
    Fusion on SM90/SM100 for models with explicit support. auto resolves to
    mnnvl on Blackwell (single- and multi-node) and trtllm on SM90
    single-node systems. Reads the mid-resolution enable_dp_attention /
    moe_a2a_backend (after the DeepSeek CP and a2a declarations), exactly
    like the legacy tail block."""
    model_arch = view.get_model_config().hf_config.architectures[0]
    if (
        view.flashinfer_allreduce_fusion_backend is None
        and model_arch in _FLASHINFER_ALLREDUCE_FUSION_ARCHS
        and (is_sm90_supported() or is_sm100_supported())
        and view.tp_size > 1
        and not view.enable_dp_attention
        and (view.nnodes == 1 or is_sm100_supported())
        and view.moe_a2a_backend == "none"
    ):
        logger.info(
            f"Auto-enabling FlashInfer AllReduce Fusion on SM90/SM10X for {model_arch}"
        )
        return {"flashinfer_allreduce_fusion_backend": "auto"}
    return {}


@register_post_process
def _enforce_disable_allreduce_fusion(view: Any) -> dict:
    """Slot pass right after the auto-enable: the user's enforce-disable
    switch wins over every model-specific adjustment."""
    if view.enforce_disable_flashinfer_allreduce_fusion:
        logger.info(
            "FlashInfer allreduce fusion is forcibly disabled "
            "via --enforce-disable-flashinfer-allreduce-fusion."
        )
        return {"flashinfer_allreduce_fusion_backend": None}
    return {}


@register_post_process
def _sampling_backend_default(view: Any) -> dict:
    if view.sampling_backend is None:
        return {
            "sampling_backend": (
                "flashinfer" if is_flashinfer_available() else "pytorch"
            )
        }
    return {}


@register_post_process
def _deterministic_sampling_backend(view: Any) -> dict:
    if view.enable_deterministic_inference and view.sampling_backend != "ascend":
        logger.warning(
            "Sampling backend is set to pytorch for deterministic inference."
        )
        return {"sampling_backend": "pytorch"}
    return {}


def _deterministic_is_deepseek_model(view: Any) -> bool:
    """Faithful copy of the deterministic handler's arch probe (pure read;
    the handler keeps its own copy for the later deepseek validation)."""
    from sglang.srt.connector import ConnectorType
    from sglang.srt.utils.common import parse_connector_type

    if parse_connector_type(view.model_path) == ConnectorType.INSTANCE:
        return False
    try:
        hf_config = view.get_model_config().hf_config
        return hf_config.architectures[0] in [
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
            "GlmMoeDsaForCausalLM",
        ]
    except Exception:
        return False


@register_post_process
def _deterministic_allreduce_fusion_disable(view: Any) -> dict:
    if (
        view.enable_deterministic_inference
        and view.flashinfer_allreduce_fusion_backend is not None
    ):
        logger.warning(
            "Disable --flashinfer-allreduce-fusion-backend because deterministic inference is enabled."
        )
        return {"flashinfer_allreduce_fusion_backend": None}
    return {}


@register_post_process
def _deterministic_attention_backend(view: Any) -> dict:
    if not view.enable_deterministic_inference:
        return {}
    from sglang.srt.server_args import DETERMINISTIC_ATTENTION_BACKEND_CHOICES

    if view.attention_backend is None:
        # User didn't specify attention backend, fallback based on GPU architecture
        if is_sm100_supported() or is_sm120_supported():
            # Blackwell and newer architectures
            if _deterministic_is_deepseek_model(view):
                # fallback to triton for DeepSeek models because flashinfer
                # doesn't support deterministic inference for DeepSeek models yet
                backend = "triton"
            else:
                # fallback to flashinfer on Blackwell for non-DeepSeek models
                backend = "flashinfer"
        else:
            # Hopper (SM90) and older architectures
            backend = "fa3"
        logger.warning(
            f"Attention backend not specified. Falling back to '{backend}' for deterministic inference. "
            f"You can explicitly set --attention-backend to one of {DETERMINISTIC_ATTENTION_BACKEND_CHOICES}."
        )
        return {"attention_backend": backend}
    elif view.attention_backend not in DETERMINISTIC_ATTENTION_BACKEND_CHOICES:
        # User explicitly specified an incompatible attention backend
        raise ValueError(
            f"Currently only {DETERMINISTIC_ATTENTION_BACKEND_CHOICES} attention backends are supported for deterministic inference, "
            f"but you explicitly specified '{view.attention_backend}'."
        )
    return {}


@register_post_process
def _attention_backend_default(view: Any) -> dict:
    if view.prefill_attention_backend is not None and (
        view.prefill_attention_backend == view.decode_attention_backend
    ):  # override the default attention backend
        return {"attention_backend": view.prefill_attention_backend}
    if view.attention_backend is None:
        backend = view._get_default_attn_backend(
            view.use_mla_backend(), view.get_model_config()
        )
        logger.info(
            f"Attention backend not specified. Use {backend} backend by default."
        )
        return {"attention_backend": backend}
    return {}


@register_post_process
def _mla_backend_page_constraints(view: Any) -> dict:
    """Page-size constraints of the MLA/TRTLLM backend family (the raises and
    the cutedsl prefill fallback stay in the handler; only the page snaps are
    declared). The snaps chain on a local value exactly as the legacy blocks
    chained on self.page_size."""
    page_size = view.page_size
    if (
        view.attention_backend == "flashmla"
        or view.decode_attention_backend == "flashmla"
    ):
        logger.warning(
            "FlashMLA only supports a page_size of 64, change page_size to 64."
        )
        page_size = 64
    if (
        view.attention_backend == "cutlass_mla"
        or view.decode_attention_backend == "cutlass_mla"
    ):
        logger.warning(
            "Cutlass MLA only supports a page_size of 128, change page_size to 128."
        )
        page_size = 128
    if (
        view.attention_backend == "trtllm_mla"
        or view.decode_attention_backend == "trtllm_mla"
    ):
        if page_size not in [32, 64]:
            logger.warning(
                f"TensorRT-LLM MLA only supports page_size of 32 or 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "tokenspeed_mla"
        or view.decode_attention_backend == "tokenspeed_mla"
    ):
        if page_size not in [32, 64]:
            logger.warning(
                f"tokenspeed_mla only supports page_size of 32 or 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "cutedsl_mla"
        or view.decode_attention_backend == "cutedsl_mla"
        or view.prefill_attention_backend == "cutedsl_mla"
    ):
        if page_size not in [32, 64]:
            logger.warning(
                f"CuteDSL MLA only supports page_size of 32 or 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if (
        view.attention_backend == "trtllm_mha"
        or view.decode_attention_backend == "trtllm_mha"
        or view.prefill_attention_backend == "trtllm_mha"
    ):
        if page_size not in [16, 32, 64]:
            logger.warning(
                f"TensorRT-LLM MHA only supports page_size of 16, 32 or 64, changing page_size from {page_size} to 64."
            )
            page_size = 64
    if page_size != view.page_size:
        return {"page_size": page_size}
    return {}


@register_post_process
def _mla_kv_cache_dtype_checks(view: Any) -> dict:
    """Read-only validation pass in the attention-backend compatibility
    handler: the TRT-LLM and tokenspeed MLA backends constrain the resolved
    kv-cache dtype (declarations never reach the field, so the checks read
    the view)."""
    if (
        view.attention_backend == "trtllm_mla"
        or view.decode_attention_backend == "trtllm_mla"
    ):
        if not is_blackwell_supported():
            raise ValueError(
                "TRTLLM MLA backend is only supported on Blackwell GPUs (SM100/SM12x). Please use a different backend."
            )
        if view.kv_cache_dtype not in ["fp8_e4m3", "fp4_e2m1", "bf16", "auto"]:
            raise ValueError(
                "TensorRT-LLM MLA backend only supports kv-cache-dtype of fp8_e4m3, fp4_e2m1, bf16, or auto."
            )
    if (
        view.attention_backend == "tokenspeed_mla"
        or view.decode_attention_backend == "tokenspeed_mla"
    ):
        if not is_blackwell_supported():
            raise ValueError(
                "tokenspeed_mla backend is only supported on Blackwell GPUs (SM100/SM12x)."
            )
        if view.kv_cache_dtype not in ["fp8_e4m3"]:
            raise ValueError(
                "tokenspeed_mla backend requires kv-cache-dtype=fp8_e4m3, "
                f"got {view.kv_cache_dtype}."
            )
    return {}


@register_post_process
def _hisparse_validation(view: Any) -> dict:
    """Read-only validation pass: --enable-hisparse constraints (model class,
    radix cache, kv dtype, DSA backends) read the resolved values through the
    view."""
    from sglang.srt.arg_groups.hisparse_hook import validate_hisparse

    validate_hisparse(view)
    return {}


@register_post_process
def _cutedsl_prefill_backend_fill(view: Any) -> dict:
    """Slot pass in the attention-backend compatibility handler: CuteDSL MLA
    is decode-only, so validate the combination and default the prefill side
    to trtllm_mla. The trtllm_mha check that follows at the legacy slot reads
    the resolved value through the view."""
    if not (
        view.attention_backend == "cutedsl_mla"
        or view.decode_attention_backend == "cutedsl_mla"
        or view.prefill_attention_backend == "cutedsl_mla"
    ):
        return {}
    assert (
        view.prefill_attention_backend != "cutedsl_mla"
    ), "CuteDSL MLA only supports decoding for now"
    if not is_sm100_supported():
        raise ValueError(
            "CuteDSL MLA backend is only supported on Blackwell GPUs (SM100). Please use a different backend."
        )
    if view.kv_cache_dtype not in [
        "fp8_e4m3",
        "bf16",
        "bfloat16",
        "auto",
    ]:
        raise ValueError(
            "CuteDSL MLA backend only supports kv-cache-dtype of fp8_e4m3, bf16, or auto."
        )
    if view.prefill_attention_backend is None:
        return {"prefill_attention_backend": "trtllm_mla"}
    return {}


@register_post_process
def _attention_backend_fa3_fp8_fallback(view: Any) -> dict:
    if view.attention_backend == "fa3" and view.kv_cache_dtype == "fp8_e5m2":
        logger.warning(
            "FlashAttention3 only supports fp8_e4m3 if using FP8; "
            "Setting attention backend to triton."
        )
        return {"attention_backend": "triton"}
    return {}


@register_post_process
def _fa4_page_constraint(view: Any) -> dict:
    if (
        (
            view.attention_backend == "fa4"
            or view.decode_attention_backend == "fa4"
            or view.prefill_attention_backend == "fa4"
        )
        and not view.use_mla_backend()
        and is_sm100_supported()
        # EAGLE topk>1 spec runs the two-pass page-tree cascade, which the FA4
        # CUTLASS kernel aborts on at page_size>1. That path only works at
        # page_size==1, so skip the 128 auto-force for it and keep the default.
        and (view.speculative_eagle_topk or 0) <= 1
    ):
        logger.warning(
            f"FA4 backend only supports page size 128 for non-MLA model architectures, changing page_size from {view.page_size} to 128."
        )
        return {"page_size": 128}
    return {}


@register_post_process
def _attention_backend_platform_fallbacks(view: Any) -> dict:
    if (
        view.attention_backend == "intel_amx"
        and view.device == "cpu"
        and not cpu_has_amx_support()
    ):
        logger.warning(
            "The current platform does not support Intel AMX, will fallback to torch_native backend."
        )
        return {"attention_backend": "torch_native"}
    if (
        view.attention_backend == "intel_xpu"
        and view.device == "xpu"
        and not xpu_has_xmx_support()
    ):
        logger.warning(
            "The current platform does not support Intel XMX, will fallback to triton backend."
        )
        return {"attention_backend": "triton"}
    return {}


@register_post_process
def _intel_xpu_page_constraint(view: Any) -> dict:
    _, decode_backend = attention_backends_of(view)
    if decode_backend == "intel_xpu":
        if view.use_mla_backend():
            supported_page_sizes = [16, 32, 64, 128]
            msg = "Intel XPU attention backend for MLA Decode"
        else:
            supported_page_sizes = [64, 128]
            msg = "Intel XPU attention backend"
        if view.page_size not in supported_page_sizes:
            logger.warning(
                f"{msg} only supports page_sizes of {supported_page_sizes}, changing page_size from {view.page_size} to 128."
            )
            return {"page_size": 128}
    return {}


@register_post_process
def _attention_backend_dual_chunk(view: Any) -> dict:
    if (
        getattr(view.get_model_config().hf_config, "dual_chunk_attention_config", None)
        is not None
    ):
        if view.attention_backend is None:
            logger.info("Dual chunk attention is turned on by default.")
            return {"attention_backend": "dual_chunk_flash_attn"}
        elif view.attention_backend != "dual_chunk_flash_attn":
            raise ValueError(
                "Dual chunk attention is enabled, but attention backend is set to "
                f"{view.attention_backend}. Please set it to 'dual_chunk_flash_attn'."
            )
    return {}


@register_post_process
def _page_size_default(view: Any) -> dict:
    if view.page_size is not None:
        return {}
    from sglang.srt.environ import envs

    # SHUFFLE 5D vectorized KV layout (aiter backend + pa_decode_gluon)
    # is tuned for and prefers page_size=64 — making it the default
    # when the layout flag is set avoids users having to pass
    # --page-size 64 explicitly. The env var is only consumed by the
    # ROCm AITER backend, so the auto-bump is gated on HIP; on other
    # platforms the SHUFFLE 5D pool has no consumer kernels and the
    # env var is silently ignored (see MHATokenToKVPool).
    if is_hip() and envs.SGLANG_AITER_KV_CACHE_LAYOUT.get().lower() == "vectorized_5d":
        logger.info(
            "Setting page_size=64 as default for "
            "SGLANG_AITER_KV_CACHE_LAYOUT=vectorized_5d."
        )
        return {"page_size": 64}
    if not is_musa():
        return {"page_size": 1}
    return {"page_size": 64}


@register_post_process
def _data_parallelism_defaults(view: Any) -> dict:
    if view.dp_size == 1:
        return {"enable_dp_attention": False, "enable_dp_lm_head": False}
    return {}


@register_post_process
def _dp_lm_head_validation(view: Any) -> dict:
    """Read-only validation pass: dp-attention is a prerequisite for the
    dp LM head. Reads the mid-resolution values through the view."""
    if view.enable_dp_lm_head:
        assert (
            view.enable_dp_attention
        ), "Please enable dp attention when setting enable_dp_lm_head. "
    return {}


@register_post_process
def _moe_runner_backend_quant_constraints(view: Any) -> dict:
    """The quantization-driven moe_runner_backend resolutions at the head of
    _handle_moe_kernel_config. The backend-compatibility asserts and the
    disable_shared_experts_fusion writes (post-publish writers exist for that
    field) stay in the handler."""
    moe_runner_backend = view.moe_runner_backend
    if view.quantization == "nvfp4_online":
        if not is_sm100_supported():
            raise ValueError(
                "--quantization nvfp4_online is supported only on "
                "NVIDIA Blackwell SM100/SM103 GPUs."
            )
        if moe_runner_backend == "auto":
            moe_runner_backend = "flashinfer_trtllm"
        elif moe_runner_backend not in [
            "flashinfer_trtllm",
            "flashinfer_trtllm_routed",
        ]:
            raise ValueError(
                "--quantization nvfp4_online supports only "
                "--moe-runner-backend flashinfer_trtllm or "
                "flashinfer_trtllm_routed."
            )
    if view.quantization == "mxfp8":
        if moe_runner_backend == "auto":
            moe_runner_backend = "flashinfer_trtllm"
        elif moe_runner_backend not in [
            "cutlass",
            "flashinfer_trtllm",
            "flashinfer_trtllm_routed",
        ]:
            logger.warning(
                "mxfp8 quantization supports only cutlass, flashinfer_trtllm, "
                "or flashinfer_trtllm_routed backends. "
                f"Overriding {moe_runner_backend!r}."
            )
            moe_runner_backend = "flashinfer_trtllm"
    if (
        moe_runner_backend == "auto"
        and view.quantization == "modelopt_fp4"
        and is_sm120_supported()
    ):
        moe_runner_backend = "flashinfer_cutlass"
        logger.info(
            "Use flashinfer_cutlass as MoE runner backend on SM120 for "
            "modelopt_fp4 (trtllm-gen MoE kernels are SM100-only)"
        )
    if moe_runner_backend != view.moe_runner_backend:
        return {"moe_runner_backend": moe_runner_backend}
    return {}


@register_post_process
def _moe_runner_fusion_disable(view: Any) -> dict:
    """FlashInfer CuteDSL / TRT-LLM / TRT-LLM-routed MoE runners require the
    shared-experts fusion disabled; declared at the legacy write slots in
    _handle_moe_kernel_config (before the deprecated cutlass env override, so
    the runner value observed is the pre-override one)."""
    runner = view.moe_runner_backend
    if runner == "flashinfer_cutedsl":
        logger.warning(
            "FlashInfer CuteDSL MoE is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    if runner in ("flashinfer_trtllm", "experimental_sgl_trtllm"):
        logger.warning(
            "FlashInfer TRTLLM MoE is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    if runner == "flashinfer_trtllm_routed":
        logger.warning(
            "FlashInfer TRTLLM routed MoE is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    return {}


def _a2a_fusion_adjustments(view: Any) -> dict:
    """A2A-backend-driven shared-experts fusion adjustments, declared at the
    legacy write slots in _handle_a2a_moe: DeepEP Waterfill requires the
    fusion enabled; FlashInfer A2A requires it disabled."""
    if view.moe_a2a_backend == "deepep" and view.enable_deepep_waterfill:
        if view.disable_shared_experts_fusion:
            logger.warning(
                "disable_shared_experts_fusion is overridden to False because DeepEP Waterfill requires shared expert fusion."
            )
            return {"disable_shared_experts_fusion": False}
        return {}
    if view.moe_a2a_backend == "flashinfer":
        logger.warning(
            "Flashinfer MoE A2A is enabled. --disable-shared-experts-fusion is automatically set."
        )
        return {"disable_shared_experts_fusion": True}
    return {}


def _cutlass_moe_env_override(view: Any) -> dict:
    from sglang.srt.environ import envs

    if envs.SGLANG_CUTLASS_MOE.get():
        logger.warning(
            "SGLANG_CUTLASS_MOE is deprecated, use --moe-runner-backend=cutlass and/or --speculative-moe-runner-backend=cutlass instead"
        )
        assert view.quantization in [
            "fp8",
            "mxfp8",
        ], "cutlass MoE is only supported with fp8/mxfp8 quantization"
        return {"moe_runner_backend": "cutlass"}
    return {}


# Every A2A backend that forces expert parallelism to span the TP group.
_A2A_EP_SPANNING_BACKENDS = frozenset(
    {"megamoe", "deepep", "mooncake", "nixl", "ascend_fuseep", "flashinfer", "mori"}
)


@register_post_process
def _a2a_backend_overrides(view: Any) -> dict:
    from sglang.srt.environ import envs

    moe_a2a_backend = view.moe_a2a_backend
    if view.enable_deepep_waterfill and moe_a2a_backend != "deepep":
        logger.warning(
            "moe_a2a_backend is overridden to 'deepep' because DeepEP "
            "Waterfill requires the DeepEP backend."
        )
        moe_a2a_backend = "deepep"
    if envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get() and moe_a2a_backend != "megamoe":
        moe_a2a_backend = "megamoe"
        logger.info(
            "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE is set, "
            "auto-configuring --moe-a2a-backend megamoe."
        )
    if moe_a2a_backend != view.moe_a2a_backend:
        return {"moe_a2a_backend": moe_a2a_backend}
    return {}


@register_post_process
def _a2a_ep_size(view: Any) -> dict:
    if view.moe_a2a_backend in _A2A_EP_SPANNING_BACKENDS:
        return {"ep_size": view.tp_size}
    return {}


@register_post_process
def _pipeline_parallel_overlap_disable(view: Any) -> dict:
    if view.pp_size > 1:
        logger.warning("Pipeline parallelism is incompatible with overlap schedule.")
        return {"disable_overlap_schedule": True}
    return {}


@register_post_process
def _speculative_moe_runner_default(view: Any) -> dict:
    """Default the speculative (draft) MoE runner backend to the resolved
    target-model backend. Invoked at the head of the speculative-decoding
    hook, after the MoE kernel chain has resolved."""
    if view.speculative_moe_runner_backend is None:
        return {"speculative_moe_runner_backend": view.moe_runner_backend}
    return {}


@register_post_process
def _gguf_quantization(view: Any) -> dict:
    from sglang.srt.utils.hf_transformers_utils import check_gguf_file

    if (view.load_format == "auto" or view.load_format == "gguf") and check_gguf_file(
        view.model_path
    ):
        return {"quantization": "gguf"}
    return {}


@register_post_process
def _dllm_attention_backend(view: Any) -> dict:
    if view.dllm_algorithm is None:
        return {}
    if is_hip():
        if view.attention_backend not in ["triton", "aiter"]:
            logger.warning(
                "Attention backend is set to triton for diffusion LLM inference on AMD GPUs"
            )
            return {"attention_backend": "triton"}
    elif is_npu():
        if view.attention_backend != "ascend":
            logger.warning(
                "Attention backend is overridden to 'ascend' when running on NPU for diffusion LLM inference."
            )
            return {"attention_backend": "ascend"}
    elif view.cuda_graph_config.decode.backend != Backend.DISABLED:
        if view.attention_backend != "flashinfer":
            logger.warning(
                "Attention backend is set to flashinfer because of enabling cuda graph in diffusion LLM inference"
            )
            return {"attention_backend": "flashinfer"}
    return {}


@register_post_process
def _dllm_overlap_disable(view: Any) -> dict:
    if view.dllm_algorithm is None:
        return {}
    if view.disable_overlap_schedule:
        return {}
    logger.warning(
        "Overlap schedule is disabled because of using diffusion LLM inference"
    )
    return {"disable_overlap_schedule": True}


@register_post_process
def _dllm_page_size(view: Any) -> dict:
    if view.dllm_algorithm is None:
        return {}
    from sglang.srt.dllm.config import DllmConfig

    config = DllmConfig.from_server_args(view)
    if not view.disable_radix_cache and view.page_size % config.block_size != 0:
        logger.warning(
            f"Setting page size to {config.block_size} for diffusion LLM inference"
        )
        return {"page_size": config.block_size}
    if view.page_size > config.block_size:
        # Legacy scheduler-init fallback, folded into the pass: the page
        # size must not exceed the dllm block size.
        logger.warning(
            "WARNING: "
            f"The page size {view.page_size} should not be larger than dllm block size {config.block_size}."
            f"Page size now falls back to {config.block_size}"
        )
        return {"page_size": config.block_size}
    return {}


def validate_declarations(
    server_args: Any,
    declarations: Sequence[Tuple[str, Dict[str, Any]]],
) -> None:
    """Fail-fast whitelist check at declaration time: a registry typo or a
    not-yet-resolvable field must be rejected at its slot, not only at
    publish time. Declarations never mutate ``server_args``.
    """
    # Non-dataclass fixtures carry no Arg metadata (mirrors the
    # resolvable_fields escape); only real ServerArgs is validated.
    if not dataclasses.is_dataclass(type(server_args)):
        return
    whitelist = resolvable_fields(type(server_args))
    for source, decl in declarations:
        unknown = set(decl) - whitelist
        if unknown:
            raise ValueError(
                f"{source}: {sorted(unknown)} not model-overridable; "
                "declarations are limited to the fields the publish gate "
                "accepts."
            )


def _hrm_text_attention_force(view: Any) -> dict:
    """HRM-Text's bidirectional prefix attention only works on the Triton
    backend. Invoked as the last attention declaration of the resolution
    (mirroring the legacy runner-side force, which ran after the whole
    pipeline)."""
    if view.attention_backend not in (None, "triton"):
        logger.warning(
            f"Overriding --attention-backend "
            f"{view.attention_backend!r} -> 'triton': only the "
            "Triton backend supports HRM-Text's bidirectional prefix "
            "attention."
        )
    return {"attention_backend": "triton"}
