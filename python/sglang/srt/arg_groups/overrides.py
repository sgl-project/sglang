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
resolved into the flags tier through the ``apply_model_overrides`` gate —
model code never mutates ``ServerArgs``, which stays the pristine user input.

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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sglang.srt.arg_groups.arg_utils import model_overridable_fields
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import resolve_flag_leaf
from sglang.srt.utils.common import (
    cpu_has_amx_support,
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
    passes.

    During the dual-apply transition the view forwards every read to the live
    ``server_args`` — the pristine input plus the declarations replayed so far
    plus any residual imperative writes — which is exactly the state the
    legacy handler at the same slot observed. In the end state (dual-apply
    retired) the same type overlays the accumulated declarations on the
    pristine object. Writes are rejected: passes return declarations.
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


def run_post_process_pass(server_args: Any, fn: Callable[..., dict]) -> None:
    """Transition-period invocation of one pass at its legacy handler slot.

    Evaluates the pass on the live state (through a read-only view), appends
    its declaration to the R0 stash, and dual-applies it in place —
    byte-identical to the imperative handler write this replaces.
    """
    declared = fn(ResolvedView(server_args))
    if not isinstance(declared, dict):
        raise TypeError(
            f"post-process pass {fn.__qualname__} must return a dict, "
            f"got {type(declared).__name__}"
        )
    if declared:
        entry = (fn.__qualname__, dict(declared))
        server_args._resolved_overrides.append(entry)
        apply_declarations_to_server_args(server_args, [entry])


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
            # time (dsa.utils imports get_global_server_args).
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
    if default_attn_backend == "trtllm_mha" and not (
        not server_args.enable_mamba_extra_buffer()
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


# Keep in sync with the DeepSeek family list on _deepseek_family_overrides.
_DEEPSEEK_FAMILY_ARCHS = frozenset(
    {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "KimiK25ForConditionalGeneration",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "GlmMoeDsaForCausalLM",
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
    return overrides


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
    _, decode_backend = view.get_attention_backends()
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
def _moe_runner_backend_quant_constraints(view: Any) -> dict:
    """The quantization-driven moe_runner_backend resolutions at the head of
    _handle_moe_kernel_config. The backend-compatibility asserts and the
    disable_shared_experts_fusion writes (post-publish writers exist for that
    field — R2 territory) stay in the handler."""
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
def _dllm_page_size(view: Any) -> dict:
    if view.dllm_algorithm is None or view.disable_radix_cache:
        return {}
    from sglang.srt.dllm.config import DllmConfig

    config = DllmConfig.from_server_args(view)
    if view.page_size % config.block_size != 0:
        logger.warning(
            f"Setting page size to {config.block_size} for diffusion LLM inference"
        )
        return {"page_size": config.block_size}
    return {}


@dataclasses.dataclass(frozen=True)
class OverrideRecord:
    """Provenance of one resolved write: ``base`` is the value before this
    declaration applied (the pristine value for the first writer)."""

    source: str
    field: str
    base: Any
    resolved: Any


def apply_model_overrides(
    flags: Any,
    server_args: Any,
    declarations: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    terminal: Sequence[Tuple[str, Dict[str, Any]]] = (),
    whitelist: Optional[Iterable[str]] = None,
    leaf_map: Optional[Dict[str, str]] = None,
) -> List[OverrideRecord]:
    """Resolve model-override declarations into the flags tier.

    - **Transactional**: every declaration (``terminal`` included) is
      validated against the whitelist and the flag-leaf layout BEFORE any
      write; on error nothing is applied.
    - **Ordering**: ``declarations`` apply in order (last writer wins), then
      ``terminal`` (the enforce-disable pass) applies after everything.
    - **Materialization**: every whitelisted field becomes a flag leaf —
      declared fields carry the resolved value, undeclared ones the pristine
      ``server_args`` value — so readers only ever read flags, never a
      "flag or fallback to config" combination.
    - ``server_args`` is read-only here: resolution output lives on flags.

    Returns the provenance log, one record per declared write.
    """
    if whitelist is None:
        whitelist = model_overridable_fields(type(server_args))
    whitelist = frozenset(whitelist)

    ordered = list(declarations) + list(terminal)

    problems = [
        f"{source}: {sorted(set(decl) - whitelist)} not model-overridable"
        for source, decl in ordered
        if set(decl) - whitelist
    ]
    if problems:
        raise ValueError(
            "model override validation failed (nothing was applied): "
            + "; ".join(problems)
        )
    for field in sorted(whitelist):
        owner, leaf = resolve_flag_leaf(flags, field, leaf_map=leaf_map)
        if leaf not in type(owner).__dataclass_fields__:
            raise ValueError(
                f"flag leaf for '{field}' is not declared on "
                f"{type(owner).__name__} (declare the dataclass field and map "
                "it in FLAG_LEAF_MAP); nothing was applied"
            )
        if getattr(owner, "_frozen", False):
            raise RuntimeError(
                f"cannot resolve '{field}': {type(owner).__name__} is frozen; "
                "nothing was applied"
            )

    resolved = {field: getattr(server_args, field) for field in whitelist}
    records: List[OverrideRecord] = []
    for source, decl in ordered:
        for field, value in decl.items():
            records.append(OverrideRecord(source, field, resolved[field], value))
            resolved[field] = value

    for field, value in resolved.items():
        owner, leaf = resolve_flag_leaf(flags, field, leaf_map=leaf_map)
        setattr(owner, leaf, value)
    return records


def apply_declarations_to_server_args(
    server_args: Any,
    declarations: Sequence[Tuple[str, Dict[str, Any]]],
    *,
    terminal: Sequence[Tuple[str, Dict[str, Any]]] = (),
) -> None:
    """Transition-period dual-apply: replay declarations onto ``server_args``
    in gate order, byte-identical to the legacy imperative writes.

    Retired per field once that field's readers have all flipped to the flags
    tier (at which point the server_args field returns to pristine).

    Validates against the same whitelist as the publish gate BEFORE any write:
    a registry typo or a not-yet-resolvable field must fail fast here, not
    mutate ``server_args`` and only be rejected at publish time.
    """
    # Non-dataclass fixtures carry no Arg metadata (mirrors the
    # model_overridable_fields escape); only real ServerArgs is validated.
    if dataclasses.is_dataclass(type(server_args)):
        whitelist = model_overridable_fields(type(server_args))
        for source, decl in list(declarations) + list(terminal):
            unknown = set(decl) - whitelist
            if unknown:
                raise ValueError(
                    f"{source}: {sorted(unknown)} not model-overridable; the "
                    "transition dual-apply refuses fields the publish gate "
                    "would reject."
                )
    for _source, decl in list(declarations) + list(terminal):
        for field, value in decl.items():
            setattr(server_args, field, value)


def refresh_declared_fields(server_args: Any, fields: Iterable[str]) -> None:
    """Transition helper for legacy code that overwrites a resolved field
    AFTER the R0 collection in ``__post_init__`` (e.g.
    ``ModelRunner.model_specific_adjustment`` forcing ``attention_backend``
    for HRM-Text). Redeclares the live value so publish parity holds and the
    flags tier materializes the adjusted end state.
    """
    _missing = object()
    declarations = server_args._resolved_overrides
    for field in fields:
        effective = _missing
        for _source, decl in declarations:
            if field in decl:
                effective = decl[field]
        if effective is _missing:
            continue
        live = getattr(server_args, field)
        if effective != live:
            declarations.append((f"runtime_adjustment[{field}]", {field: live}))


def assert_flag_parity(
    flags: Any,
    server_args: Any,
    fields: Iterable[str],
    *,
    leaf_map: Optional[Dict[str, str]] = None,
) -> None:
    """Dual-apply drift guard: each migrated field's flag leaf must equal the
    (dual-applied) ``server_args`` value."""
    mismatches = []
    for field in fields:
        owner, leaf = resolve_flag_leaf(flags, field, leaf_map=leaf_map)
        flag_value = getattr(owner, leaf)
        args_value = getattr(server_args, field)
        if flag_value != args_value:
            mismatches.append(
                f"{field}: flags={flag_value!r} server_args={args_value!r}"
            )
    if mismatches:
        raise AssertionError("flag/server_args parity broken: " + "; ".join(mismatches))
