# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Kimi K3 packed-MXFP4 megakernel spec for GB300/SM103."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

from sglang.srt.kernel_plugins.nvidia.kimi_k3.profile import (
    full_decode_enabled,
    native_config_reasons,
    native_mla_enabled,
)
from sglang.srt.kernel_plugins.nvidia.kimi_k3.shape_policy import (
    DECODE_WIDTHS,
    prepared_widths,
)
from sglang.srt.runtime_context import get_model

try:
    from sglang.kernels.plugin_kernels.nvidia.spec import (
        BuildContext,
        BuiltMegakernel,
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
        Step,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.megakernels.spec import (
        BuildContext,
        BuiltMegakernel,
        Feature,
        MegakernelSpec,
        Numerics,
        NumericsKind,
        Requirements,
        Step,
    )

if TYPE_CHECKING:
    try:
        from sglang.srt.kernel_plugins.nvidia.binding import PreparationView
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.megakernels.binding import PreparationView

NATIVE_MLA = native_mla_enabled()
FULL_DECODE = full_decode_enabled()
EXTENDED_PROFILE = NATIVE_MLA or FULL_DECODE
BASE_WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)
WIDTHS = DECODE_WIDTHS if EXTENDED_PROFILE else BASE_WIDTHS
# Keep this independent of the host constant so find_spec rejects incompatibility.
REQUIRED_MEGAKERNEL_API_VERSION = 2

# Names read from SGLang internals. A contract test checks these before GPU use.
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py|srt/megakernels/spec.py": (
        "BuildContext",
        "BuiltMegakernel",
        "Feature",
        "MULTI_NODE_TP",
        "MegakernelSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "Step",
        "TOLERANCE",
        "binding",
        "capture_widths",
        "deadline_s",
        "mode",
        "prepare",
        "rows",
        "tokens_per_request",
        "view",
        "widths",
    ),
    "srt/kernel_plugins/nvidia/binding.py|srt/megakernels/binding.py": (
        "PreparationView",
        "MegakernelBinding",
        "megakernel_agree",
        "native_mla_supports_backend",
    ),
    "srt/runtime_context.py": ("get_model", "get_parallel"),
    "srt/arg_groups/fields/*.py": (
        "context_length",
        "cuda_graph_config",
        "decode_attention_backend",
        "dp_size",
        "enable_page_major_kv_layout",
        "enable_unified_memory",
        "flashinfer_mxfp4_moe_precision",
        "kv_cache_dtype",
        "moe_runner_backend",
        "page_size",
        "prefill_attention_backend",
        "dcp_size",
        "attn_cp_size",
        "enable_prefill_cp",
    ),
    "srt/models/kimi_k3.py": (
        "KimiK3ForConditionalGeneration",
        "language_model",
        "logits_processor",
        "lm_head",
    ),
    "srt/configs/kimi_linear.py": (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "mla_use_nope",
        "mla_use_output_gate",
        "hidden_act",
        "n_routed_experts",
        "num_experts",
        "num_experts_per_token",
        "num_expert_group",
        "topk_group",
        "moe_router_activation_func",
        "routed_expert_hidden_size",
        "moe_intermediate_size",
        "num_shared_experts",
        "latent_moe_use_norm",
        "linear_attn_config",
        "attn_res_block_size",
    ),
}


def _deadline_ns(seconds: Any) -> int:
    """Validate the build deadline once and freeze it into compiled-kernel keys."""
    if isinstance(seconds, bool):
        raise ValueError(
            f"Kimi K3 megakernel needs a finite positive deadline: {seconds!r}"
        )
    try:
        value = float(seconds)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Kimi K3 megakernel needs a finite positive deadline: {seconds!r}"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"Kimi K3 megakernel needs a finite positive deadline: {seconds!r}"
        )
    if value > ((1 << 63) - 1) / 1_000_000_000:
        raise ValueError(f"Kimi K3 megakernel deadline is out of range: {seconds!r}")
    nanoseconds = int(value * 1_000_000_000)
    if nanoseconds <= 0 or nanoseconds > (1 << 63) - 1:
        raise ValueError(f"Kimi K3 megakernel deadline is out of range: {seconds!r}")
    return nanoseconds


def _config_disable_reasons(cfg: Any, hf_config: Any) -> list[str]:
    reasons: list[str] = []
    graph_config = cfg.cuda_graph_config
    if graph_config is None or graph_config.decode.backend != "full":
        reasons.append("needs the FULL decode CUDA-graph backend")
    # Capture-only resolution clears speculative_algorithm, but still needs
    # intermediate layer taps that the specialized executor does not produce.
    if getattr(cfg, "aux_hidden_state_capture", None) is not None:
        reasons.append(
            "Kimi K3 specialized decode does not support "
            "--enable-return-aux-hidden-states"
        )
    if EXTENDED_PROFILE:
        reasons.extend(native_config_reasons(cfg))
        try:
            from sglang.srt.kernel_plugins.nvidia.binding import MegakernelBinding

            if getattr(MegakernelBinding, "native_mla_api_version", 0) != 1:
                reasons.append(
                    "extended decode profile requires SGLang native MLA API v1"
                )
            if getattr(cfg, "decode_attention_backend", None) == "cutedsl_mla":
                supports = getattr(
                    MegakernelBinding, "native_mla_supports_backend", None
                )
                if not callable(supports) or not supports("cutedsl_mla"):
                    reasons.append(
                        "extended Cute DSL decode requires SGLang's native "
                        "MLA backend capability"
                    )
        except ModuleNotFoundError:
            reasons.append("extended decode profile requires SGLang native MLA API v1")
    elif int(cfg.context_length or 0) != 8192:
        reasons.append("initial Kimi K3 coverage requires context length 8192")
    return reasons


def _kernel_config_disable_reasons(config: Any) -> list[str]:
    """Reject native K3 variants the fixed geometry/routing kernels cannot run.

    In particular, valid native softmax/grouped/top-k variants must not silently
    become the prepared router's one-group sigmoid top-16. SiTU coefficients,
    routed scaling and renormalization are supplied to providers dynamically.
    """
    expected = {
        "hidden_size": 7168,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "hidden_act": "situ",
        "n_routed_experts": 896,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_expert_group": 1,
        "topk_group": 1,
        "moe_router_activation_func": "sigmoid",
        "routed_expert_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "num_shared_experts": 2,
        "latent_moe_use_norm": True,
    }
    reasons = [
        f"prepared Kimi K3 kernels need {name}={value!r}, "
        f"found {getattr(config, name, None)!r}"
        for name, value in expected.items()
        if getattr(config, name, None) != value
    ]
    linear = getattr(config, "linear_attn_config", None)
    linear = linear if isinstance(linear, dict) else {}
    for name, value in (
        ("num_heads", 96),
        ("head_dim", 128),
        ("short_conv_kernel_size", 4),
        ("use_full_rank_gate", True),
    ):
        if linear.get(name) != value:
            reasons.append(
                f"prepared Kimi K3 kernels need linear_attn_config.{name}={value!r}, "
                f"found {linear.get(name)!r}"
            )
    block_size = getattr(config, "attn_res_block_size", None)
    if type(block_size) is not int or block_size <= 0 or math.ceil(93 / block_size) > 8:
        reasons.append("prepared Kimi K3 kernels need at most 8 AttnRes blocks")
    return reasons


def _model_disable_reasons(model: Any, hf_config: Any) -> list[str]:
    reasons: list[str] = []
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return ["the Kimi K3 language model is absent"]
    layers = tuple(getattr(getattr(language_model, "model", None), "layers", ()))
    if len(layers) != 93:
        reasons.append(f"needs the full 93-layer model, found {len(layers)} layers")
    text_config = getattr(language_model, "config", None)
    reasons.extend(_kernel_config_disable_reasons(text_config))
    full_ids = tuple(
        int(value) for value in getattr(text_config, "full_attention_layer_ids", ())
    )
    linear_ids = tuple(
        int(value) for value in getattr(text_config, "linear_layer_ids", ())
    )
    if (len(full_ids), len(linear_ids)) != (24, 69):
        reasons.append(
            "needs the 24-MLA/69-KDA Kimi K3 attention split, found "
            f"{len(full_ids)} MLA and {len(linear_ids)} KDA layers"
        )
    methods = []
    for layer in layers:
        if not bool(getattr(layer, "_is_moe_layer", False)):
            continue
        method = getattr(getattr(layer.mlp, "experts", None), "quant_method", None)
        methods.append(type(method).__name__.lower())
    if not methods or not all("mxfp4" in name for name in methods):
        reasons.append("needs packed-MXFP4 routed experts on every MoE layer")
    return reasons


def _call_option(
    args: tuple[Any, ...], kwargs: dict[str, Any], index: int, name: str, default: Any
) -> Any:
    return kwargs.get(name, args[index] if len(args) > index else default)


class _Built:
    """Top-level model-forward replacement for the exact decode shapes served."""

    draft_sampler = None
    draft_loop = None
    draft_extend = None
    begin_draft_round = None
    graph_variants = None
    speculative_runtime_hooks = None

    def __init__(self, model: Any, resources: Any, widths: tuple[int, ...]):
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.executor import (
            KimiK3MegakernelExecutor,
        )

        self.model = model
        self.language_model = model.language_model
        self.resources = resources
        self.capture_only = EXTENDED_PROFILE
        self.widths = widths
        self.executor = KimiK3MegakernelExecutor(resources)
        # Local validation observers read this without changing the public API.
        model.__dict__["_nvidia_kimi_k3_megakernel"] = self

    def summary(self) -> str:
        suffix = (
            "; native MLA/DCP and cache restore hooks"
            if getattr(self.resources, "native_mla", False)
            else ""
        )
        if getattr(self.resources, "full_decode", False):
            suffix = "; fused MLA/DCP, skinny projection, embedding and L2 prefetch"
        rows = ", ".join(map(str, self.widths))
        return f"TP8 one-token decode at prepared rows {rows}" + suffix

    def plan(
        self, input_ids: Any, forward_batch: Any, *args: Any, **kwargs: Any
    ) -> Step | None:
        if getattr(self, "capture_only", False):
            import torch

            # Keep profiling and warmup native so fallback kernels are ready
            # before graph capture. Only full decode graphs run this profile.
            if not torch.cuda.is_current_stream_capturing():
                return None
        get_embedding = bool(_call_option(args, kwargs, 0, "get_embedding", False))
        pp_proxy_tensors = _call_option(args, kwargs, 1, "pp_proxy_tensors", None)
        if get_embedding or pp_proxy_tensors is not None:
            return None
        mode = forward_batch.forward_mode
        rows = int(input_ids.shape[0])
        if not mode.is_decode() or rows not in self.widths:
            return None
        if int(forward_batch.batch_size) != rows:
            return None
        return Step(mode="decode", rows=rows)

    def forward(
        self,
        step: Step,
        original: Any,
        input_ids: Any,
        positions: Any,
        forward_batch: Any,
        get_embedding: bool = False,
        pp_proxy_tensors: Any = None,
    ) -> Any:
        if get_embedding or pp_proxy_tensors is not None:
            raise RuntimeError("Kimi K3 megakernel received an unplanned model call")
        if step.mode != "decode" or step.rows not in self.widths:
            raise RuntimeError(f"unsupported Kimi K3 megakernel step: {step}")
        if int(forward_batch.batch_size) != step.rows:
            raise RuntimeError("Kimi K3 megakernel plan and forward batch disagree")
        if step.view is None or step.view.hybrid is None:
            raise RuntimeError("Kimi K3 megakernel received no typed hybrid step view")
        hidden_states = self.executor.run(input_ids, None, step.view.hybrid)
        return self.language_model.logits_processor(
            input_ids,
            hidden_states,
            self.language_model.lm_head,
            forward_batch,
            None,
        )


def _prepare(built: BuiltMegakernel, view: PreparationView) -> None:
    """Bind typed SGLang-owned pools after allocation, before graph warmup."""
    if view.hybrid is None:
        raise ValueError("Kimi K3 megakernel requires a hybrid MLA/KDA pool")
    cast(_Built, built).resources.bind(view.hybrid)


def _build(model: Any, hf_config: Any, context: BuildContext) -> _Built:
    if context.tokens_per_request != 1:
        raise ValueError("Kimi K3 megakernel serves one-token decode only")
    widths = (
        prepared_widths(context.widths, getattr(context, "capture_widths", None))
        if EXTENDED_PROFILE
        else tuple(width for width in context.widths if width in WIDTHS)
    )
    if not widths:
        raise ValueError(
            f"Kimi K3 megakernel found no supported width in {list(context.widths)}"
        )
    if context.binding is None:
        raise ValueError("Kimi K3 megakernel received no SGLang runtime binding")
    native_mla = NATIVE_MLA
    if EXTENDED_PROFILE and getattr(context.binding, "native_mla_api_version", 0) != 1:
        raise ValueError("extended decode profile requires SGLang native MLA API v1")
    deadline_ns = _deadline_ns(context.deadline_s)
    from sglang.srt.kernel_plugins.nvidia.kimi_k3.compat import (
        validate_runtime_compatibility,
    )

    validate_runtime_compatibility()
    if FULL_DECODE:
        from sglang.srt.kernel_plugins.nvidia.kimi_k3.compat import (
            validate_tokenspeed_compatibility,
        )

        validate_tokenspeed_compatibility()
    from sglang.srt.kernel_plugins.nvidia.kimi_k3.providers import (
        _cuda_memory,
        build_resources,
    )

    if EXTENDED_PROFILE:
        import torch

        from sglang.srt.kernel_plugins.nvidia.binding import megakernel_agree
        from sglang.srt.runtime_context import get_parallel

        megakernel_agree(
            (
                int(get_model().context_length),
                int(get_parallel().attn_dcp_size),
                int(
                    torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    ).multi_processor_count
                ),
            ),
            what="K3 extended decode context/DCP/SM topology",
        )

    resources = build_resources(
        model.language_model,
        context_length=int(get_model().context_length),
        memory_before=_cuda_memory(),
        deadline_ns=deadline_ns,
        **(
            {"native_mla": True, "batches": widths}
            if native_mla
            else {"full_decode": True, "batches": widths}
            if FULL_DECODE
            else {}
        ),
    )
    return _Built(model, resources, widths)


SPEC = MegakernelSpec(
    name=(
        "Kimi K3 packed-MXFP4 (full decode)"
        if FULL_DECODE
        else "Kimi K3 packed-MXFP4 (native MLA)"
        if NATIVE_MLA
        else "Kimi K3 packed-MXFP4"
    ),
    requires=Requirements(
        isas=frozenset({"sm_103"}),
        tp_sizes=frozenset({8}),
        resident_blocks=148 if EXTENDED_PROFILE else 152,
        dtypes=frozenset({"bfloat16"}),
        decode_graph_backends=frozenset({"full"}),
        server_args={
            "kv_cache_dtype": frozenset({"fp8_e4m3"}),
            "moe_runner_backend": frozenset({"flashinfer_mxfp4"}),
            "flashinfer_mxfp4_moe_precision": frozenset({"default"}),
            "prefill_attention_backend": frozenset({"trtllm_mla"}),
            "decode_attention_backend": (
                frozenset({"trtllm_mla", "cutedsl_mla"})
                if EXTENDED_PROFILE
                else frozenset({"trtllm_mla"})
            ),
            "page_size": frozenset({64}),
            "dp_size": frozenset({1}),
            "enable_unified_memory": frozenset({False}),
            "enable_page_major_kv_layout": frozenset({False}),
        },
        hf_config=(
            {"encoder_only": False}
            if EXTENDED_PROFILE
            else {"language_only": True, "encoder_only": False}
        ),
    ),
    features=(
        frozenset({Feature.CONTEXT_PARALLEL, Feature.KV_OFFLOAD})
        if EXTENDED_PROFILE
        else frozenset({Feature.MULTI_NODE_TP})
    ),
    numerics=Numerics(
        kind=NumericsKind.TOLERANCE,
        tolerances={"forward.selected_logprobs.max_abs": 0.5},
    ),
    max_rows=max(WIDTHS),
    build=_build,
    prepare=_prepare,
    api_version=REQUIRED_MEGAKERNEL_API_VERSION,
    model_disable_reasons=_model_disable_reasons,
    config_disable_reasons=_config_disable_reasons,
)
