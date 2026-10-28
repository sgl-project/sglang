# SPDX-License-Identifier: Apache-2.0
"""Reject accelerations this environment cannot run before the SGLD worker starts.

The worker is spawned from this interpreter, so asking SGLang's own backend
resolution here gives the same answer the worker would get after loading the
model; a missing kernel then fails in the loader node instead of at the first
sampling step, or silently as another backend.
"""

from pkgutil import resolve_name

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

_INSTALL_HINTS = {
    AttentionBackendEnum.SAGE_ATTN: "install SageAttention 2: https://github.com/thu-ml/SageAttention",
    AttentionBackendEnum.SOL_ATTN: (
        "install sol-attn: pip install git+https://github.com/NVlabs/Sana.git"
        "@sol-engine#subdirectory=techniques/sparse_backends"
    ),
    AttentionBackendEnum.SAGE_ATTN_3: (
        "SageAttention 3 targets Blackwell GPUs: "
        "https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell"
    ),
}


def requested_attention_backends(sgld_options: dict) -> dict[str, str]:
    """Option name -> requested backend, for every attention backend the options select."""
    requested = {}
    if sgld_options.get("attention_backend"):
        requested["attention_backend"] = sgld_options["attention_backend"]
    components = sgld_options.get("component_attention_backends") or {}
    if isinstance(components, str):
        components = dict(
            item.split("=", 1) for item in components.split(",") if "=" in item
        )
    for component, backend in components.items():
        requested[f"component_attention_backends.{component.strip()}"] = backend
    return requested


def _resolved_backend(backend: AttentionBackendEnum) -> AttentionBackendEnum:
    # Head size and dtype of a typical DiT layer; only availability matters here.
    cls_path = current_platform.get_attn_backend_cls_str(backend, 128, torch.bfloat16)
    return resolve_name(cls_path).get_enum()


def check_attention_backends(sgld_options: dict) -> None:
    for option, name in requested_attention_backends(sgld_options).items():
        try:
            backend = AttentionBackendEnum[name.strip().upper()]
        except KeyError:
            raise ValueError(
                f"{option}={name!r} is not an SGLang attention backend"
            ) from None
        try:
            resolved = _resolved_backend(backend)
        except (ImportError, ValueError) as error:
            raise ValueError(
                f"{option}={name} cannot run in this environment: {error}"
            ) from error
        if resolved is not backend:
            hint = _INSTALL_HINTS.get(backend, "install it or choose another backend")
            raise ValueError(
                f"{option}={name} is not available in this environment (SGLang "
                f"would run {resolved.name.lower()} instead); {hint}"
            )


def check_parallel_layout(sgld_options: dict) -> None:
    dp = sgld_options.get("dp_size") or 1
    if dp > 1:
        # Each sampler step is one request carrying CUDA IPC tensors from this
        # process, and ComfyUI runs one prompt at a time.
        raise ValueError(
            "dp_size > 1 is not supported in ComfyUI integrated mode: replicas never "
            "overlap behind ComfyUI's single prompt queue. Run one ComfyUI instance "
            "per GPU, or use tp_size / sp_degree for one request."
        )
    tp = sgld_options.get("tp_size")
    sp = sgld_options.get("sp_degree")
    if (
        sp is None
        and sgld_options.get("ulysses_degree")
        and sgld_options.get("ring_degree")
    ):
        sp = sgld_options["ulysses_degree"] * sgld_options["ring_degree"]
    if tp is None or sp is None:
        return  # SGLang fills unset degrees from num_gpus
    cfg = (
        (sgld_options.get("cfg_parallel_degree") or 2)
        if sgld_options.get("enable_cfg_parallel")
        else 1
    )
    num_gpus = sgld_options.get("num_gpus") or 1
    if dp * tp * sp * cfg != num_gpus:
        # A GPU outside the layout gets no process group; its worker hangs at startup.
        raise ValueError(
            f"num_gpus ({num_gpus}) must equal tp_size * sp_degree"
            f"{' * cfg_parallel_degree' if cfg > 1 else ''} = {dp * tp * sp * cfg}; "
            "set the degrees to fill every GPU or leave sp_degree at -1 (auto)"
        )


def check_sgld_options(sgld_options: dict) -> None:
    check_attention_backends(sgld_options)
    check_parallel_layout(sgld_options)


# ===== Model-aware rules (run once the loader has resolved the model) =====

H3_MODELS = ("minimax_h3", "fast_h3", "vdn_h3")
_DISTILLED_H3 = ("fast_h3", "vdn_h3")
# Every ComfyUI step is its own request, so a step cache needs a per-run state;
# only the H3 step stage keeps one today.
_CACHE_MODELS = {"cache_dit": H3_MODELS, "spectrum": H3_MODELS, "teacache": ()}
_ATTENTION_BY_MODEL = {
    "generic": ("fa", "torch_sdpa", "sage_attn", "sage_attn_3"),
    "minimax_h3": (
        "fa",
        "torch_sdpa",
        "sage_attn",
        "sage_attn_3",
        "sol_attn",
        "subblock_sparse_attn",
        "video_sparse_attn_h3",
    ),
    "fast_h3": ("fa", "torch_sdpa", "sage_attn", "video_sparse_attn_h3"),
    "vdn_h3": ("hybrid_window_attn_h3",),
}


def _unsupported(feature: str, model_type: str) -> ValueError:
    return ValueError(
        f"{feature} is not supported for {model_type} in ComfyUI integrated mode yet"
    )


def check_model_options(
    model_type: str, sgld_options: dict, *, native_dir: bool
) -> None:
    """Reject combinations the resolved model cannot run, instead of ignoring them."""
    _check_execution(model_type, sgld_options)
    _check_attention_for_model(model_type, sgld_options)
    _check_cache(model_type, sgld_options)
    _check_weights(model_type, sgld_options, native_dir=native_dir)


def _check_execution(model_type: str, opts: dict) -> None:
    graph = opts.get("enable_breakable_cuda_graph")
    if graph:
        if model_type != "minimax_h3":
            raise _unsupported("cuda_graph", model_type)
        if opts.get("cache_mode", "off") != "off":
            raise ValueError("cuda_graph requires cache_mode=off")
        if opts.get("dit_layerwise_offload"):
            raise ValueError("cuda_graph cannot be combined with layerwise offload")
    if model_type in _DISTILLED_H3:
        if opts.get("enable_torch_compile") or graph:
            raise ValueError(f"{model_type} requires execution_mode=eager")
        if (opts.get("ring_degree") or 1) > 1:
            raise ValueError(f"{model_type} requires ring_degree=1; use Ulysses")
    if (
        model_type == "minimax_h3"
        and (opts.get("ring_degree") or 1) > 1
        and opts.get("attention_backend", "fa") != "fa"
    ):
        raise ValueError("MiniMax H3 ring_degree > 1 requires attention_backend=fa")


def _check_attention_for_model(model_type: str, opts: dict) -> None:
    backend = opts.get("attention_backend")
    if backend is None:
        return
    allowed = _ATTENTION_BY_MODEL.get(model_type, _ATTENTION_BY_MODEL["generic"])
    if backend not in allowed:
        raise ValueError(
            f"attention_backend={backend} does not apply to {model_type}; "
            f"choose auto or one of {list(allowed)}"
        )


def _check_cache(model_type: str, opts: dict) -> None:
    mode = opts.get("cache_mode", "off")
    if mode == "off":
        return
    if model_type not in _CACHE_MODELS[mode]:
        raise _unsupported(mode, model_type)
    if mode == "spectrum":
        sp = (opts.get("ulysses_degree") or 1) * (opts.get("ring_degree") or 1)
        if sp > 1 or ("sp_degree" not in opts and (opts.get("num_gpus") or 1) > 1):
            raise ValueError(
                "spectrum requires sequence parallel degree 1; set parallel_mode "
                "manual with ulysses_degree=ring_degree=1"
            )


def _check_weights(model_type: str, opts: dict, *, native_dir: bool) -> None:
    if opts.get("quantization") and not native_dir:
        raise ValueError(
            "quantization (online, at load) applies to native model directories; "
            "ComfyUI files carry their own format, so load a pre-quantized file "
            "(e.g. *_int8_convrot) instead"
        )
    kernel = opts.get("int8_kernel", "auto")
    if kernel != "auto" and model_type not in H3_MODELS:
        raise ValueError(
            f"int8_gemm={kernel} has no effect on {model_type}: only MiniMax H3 "
            "runs ConvRot INT8 weights; leave it at auto"
        )
