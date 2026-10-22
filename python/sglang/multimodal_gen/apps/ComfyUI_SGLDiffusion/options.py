# SPDX-License-Identifier: Apache-2.0
"""SGLDOptions: runtime options for the SGLD worker.

The output is a flat dict of ServerArgs kwargs plus the plugin-only keys in
PLUGIN_KEYS, which core/generator.py pops before building ServerArgs. Checks that
need the resolved model live in core/preflight.py.
"""

PLUGIN_KEYS = ("model_type", "cache_mode", "cache_dit_params", "int8_kernel")

# Display names -> SGLang backend names; the rest are passed through unchanged.
_ATTENTION_IDS = {
    "flash_attention": "fa",
    "sage_attention2": "sage_attn",
    "sage_attention3": "sage_attn_3",
    "sol_attention": "sol_attn",
}
# Display names -> SGLANG_DIFFUSION_CONVROT_INT8_BACKEND values.
_INT8_GEMMS = {"auto": "auto", "sglang_jit": "jit", "comfy_kitchen": "comfy_kitchen"}


def _field(group, kind, default, tooltip="", **kwargs):
    spec = {"default": default, "sgld_group": group, "tooltip": tooltip, **kwargs}
    return (kind, spec)


# Order is the widget order; web/options.js puts a folding header before each group.
_FIELDS = {
    "model_type": _field(
        "basic",
        ["auto-detect", "qwen_image", "qwen_image_edit", "qwen_image21"]
        + ["flux", "lumina2", "minimax_h3"],
        "auto-detect",
        "FastH3 / VDN-H3 load through the Native Model Loader.",
    ),
    "num_gpus": _field("basic", "INT", 1, min=1, max=64),
    "parallel_mode": _field(
        "parallel", ["auto", "manual"], "auto", "manual sets TP x Ulysses x Ring."
    ),
    "tp_size": _field("parallel", "INT", 1, min=1),
    "ulysses_degree": _field("parallel", "INT", 1, min=1),
    "ring_degree": _field("parallel", "INT", 1, min=1),
    "attention_backend": _field(
        "attention",
        ["auto", "flash_attention", "torch_sdpa", "sage_attention2"]
        + ["sage_attention3", "sol_attention", "subblock_sparse_attn"]
        + ["video_sparse_attn_h3", "hybrid_window_attn_h3"],
        "auto",
        "auto uses the model default; an unavailable backend is rejected.",
    ),
    "attention_backend_config": _field(
        "attention", "STRING", "", "Same format as --attention-backend-config."
    ),
    "quantization": _field(
        "quantization",
        ["none", "fp8", "convrot_int8", "mxfp8"],
        "none",
        "Online quantization: native model directories, or a BF16 Qwen-Image 2.1 "
        "file (convrot_int8).",
    ),
    "int8_gemm": _field(
        "quantization",
        list(_INT8_GEMMS),
        "auto",
        "GEMM kernel for INT8 ConvRot weights; auto prefers sglang_jit.",
    ),
    "cache_mode": _field(
        "cache",
        ["off", "cache_dit", "teacache", "spectrum"],
        "off",
        "Lossy step caching; changing it keeps the worker.",
    ),
    "cache_dit_tuning": _field(
        "cache", ["auto", "manual"], "auto", "manual sets the values below."
    ),
    "cache_warmup_steps": _field("cache", "INT", 4, "Full steps first.", min=0),
    "cache_threshold": _field(
        "cache", "FLOAT", 0.24, "Higher caches more steps.", min=0.0, step=0.01
    ),
    "cache_max_cached_steps": _field("cache", "INT", 3, "Most in a row.", min=0),
    "cache_fn_blocks": _field("cache", "INT", 1, "Leading blocks computed.", min=0),
    "execution_mode": _field(
        "execution", ["eager", "torch_compile", "cuda_graph"], "eager"
    ),
    "regional_compile": _field("execution", "BOOLEAN", False, "torch_compile only."),
    "memory_mode": _field(
        "memory",
        ["resident", "layerwise_offload"],
        "resident",
        "DiT only; ComfyUI owns the text encoder and VAE.",
    ),
    "resident_layers": _field("memory", "INT", 0, "Kept on GPU.", min=0),
    "prefetch_layers": _field("memory", "INT", 0, "Prefetched; 0 means 1.", min=0),
}


class SGLDOptions:
    """Grouped runtime options; web/options.js folds the groups."""

    RETURN_TYPES = ("SGLD_OPTIONS",)
    RETURN_NAMES = ("sgld_options",)
    FUNCTION = "create_options"
    CATEGORY = "SGLDiffusion"
    DESCRIPTION = (
        "Runtime options for the SGLD worker. Combinations the selected model "
        "cannot run are rejected when the loader runs, before the worker starts."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": _FIELDS}

    def create_options(self, **values):
        # ComfyUI has already checked each value against INPUT_TYPES.
        v = {name: spec[1]["default"] for name, spec in _FIELDS.items()} | values
        execution = v["execution_mode"]
        options = {
            "num_gpus": v["num_gpus"],
            "enable_torch_compile": execution == "torch_compile",
            "regional_compile": execution == "torch_compile" and v["regional_compile"],
            "enable_breakable_cuda_graph": execution == "cuda_graph",
            "dit_layerwise_offload": v["memory_mode"] == "layerwise_offload",
            "int8_kernel": _INT8_GEMMS[v["int8_gemm"]],
            "cache_mode": v["cache_mode"],
        }
        if v["model_type"] != "auto-detect":
            options["model_type"] = v["model_type"]
        if v["parallel_mode"] == "manual":
            options.update(_manual_layout(v))
        if options["dit_layerwise_offload"]:
            options["dit_layerwise_resident_layers"] = v["resident_layers"]
            options["dit_offload_prefetch_size"] = v["prefetch_layers"]
        backend = v["attention_backend"]
        if backend != "auto":
            options["attention_backend"] = _ATTENTION_IDS.get(backend, backend)
        if v["attention_backend_config"].strip():
            options["attention_backend_config"] = v["attention_backend_config"].strip()
        if v["quantization"] != "none":
            options["quantization"] = v["quantization"]
        if v["cache_mode"] == "cache_dit" and v["cache_dit_tuning"] == "manual":
            options["cache_dit_params"] = {
                "max_warmup_steps": v["cache_warmup_steps"],
                "residual_diff_threshold": v["cache_threshold"],
                "max_continuous_cached_steps": v["cache_max_cached_steps"],
                "Fn_compute_blocks": v["cache_fn_blocks"],
            }
        return (options,)


def _manual_layout(v: dict) -> dict:
    tp, ulysses, ring = v["tp_size"], v["ulysses_degree"], v["ring_degree"]
    if tp * ulysses * ring != v["num_gpus"]:
        raise ValueError(
            f"num_gpus ({v['num_gpus']}) must equal tp_size * ulysses_degree * "
            f"ring_degree ({tp * ulysses * ring})"
        )
    return {
        "tp_size": tp,
        "sp_degree": ulysses * ring,
        "ulysses_degree": ulysses,
        "ring_degree": ring,
    }
