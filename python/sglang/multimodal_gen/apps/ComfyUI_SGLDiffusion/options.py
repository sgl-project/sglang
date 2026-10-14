# SPDX-License-Identifier: Apache-2.0
"""SGLDOptions: runtime options for the SGLD worker.

The output is a flat dict of ServerArgs kwargs plus the plugin-only keys in
PLUGIN_KEYS, which core/generator.py pops before building ServerArgs. Checks that
need the resolved model live in core/preflight.py.
"""

import math

MODEL_TYPES = [
    "auto-detect",
    "qwen_image",
    "qwen_image_edit",
    "qwen_image21",
    "flux",
    "lumina2",
    "minimax_h3",
]
EXECUTION_MODES = ["eager", "torch_compile", "cuda_graph"]
ATTENTION_BACKENDS = [
    "auto",
    "flash_attention",
    "torch_sdpa",
    "sage_attention2",
    "sage_attention3",
    "sol_attention",
    "subblock_sparse_attn",
    "video_sparse_attn_h3",
    "hybrid_window_attn_h3",
]
MEMORY_MODES = ["resident", "layerwise_offload"]
QUANTIZATIONS = ["none", "fp8", "convrot_int8", "mxfp8"]
# Display names -> SGLang backend names; the rest are passed through unchanged.
_ATTENTION_IDS = {
    "flash_attention": "fa",
    "sage_attention2": "sage_attn",
    "sage_attention3": "sage_attn_3",
    "sol_attention": "sol_attn",
}
# Display names -> SGLANG_DIFFUSION_CONVROT_INT8_BACKEND values.
_INT8_GEMMS = {"auto": "auto", "sglang_jit": "jit", "comfy_kitchen": "comfy_kitchen"}

PLUGIN_KEYS = (
    "model_type",
    "enable_cache_dit",
    "int8_kernel",
)


def _field(kind, default, group, tooltip="", **kwargs):
    return (
        kind,
        {"default": default, "sgld_group": group, "tooltip": tooltip, **kwargs},
    )


def _validate(fields: dict, values: dict, node: str) -> dict:
    unknown = set(values) - fields.keys()
    if unknown:
        raise ValueError(f"Unknown {node} inputs: {sorted(unknown)}")
    v = {name: spec[1]["default"] for name, spec in fields.items()}
    v.update(values)
    for name, (kind, spec) in fields.items():
        value = v[name]
        if isinstance(kind, list):
            if value not in kind:
                raise ValueError(f"{node}.{name}: {value!r} is not one of {kind}")
        elif kind == "INT":
            if (
                type(value) is not int
                or value < spec.get("min", -math.inf)
                or value > spec.get("max", math.inf)
            ):
                raise ValueError(f"{node}.{name}: invalid value {value!r}")
        elif kind == "FLOAT":
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value < spec.get("min", -math.inf)
            ):
                raise ValueError(f"{node}.{name}: invalid value {value!r}")
        elif kind == "BOOLEAN":
            if type(value) is not bool:
                raise ValueError(f"{node}.{name}: invalid value {value!r}")
        elif kind == "STRING":
            if not isinstance(value, str):
                raise ValueError(f"{node}.{name}: expected a string")
            v[name] = value.strip()
    return v


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
        return {
            "required": {},
            "optional": {
                "model_type": _field(
                    MODEL_TYPES,
                    "auto-detect",
                    "basic",
                    "auto-detect reads the checkpoint. FastH3 / VDN-H3 use the "
                    "Native Model Loader.",
                ),
                "num_gpus": _field("INT", 1, "basic", min=1, max=64),
                # ===== parallel =====
                "parallel_mode": _field(
                    ["auto", "manual"],
                    "auto",
                    "parallel",
                    "auto lets SGLang pick the layout for num_gpus; manual sets "
                    "TP x Ulysses x Ring.",
                ),
                "tp_size": _field("INT", 1, "parallel", min=1),
                "ulysses_degree": _field("INT", 1, "parallel", min=1),
                "ring_degree": _field("INT", 1, "parallel", min=1),
                # ===== attention =====
                "attention_backend": _field(
                    ATTENTION_BACKENDS,
                    "auto",
                    "attention",
                    "auto uses the model default. A backend this environment cannot "
                    "run is rejected, never silently replaced.",
                ),
                "attention_backend_config": _field(
                    "STRING",
                    "",
                    "attention",
                    "Same format as --attention-backend-config.",
                ),
                # ===== quantization =====
                "quantization": _field(
                    QUANTIZATIONS,
                    "none",
                    "quantization",
                    "Online quantization at load; native model directories only. "
                    "ComfyUI files carry their own format.",
                ),
                "int8_gemm": _field(
                    list(_INT8_GEMMS),
                    "auto",
                    "quantization",
                    "GEMM implementation for INT8 ConvRot weights (*_int8_convrot "
                    "files or convrot_int8): SGLang's JIT kernel or ComfyUI's "
                    "comfy_kitchen. auto prefers sglang_jit.",
                ),
                # ===== cache =====
                "enable_cache_dit": _field(
                    "BOOLEAN",
                    False,
                    "cache",
                    "Lossy step caching. MiniMax H3 only for now; other models run "
                    "each sampler step as its own request.",
                ),
                # ===== execution =====
                "execution_mode": _field(
                    EXECUTION_MODES,
                    "eager",
                    "execution",
                    "torch_compile and cuda_graph (breakable CUDA graph) add "
                    "first-run work.",
                ),
                "regional_compile": _field(
                    "BOOLEAN", False, "execution", "torch_compile only."
                ),
                # ===== memory =====
                "memory_mode": _field(
                    MEMORY_MODES,
                    "resident",
                    "memory",
                    "DiT only; ComfyUI owns the text encoder and VAE. Layerwise "
                    "offload streams DiT layers from CPU.",
                ),
                "resident_layers": _field(
                    "INT", 0, "memory", "Layers kept on GPU when offloading.", min=0
                ),
                "prefetch_layers": _field(
                    "INT", 0, "memory", "Layers prefetched ahead; 0 means 1.", min=0
                ),
            },
        }

    def create_options(self, **values):
        v = _validate(self.INPUT_TYPES()["optional"], values, "SGLDOptions")
        execution = v["execution_mode"]
        options = {
            "num_gpus": v["num_gpus"],
            "enable_torch_compile": execution == "torch_compile",
            "regional_compile": execution == "torch_compile" and v["regional_compile"],
            "enable_breakable_cuda_graph": execution == "cuda_graph",
            "dit_layerwise_offload": v["memory_mode"] == "layerwise_offload",
            "int8_kernel": _INT8_GEMMS[v["int8_gemm"]],
        }
        if v["model_type"] != "auto-detect":
            options["model_type"] = v["model_type"]
        if v["parallel_mode"] == "manual":
            options.update(_manual_layout(v))
        if options["dit_layerwise_offload"]:
            options["dit_layerwise_resident_layers"] = v["resident_layers"]
            options["dit_offload_prefetch_size"] = v["prefetch_layers"]
        if v["attention_backend"] != "auto":
            options["attention_backend"] = _ATTENTION_IDS.get(
                v["attention_backend"], v["attention_backend"]
            )
        if v["quantization"] != "none":
            options["quantization"] = v["quantization"]
        if v["attention_backend_config"]:
            options["attention_backend_config"] = v["attention_backend_config"]
        if v["enable_cache_dit"]:
            options["enable_cache_dit"] = True
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
