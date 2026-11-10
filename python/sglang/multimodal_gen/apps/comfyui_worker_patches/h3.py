# SPDX-License-Identifier: Apache-2.0
"""ComfyUI H3 BF16 compatibility implemented outside runtime model sources.

Module instances are patched once after load, before first forward. Cloning a
function's globals keeps helper overrides local to that bound method; original
classes, module globals and other instances remain unchanged. Eager only.
"""

import hashlib
import importlib.metadata
import inspect
import types
from pathlib import Path

import torch

# A changed upstream implementation must be reviewed, never best-effort patched.
SOURCE_SHA256 = "ced7b679ee05f75045646cb81a436088557b90f23d3be980aba13f816ac23f0d"


def bind_with_globals(instance, name, overrides):
    original = getattr(instance, name).__func__
    scope = dict(original.__globals__)
    scope.update(overrides)
    fn = types.FunctionType(
        original.__code__,
        scope,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    fn.__kwdefaults__ = original.__kwdefaults__
    fn.__annotations__ = original.__annotations__
    setattr(instance, name, types.MethodType(fn, instance))


def comfy_gate(x, gate, other, indices, *, dtype, allow_inplace=True):
    return torch.addcmul(
        x, other, gate.index_select(0, indices), out=x if allow_inplace else None
    )


def comfy_rope(q, k, qw, kw, cache, positions, **kwargs):
    # Match ComfyUI Kitchen v0.2.36 HasRms float rotation exactly.
    import comfy_kitchen as ck

    table = cache.index_select(0, positions)
    cos, sin = table.chunk(2, dim=-1)
    table = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(
        1, len(q), 1, cos.shape[-1], 2, 2
    )
    ck.rms_rope_split_half_(
        q.unsqueeze(0),
        k.unsqueeze(0),
        table,
        qw,
        kw,
        epsilon=kwargs["eps"],
        rot_dim=cache.shape[-1],
    )


def _reject_lora(*args, **kwargs):
    raise ValueError("h3_comfyui_v1 does not support LoRA")


def validate(model_type, options):
    if model_type != "minimax_h3":
        raise ValueError("h3_comfyui_v1 requires model_type=minimax_h3")
    if options.get("num_gpus", 1) != 1:
        raise ValueError("h3_comfyui_v1 currently requires one GPU")
    for degree in ("tp_size", "sp_degree", "ulysses_degree", "ring_degree"):
        if options.get(degree) not in (None, -1, 1):
            raise ValueError(f"h3_comfyui_v1 requires {degree}=1 or auto")
    if (
        options.get("enable_torch_compile")
        or options.get("enable_cuda_graph")
        or options.get("enable_breakable_cuda_graph")
    ):
        raise ValueError("h3_comfyui_v1 currently requires eager execution")
    if options.get("cache_mode", "off") != "off" or options.get("quantization"):
        raise ValueError("h3_comfyui_v1 requires no cache/quantization")
    if options.get("attention_backend") != "torch_sdpa":
        raise ValueError("h3_comfyui_v1 parity is verified only with torch_sdpa")
    if (
        options.get("transformer_weights_path")
        or options.get("dit_layerwise_offload")
        or options.get("dit_cpu_offload")
    ):
        raise ValueError("h3_comfyui_v1 requires resident single-file BF16 weights")
    for key in (
        "component_attention_backends",
        "component_quantizations",
        "component_precisions",
        "lora_path",
        "minimax_h3_adaln_online",
        "minimax_h3_adaln_cache_path",
        "enable_cfg_parallel",
        "use_fsdp_inference",
    ):
        if options.get(key):
            raise ValueError(f"h3_comfyui_v1 has not been validated with {key}")


def install(pipeline):
    from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_ring_ctx,
        get_ulysses_ctx,
    )
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3

    source = Path(inspect.getsourcefile(h3)).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError(
            "H3 upstream source differs from the reviewed h3_comfyui_v1 source"
        )
    if importlib.metadata.version("comfy-kitchen") != "0.2.36":
        raise RuntimeError("h3_comfyui_v1 requires reviewed comfy-kitchen==0.2.36")
    if get_tp_world_size() != 1 or get_ring_ctx()[0] != 1 or get_ulysses_ctx()[0] != 1:
        raise ValueError("h3_comfyui_v1 is single-GPU only")
    models = [m for m in pipeline.modules.values() if type(m) is h3.MiniMaxH3DiTModel]
    if len(models) != 1:
        raise ValueError("h3_comfyui_v1 requires exactly one base H3 DiT")
    model = models[0]
    if next(model.parameters()).device.type != "cuda":
        raise ValueError("h3_comfyui_v1 requires a resident CUDA model")
    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("h3_comfyui_v1 is currently verified only on Hopper")
    if (
        model.adaln_t_table is not None
        or model.adaln_cache is not None
        or model.final_layer._pdd_heads is not None
    ):
        raise ValueError("h3_comfyui_v1 does not support PDD/curve/cached AdaLN")
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if method is not None and type(method).__name__ != "UnquantizedLinearMethod":
            raise ValueError(
                f"h3_comfyui_v1 does not support quantized layer {name}: {type(method).__name__}"
            )
    if getattr(model, "_comfy_h3_installed", False):
        return
    # Validate every dependency before modifying any method.
    import comfy_kitchen as ck

    if not callable(ck.rms_rope_split_half_):
        raise TypeError("Missing ComfyUI-compatible RMSNorm/RoPE kernel")
    for block in model.blocks:
        if not callable(getattr(block.attn, "_attend", None)):
            raise TypeError("Missing reviewed H3 attention entry point")
    embed = model._embed

    def embedding(self, *a, **kw):
        hidden, time = embed(*a, **kw)
        return hidden, time.to(torch.bfloat16)

    model._embed = types.MethodType(embedding, model)
    for block in model.blocks:
        bind_with_globals(
            block,
            "forward",
            {
                "_modulate_gate": comfy_gate,
                "minimax_h3_norm_modulate_active": lambda _: False,
            },
        )
        bind_with_globals(
            block.attn, "_attend", {"fused_inplace_qknorm_rope": comfy_rope}
        )
        block.attn._use_fused_qknorm_rope = True
    final = model.final_layer

    def project(self, hidden):
        vi, ai = self._comfy_output_positions
        vi, ai = vi.to(hidden.device), ai.to(hidden.device)
        video, _ = self.video_out(hidden.index_select(0, vi))
        audio, _ = self.audio_out(hidden.index_select(0, ai))
        return (
            hidden.new_zeros((len(hidden), video.shape[-1])).index_copy_(0, vi, video),
            hidden.new_zeros((len(hidden), audio.shape[-1])).index_copy_(0, ai, audio),
        )

    final._project = types.MethodType(project, final)
    forward = model.forward

    def model_forward(self, **kw):
        final._comfy_output_positions = (
            self._pos_ids(
                kw["img_pos_for_infer_output_info"], "img_pos_for_infer_output_info"
            ),
            self._pos_ids(kw["audio_pos_info"], "audio_pos_info"),
        )
        try:
            return forward(**kw)
        finally:
            del final._comfy_output_positions

    model.forward = types.MethodType(model_forward, model)
    pipeline.set_lora = _reject_lora
    model._comfy_h3_installed = True
