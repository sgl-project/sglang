# SPDX-License-Identifier: Apache-2.0
"""ComfyUI LTX-2.x audio-video (``ltxav``) checkpoint spec.

ComfyUI ships LTX-2.5 as a DiT-only file and LTX-2.3 as an all-in-one file
(DiT + VAEs + vocoder + text projection). Both keep the DiT under
``model.diffusion_model.`` with upstream ltx-core names and describe the
architecture in the safetensors ``config`` metadata. Comfy per-layer INT8 /
FP8 markers are honoured. The two text connectors stored next to the DiT are
loaded separately into SGLang's ``LTX2ConnectorTransformer1d``.
"""

from __future__ import annotations

import json
from typing import Any

import torch
from safetensors import safe_open

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config, LTX2RopeType
from sglang.multimodal_gen.configs.models.dits.ltx_2_5 import (
    LTX25ArchConfig,
    LTX25Config,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec import (
    ComfyUICheckpointSpec,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    inspect_comfy_quant_markers,
)

PREFIX = "model.diffusion_model."
CONNECTORS = ("video_embeddings_connector", "audio_embeddings_connector")

# metadata["config"]["transformer"] keys copied onto LTX2ArchConfig as-is.
_METADATA_FIELDS = (
    "num_attention_heads",
    "attention_head_dim",
    "in_channels",
    "out_channels",
    "cross_attention_dim",
    "caption_channels",
    "norm_eps",
    "positional_embedding_theta",
    "positional_embedding_max_pos",
    "timestep_scale_multiplier",
    "use_middle_indices_grid",
    "audio_num_attention_heads",
    "audio_attention_head_dim",
    "audio_out_channels",
    "audio_cross_attention_dim",
    "audio_positional_embedding_max_pos",
    "apply_gated_attention",
    "cross_attention_adaln",
    "caption_proj_before_connector",
)


def is_comfyui_ltx_dit_key(name: str) -> bool:
    """DiT weights only: no VAE / vocoder / connectors, no Comfy quant markers."""
    return (
        name.startswith(PREFIX)
        and not name[len(PREFIX) :].startswith(CONNECTORS)
        and not name.endswith(".comfy_quant")
    )


def _read(path: str) -> tuple[dict[str, Any], set[str]]:
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
        keys = set(checkpoint.keys())
    config = json.loads(metadata["config"]) if "config" in metadata else {}
    return dict(config.get("transformer", {})), keys


def _build_dit_config(server_args: ServerArgs) -> LTX2Config:
    # The path-detected pipeline config may be LTX-2 / 2.3 / 2.5; the file
    # metadata is authoritative, so start from the full audio-video arch.
    transformer, keys = _read(server_args.model_path)
    if f"{PREFIX}audio_adaln_single.linear.weight" not in keys:
        raise ValueError(
            f"{server_args.model_path} is not a ComfyUI LTX audio-video (ltxav) "
            "checkpoint: audio_adaln_single is missing"
        )
    dit_config = LTX25Config()
    arch = dit_config.arch_config
    for name in _METADATA_FIELDS:
        if name in transformer:
            setattr(arch, name, transformer[name])
    if "rope_type" in transformer:
        arch.rope_type = LTX2RopeType(transformer["rope_type"])
    if "frequencies_precision" in transformer:
        arch.double_precision_rope = transformer["frequencies_precision"] == "float64"
    block = f"{PREFIX}transformer_blocks.0."
    # LTX-2.5 drops the video FF bias; the metadata does not always say so.
    arch.ff_bias = f"{block}ff.net.0.proj.bias" in keys
    arch.audio_ff_bias = f"{block}audio_ff.net.0.proj.bias" in keys
    arch.use_keyframes_abs_pos_embedding = (
        f"{PREFIX}keyframes_abs_pos_embedding" in keys
    )
    arch.num_layers = 1 + max(
        int(key.split(".")[3])
        for key in keys
        if key.startswith(f"{PREFIX}transformer_blocks.")
    )
    arch.__post_init__()
    server_args.pipeline_config.dit_config = dit_config
    return dit_config


def comfyui_ltx_quant_markers(paths: list[str]) -> dict[str, dict[str, Any]]:
    """Per-layer Comfy markers (INT8 ConvRot / FP8) of the DiT, by SGLang prefix."""
    name_fn = get_param_names_mapping(LTX25ArchConfig().param_names_mapping)
    return {
        name_fn(f"{prefix}.weight")[0].removesuffix(".weight"): marker
        for prefix, marker in inspect_comfy_quant_markers(paths).items()
        if is_comfyui_ltx_dit_key(prefix)
    }


def _regular_hadamard(size: int, device: torch.device) -> torch.Tensor:
    """Normalized regular Hadamard (Kronecker powers of H4) used by Comfy ConvRot."""
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
        device=device,
    )
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / size**0.5


def _dequantize(
    weight: torch.Tensor, scale: torch.Tensor | None, group: int
) -> torch.Tensor:
    """fp32 weight of a Comfy-quantized (FP8 / INT8 ConvRot) linear."""
    weight = weight.float() * scale.float()
    if group:
        out_features, in_features = weight.shape
        hadamard = _regular_hadamard(group, weight.device)
        # Stored as W @ H^T per input group; H is symmetric and orthonormal.
        weight = (weight.view(out_features, -1, group) @ hadamard).view(
            out_features, in_features
        )
    return weight


class _StoredQuantLinear(torch.nn.Module):
    """Linear that keeps Comfy's quantized weight and dequantizes per call.

    The connectors run once per prompt; keeping the stored form saves the
    memory a bf16 copy would take on the worker.
    """

    def __init__(self, weight, scale, bias, group: int) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scale)
        self.bias = torch.nn.Parameter(bias, requires_grad=False)
        self.group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = _dequantize(self.weight, self.weight_scale, self.group)
        return torch.nn.functional.linear(x, weight.to(x.dtype), self.bias)


def load_comfyui_ltx_connectors(
    path: str, device: torch.device, dtype: torch.dtype
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """The (video, audio) text connectors of a ComfyUI LTX file as SGLang modules."""
    from sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector import (
        LTX2ConnectorTransformer1d,
    )

    transformer, keys = _read(path)
    max_pos = transformer.get("connector_positional_embedding_max_pos", [4096])
    modules = []
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for name in CONNECTORS:
            stem = f"{PREFIX}{name}."
            gate = f"{stem}transformer_1d_blocks.0.attn1.to_gate_logits.weight"
            if gate not in keys:
                raise NotImplementedError(
                    "Only gated-attention LTX connectors (LTX-2.3 / 2.5) are "
                    f"supported in integrated mode; {path} has none"
                )
            heads, dim = checkpoint.get_slice(gate).get_shape()
            registers = checkpoint.get_slice(f"{stem}learnable_registers").get_shape()
            with torch.device("meta"):
                connector = LTX2ConnectorTransformer1d(
                    num_attention_heads=heads,
                    attention_head_dim=dim // heads,
                    num_layers=1
                    + max(
                        int(key[len(stem) :].split(".")[1])
                        for key in keys
                        if key.startswith(f"{stem}transformer_1d_blocks.")
                    ),
                    num_learnable_registers=registers[0],
                    rope_base_seq_len=int(max_pos[0]),
                    rope_theta=float(
                        transformer.get("positional_embedding_theta", 1e4)
                    ),
                    rope_type=transformer.get("rope_type", "split"),
                    apply_gated_attention=True,
                )
            state = {}
            for key in keys:
                if not key.startswith(stem) or key.endswith(
                    (".comfy_quant", ".weight_scale", ".input_scale")
                ):
                    continue
                target = (
                    key[len(stem) :]
                    .replace("transformer_1d_blocks.", "transformer_blocks.")
                    .replace(".q_norm.", ".norm_q.")
                    .replace(".k_norm.", ".norm_k.")
                )
                state[target] = checkpoint.get_tensor(key).to(device)
            for linear in [n for n, _ in connector.named_modules()]:
                weight = state.get(f"{linear}.weight")
                if weight is None or weight.is_floating_point() and weight.itemsize > 1:
                    continue
                # FP8 / INT8 weight: keep the stored form (see _StoredQuantLinear).
                raw = f"{stem}{linear}".replace(
                    "transformer_blocks.", "transformer_1d_blocks."
                )
                marker = {}
                if f"{raw}.comfy_quant" in keys:
                    marker = json.loads(
                        bytes(checkpoint.get_tensor(f"{raw}.comfy_quant").tolist())
                    )
                parent, _, child = linear.rpartition(".")
                setattr(
                    connector.get_submodule(parent),
                    child,
                    _StoredQuantLinear(
                        state.pop(f"{linear}.weight"),
                        checkpoint.get_tensor(f"{raw}.weight_scale").to(device),
                        state.pop(f"{linear}.bias").to(dtype),
                        (
                            int(marker["convrot_groupsize"])
                            if marker.get("convrot")
                            else 0
                        ),
                    ),
                )
            state = {k: v.to(dtype) for k, v in state.items()}
            connector.load_state_dict(state, strict=False, assign=True)
            if any(p.is_meta for p in connector.parameters()):
                raise ValueError(f"{path}: incomplete {name} weights")
            modules.append(connector.eval())
    return modules[0], modules[1]


register_comfyui_checkpoint(
    "LTX2Pipeline",
    ComfyUICheckpointSpec(
        dit_cls_name="LTX2VideoTransformer3DModel",
        build_dit_config=_build_dit_config,
        # INT8 ConvRot (LTX-2.5) or static-scale FP8 (LTX-2.3) linears.
        quant_markers=comfyui_ltx_quant_markers,
        checkpoint_key_filter=is_comfyui_ltx_dit_key,
    ),
)
