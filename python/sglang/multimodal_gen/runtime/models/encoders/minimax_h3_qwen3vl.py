# SPDX-License-Identifier: Apache-2.0
"""Native, TP-foldable Qwen3-VL conditioning encoder for MiniMax H3."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.minimax_h3_qwen3vl import (
    MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
    MiniMaxH3Qwen3VLConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLModel
from sglang.multimodal_gen.runtime.weights.source import (
    materialize_weight,
    resolve_weight,
)

MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120
_LAYER_WEIGHT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_PARAM_NAMES_MAPPING = {
    r"^model\.(embed_tokens|layers|norm|rotary_emb)\.": r"model.language_model.\1.",
    r"^language_model\.": r"model.language_model.",
    r"^visual\.": r"model.visual.",
    r"^(model\.visual\.blocks\.\d+\.attn\.)qkv\.": r"\1qkv_proj.",
}
_MAP_CHECKPOINT_NAME = get_param_names_mapping(_PARAM_NAMES_MAPPING)


def _map_checkpoint_name(name: str) -> str:
    return _MAP_CHECKPOINT_NAME(name)[0]


def _is_unconsumed_checkpoint_weight(name: str, selected_layer: int) -> bool:
    """Weights intentionally absent from the selected feature extractor."""

    if name == "lm_head.weight" or name.startswith("model.language_model.norm."):
        return True
    match = _LAYER_WEIGHT_RE.match(name)
    return bool(match and int(match.group(1)) >= selected_layer)


class _FrozenLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)


class MiniMaxH3ConditioningProjection(nn.Module):
    """Apply the safe ClipProj conditioning format without a custom runtime."""

    _REQUIRED_TENSORS = ("mean_in", "std_in", "mean_out", "std_out")

    @staticmethod
    def inspect(path: str) -> tuple[int, int, int]:
        if not path.endswith(".safetensors"):
            raise ValueError("H3 conditioning projections must use safetensors")
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            try:
                tap = int(metadata["tap"])
                input_dim = int(handle.get_slice("mean_in").get_shape()[0])
                output_dim = int(handle.get_slice("mean_out").get_shape()[0])
            except (IndexError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid H3 conditioning projection metadata in {path!r}"
                ) from error
        return tap, input_dim, output_dim

    def __init__(self, path: str) -> None:
        super().__init__()
        self.tap, self.input_dim, self.output_dim = self.inspect(path)
        tensors = load_file(path, device="cpu")
        missing = set(self._REQUIRED_TENSORS) - set(tensors)
        if missing:
            raise ValueError(
                f"H3 conditioning projection is missing tensors: {sorted(missing)}"
            )

        for name in self._REQUIRED_TENSORS:
            self.register_buffer(name, tensors.pop(name).float())
        expected_shapes = {
            "mean_in": (self.input_dim,),
            "std_in": (self.input_dim,),
            "mean_out": (self.output_dim,),
            "std_out": (self.output_dim,),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(self.get_buffer(name).shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"H3 conditioning projection {name} has shape "
                    f"{actual_shape}, expected {expected_shape}"
                )
        self.register_buffer(
            "weight",
            tensors.pop("W").float() if "W" in tensors else None,
        )
        self.register_buffer(
            "sink_out",
            tensors.pop("sink_out").float() if "sink_out" in tensors else None,
        )

        layer_indices = sorted(
            {
                int(name.split(".")[1])
                for name in tensors
                if re.fullmatch(r"mlp\.\d+\.weight", name)
            }
        )
        layers: list[nn.Module] = []
        layer_input_dim = self.input_dim
        for layer_index in layer_indices:
            weight_name = f"mlp.{layer_index}.weight"
            bias_name = f"mlp.{layer_index}.bias"
            weight = tensors.pop(weight_name)
            bias = tensors.pop(bias_name, None)
            if weight.ndim != 2 or int(weight.shape[1]) != layer_input_dim:
                raise ValueError(
                    f"H3 conditioning projection {weight_name} cannot follow "
                    f"width {layer_input_dim}: got {tuple(weight.shape)}"
                )
            if bias is not None and tuple(bias.shape) != (int(weight.shape[0]),):
                raise ValueError(
                    f"H3 conditioning projection {bias_name} has shape "
                    f"{tuple(bias.shape)}, expected ({int(weight.shape[0])},)"
                )
            layers.append(_FrozenLinear(weight, bias))
            layer_input_dim = int(weight.shape[0])
        if tensors:
            raise ValueError(
                "H3 conditioning projection contains unsupported tensors: "
                f"{sorted(tensors)}"
            )
        if self.weight is None and not layers:
            raise ValueError("H3 conditioning projection has neither W nor an MLP")
        if layers and layer_input_dim != self.output_dim:
            raise ValueError(
                f"H3 conditioning projection MLP outputs width {layer_input_dim}, "
                f"expected {self.output_dim}"
            )
        if self.weight is not None and tuple(self.weight.shape) != (
            self.input_dim,
            self.output_dim,
        ):
            raise ValueError(
                "H3 conditioning projection W has shape "
                f"{tuple(self.weight.shape)}, expected "
                f"({self.input_dim}, {self.output_dim})"
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if int(hidden_states.shape[-1]) != self.input_dim:
            raise ValueError(
                f"H3 conditioning projection expects width {self.input_dim}, "
                f"got {int(hidden_states.shape[-1])}"
            )
        normalized = (hidden_states.float() - self.mean_in) / self.std_in
        projected = normalized @ self.weight if self.weight is not None else None
        if self.layers:
            residual = normalized.to(self.layers[0].weight.dtype)
            for index, layer in enumerate(self.layers):
                residual = layer(residual)
                if index + 1 < len(self.layers):
                    residual = F.gelu(residual)
            residual = residual.float()
            projected = residual if projected is None else projected + residual
        if projected is None:
            raise RuntimeError("H3 conditioning projection produced no output")
        output = projected * self.std_out + self.mean_out
        if self.sink_out is not None and int(output.shape[-2]) > 0:
            output[..., 0, :] = self.sink_out
        return output


class MiniMaxH3Qwen3VLEncoder(TextEncoder):
    """Qwen3-VL multimodal backbone producing MiniMax H3 conditioning.

    The component loader builds and loads this module under the encoder-folding
    TP group. A TP=1/SP=8 DiT deployment therefore shards the encoder over all
    eight otherwise-idle ranks during encoding.
    """

    # The inherited text-layer list covers Qwen's language stack; reference
    # modes also execute the embedded visual tower.
    layer_names = [
        *TextEncoder.layer_names,
        "model.visual.blocks",
        "conditioning_projection.layers",
    ]

    supports_dp_encode = True
    param_names_mapping = _PARAM_NAMES_MAPPING
    # Comfy packs the vision tower across whole tensors rather than rows. Keep
    # its language/vocabulary matrices packed and restore this smaller tower.
    gguf_dequantize_prefixes = ("visual.", "model.visual.")

    @classmethod
    def configure_component_paths(
        cls,
        config: MiniMaxH3Qwen3VLConfig,
        component_paths: dict[str, str],
    ) -> None:
        arch = config.arch_config
        source = component_paths.get("conditioning_projection")
        if source is None:
            if (
                int(arch.hidden_size) != MINIMAX_H3_QWEN3VL_HIDDEN_DIM
                or int(arch.checkpoint_num_hidden_layers)
                < MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
            ):
                raise ValueError(
                    "MiniMax H3 Qwen3-VL encoders smaller than 32B require "
                    "--component-paths.conditioning_projection"
                )
            return

        projection_path = materialize_weight(resolve_weight(source))
        tap, input_dim, output_dim = MiniMaxH3ConditioningProjection.inspect(
            projection_path
        )
        if input_dim != int(arch.hidden_size):
            raise ValueError(
                f"H3 conditioning projection expects encoder width {input_dim}, "
                f"but the selected text encoder has width {int(arch.hidden_size)}"
            )
        if output_dim != MINIMAX_H3_QWEN3VL_HIDDEN_DIM:
            raise ValueError(
                f"H3 conditioning projection must output width "
                f"{MINIMAX_H3_QWEN3VL_HIDDEN_DIM}, got {output_dim}"
            )
        if tap <= 0 or tap > int(arch.checkpoint_num_hidden_layers):
            raise ValueError(
                f"H3 conditioning projection tap {tap} is outside the selected "
                f"encoder's {int(arch.checkpoint_num_hidden_layers)} layers"
            )
        arch.conditioning_projection_path = projection_path
        arch.num_hidden_layers = tap
        arch.text_config.num_hidden_layers = tap

    def should_materialize_checkpoint_weight(self, name: str) -> bool:
        name = _map_checkpoint_name(name)
        return (
            "rotary_emb.inv_freq" not in name
            and not _is_unconsumed_checkpoint_weight(name, self.selected_lm_layer)
        )

    def __init__(self, config: MiniMaxH3Qwen3VLConfig) -> None:
        super().__init__(config)
        arch = config.arch_config
        selected_layer = int(arch.text_config.num_hidden_layers)
        if selected_layer <= 0 or int(arch.num_hidden_layers) != selected_layer:
            raise ValueError(
                "MiniMax H3 Qwen3-VL language-layer configuration is "
                f"inconsistent: {selected_layer} vs {int(arch.num_hidden_layers)}"
            )
        self.model = Qwen3VLModel(
            arch,
            quant_config=config.quant_config,
            use_tensor_parallel=True,
            prefix="model",
        )
        # H3 and ClipProj consume an unnormalized intermediate residual stream.
        self.model.language_model.norm = nn.Identity()
        self.image_token_id = int(arch.image_token_id)
        self.video_token_id = int(arch.video_token_id)
        self.selected_lm_layer = selected_layer
        self.hidden_dim = MINIMAX_H3_QWEN3VL_HIDDEN_DIM
        self.conditioning_projection = (
            MiniMaxH3ConditioningProjection(arch.conditioning_projection_path)
            if arch.conditioning_projection_path is not None
            else None
        )

    @property
    def device(self) -> torch.device:
        """Device this encoder's forward runs on.

        Deliberately not `next(self.parameters()).device`. `--text-encoder-cpu-offload`
        loads this component under an FSDP CPU offload policy, which keeps the sharded
        parameters on CPU and all-gathers them to the accelerator for the forward. The
        parameter device then names the storage side, not the compute side, so inputs
        built from it stay on CPU while the forward runs on the accelerator.
        """
        return get_local_torch_device()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> BaseEncoderOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )
        return BaseEncoderOutput(last_hidden_state=outputs.last_hidden_state)

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1-D, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be given together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be given together"
            )

        host_ids = input_ids.to(device="cpu", dtype=torch.long)[None]
        host_image_grid_thw = (
            image_grid_thw.to(device="cpu", dtype=torch.long)
            if image_grid_thw is not None
            else None
        )
        host_video_grid_thw = (
            video_grid_thw.to(device="cpu", dtype=torch.long)
            if video_grid_thw is not None
            else None
        )
        position_ids = None
        if host_image_grid_thw is not None or host_video_grid_thw is not None:
            position_ids, _ = self.model.get_rope_index(
                host_ids,
                host_image_grid_thw,
                host_video_grid_thw,
                attention_mask=torch.ones_like(host_ids),
            )
        ids = host_ids.to(self.device)
        call_kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }
        if position_ids is not None:
            call_kwargs["position_ids"] = position_ids.to(self.device)
        if pixel_values is not None:
            call_kwargs["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            call_kwargs["image_grid_thw"] = host_image_grid_thw
        if pixel_values_videos is not None:
            call_kwargs["pixel_values_videos"] = pixel_values_videos.to(
                self.device, torch.bfloat16
            )
            call_kwargs["video_grid_thw"] = host_video_grid_thw

        hidden = self(**call_kwargs).last_hidden_state[0]
        if self.conditioning_projection is not None:
            hidden = self.conditioning_projection(hidden)
        hidden = hidden.to(torch.bfloat16)
        expected_shape = [int(ids.shape[1]), self.hidden_dim]
        if list(hidden.shape) != expected_shape:
            raise ValueError(
                f"unexpected hidden shape {list(hidden.shape)}, "
                f"expected {expected_shape}"
            )
        return hidden

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            name = _map_checkpoint_name(name)
            if not self.should_materialize_checkpoint_weight(name):
                continue
            param_name = name
            param = params.get(param_name)
            if param is None:
                raise KeyError(
                    "Unexpected MiniMax H3 Qwen3-VL checkpoint weight: "
                    f"{name} (mapped to {param_name})"
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                can_keep_checkpoint_tensor = bool(
                    getattr(self, "_keep_checkpoint_mapping", False)
                    and weight_loader is default_weight_loader
                    and param.device.type == "cpu"
                    and loaded_weight.device.type == "cpu"
                    and loaded_weight.dtype == param.dtype
                    and tuple(loaded_weight.shape) == tuple(param.shape)
                )
                if can_keep_checkpoint_tensor:
                    param.data = loaded_weight
                else:
                    weight_loader(param, loaded_weight.to(param.dtype))
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load MiniMax H3 Qwen3-VL weight "
                    f"{name!r}: checkpoint={tuple(loaded_weight.shape)}, "
                    f"parameter={tuple(param.shape)}"
                ) from exc
            loaded.add(param_name)
        return loaded


EntryClass = MiniMaxH3Qwen3VLEncoder

__all__ = ["MiniMaxH3Qwen3VLEncoder"]
