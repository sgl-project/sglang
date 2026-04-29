# SPDX-License-Identifier: Apache-2.0

"""SRT-owned BAGEL visual feature extractor helpers."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN

from sglang.srt.model_loader.weight_utils import default_weight_loader


class BAGELVisualFeatureMixin:
    """Loads and runs BAGEL visual feature extractors inside the SRT model."""

    def _load_visual_feature_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        *,
        params_dict: dict[str, nn.Parameter],
    ) -> bool:
        if not is_bagel_visual_feature_key(name):
            return False
        if not self.bagel_visual_feature_extractors_loaded:
            return True
        param = params_dict.get(name)
        if param is None:
            return True
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return True

    def _init_visual_feature_extractors(self, config) -> None:
        checkpoint_dir = _bagel_checkpoint_dir_from_config(config)
        enabled = bool(getattr(config, "bagel_enable_visual_feature_extractors", False))
        if checkpoint_dir is None and not enabled:
            return

        vit_config_path = Path(
            getattr(
                config,
                "bagel_vit_config_path",
                str(checkpoint_dir / "vit_config.json") if checkpoint_dir else "",
            )
        )
        ae_path = Path(
            getattr(
                config,
                "bagel_ae_path",
                str(checkpoint_dir / "ae.safetensors") if checkpoint_dir else "",
            )
        )
        if not enabled and not (vit_config_path.exists() and ae_path.exists()):
            return
        if not vit_config_path.exists() or not ae_path.exists():
            raise RuntimeError(
                "BAGEL SRT visual feature extractors require vit_config.json and "
                f"ae.safetensors, got {vit_config_path} and {ae_path}"
            )

        symbols = _import_bagel_visual_loader_symbols()
        vit_config = symbols["SiglipVisionConfig"].from_json_file(str(vit_config_path))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        vae_model, vae_config = symbols["load_ae"](local_path=str(ae_path))
        self.latent_channel = int(
            getattr(vae_config, "z_channels", self.latent_channel)
        )
        self.latent_downsample = (
            int(getattr(vae_config, "downsample", self.latent_downsample))
            * self.latent_patch_size
        )
        expected_patch_dim = self.latent_patch_size**2 * self.latent_channel
        if expected_patch_dim != self.vae2llm.in_features:
            raise RuntimeError(
                "BAGEL VAE latent patch dim does not match SRT vae2llm: "
                f"{expected_patch_dim} vs {self.vae2llm.in_features}"
            )

        self.vae_model = vae_model.eval()
        self.vit_model = symbols["SiglipVisionModel"](vit_config)
        self.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config,
            meta=False,
        )
        self.vit_patch_size = int(getattr(vit_config, "patch_size", 14))
        self.vit_max_num_patch_per_side = int(
            getattr(config, "bagel_vit_max_num_patch_per_side", 70)
        )
        self.vit_hidden_size = int(getattr(vit_config, "hidden_size"))
        connector_act = getattr(config, "bagel_connector_act", "gelu_pytorch_tanh")
        self.connector = BAGELMLPConnector(
            self.vit_hidden_size,
            config.hidden_size,
            connector_act,
        )
        self.vit_pos_embed = BAGEL2DPositionEmbedding(
            self.vit_max_num_patch_per_side,
            config.hidden_size,
        )
        self._move_vae_to_runtime_device()
        self.bagel_visual_feature_extractors_loaded = True

    def _move_vae_to_runtime_device(self) -> None:
        vae_model = getattr(self, "vae_model", None)
        if vae_model is None:
            return
        device = self.vae2llm.weight.device
        if device.type == "meta":
            return
        dtype = self.vae2llm.weight.dtype
        vae_model.to(device=device, dtype=dtype)

    def _require_visual_feature_extractors(self) -> None:
        if not self.bagel_visual_feature_extractors_loaded:
            raise RuntimeError(
                "BAGEL SRT visual feature extractors are not loaded on this model"
            )

    def prepare_vit_images(
        self,
        curr_kvlens,
        curr_rope,
        images,
        transforms,
        new_token_ids,
    ):
        self._require_visual_feature_extractors()
        data_utils = _bagel_data_utils()
        packed_vit_token_indexes = []
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = [], [], []
        packed_text_ids, packed_text_indexes = [], []
        packed_seqlens, packed_position_ids, packed_indexes = [], [], []
        packed_key_value_indexes = []

        query_curr = curr = 0
        newlens, new_rope = [], []
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self._flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.vit_patch_size,
                max_num_patches_per_side=self.vit_max_num_patch_per_side,
            )
            vit_tokens = data_utils.patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(
                range(query_curr, query_curr + num_img_tokens)
            )
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            query_curr += num_img_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(
                packed_vit_token_indexes,
                dtype=torch.long,
            ),
            "packed_position_ids": torch.tensor(
                packed_position_ids,
                dtype=torch.long,
            ),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes,
                dtype=torch.long,
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, newlens, new_rope

    def prepare_vae_images(
        self,
        curr_kvlens,
        curr_rope,
        images,
        transforms,
        new_token_ids,
        timestep=0,
    ):
        self._require_visual_feature_extractors()
        patchified_vae_latent_shapes, packed_vae_position_ids = [], []
        packed_vae_token_indexes = []
        packed_text_ids, packed_text_indexes = [], []
        packed_seqlens, packed_position_ids, packed_indexes = [], [], []
        packed_key_value_indexes = []

        query_curr = curr = 0
        vae_image_tensors = []
        newlens, new_rope = [], []
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_position_ids = self._flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            packed_vae_position_ids.append(vae_position_ids)
            height, width = image_tensor.shape[1:]
            latent_height = height // self.latent_downsample
            latent_width = width // self.latent_downsample
            patchified_vae_latent_shapes.append((latent_height, latent_width))

            num_img_tokens = latent_width * latent_height
            packed_vae_token_indexes.extend(
                range(query_curr, query_curr + num_img_tokens)
            )
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            query_curr += num_img_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in zip(*image_sizes)]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, : image_tensor.shape[1], : image_tensor.shape[2]] = (
                image_tensor
            )

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(
                packed_vae_token_indexes,
                dtype=torch.long,
            ),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(
                packed_position_ids,
                dtype=torch.long,
            ),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes,
                dtype=torch.long,
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, newlens, new_rope

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        self._require_visual_feature_extractors()
        packed_text_ids, packed_text_indexes = [], []
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = (
            [],
            [],
            [],
        )
        packed_position_ids, packed_seqlens, packed_indexes = [], [], []
        packed_key_value_indexes = []

        query_curr = curr = 0
        for (height, width), curr_kvlen, curr_position_id in zip(
            image_sizes,
            curr_kvlens,
            curr_rope,
        ):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids["start_of_image"])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_position_ids = self._flattened_position_ids(
                height,
                width,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            packed_vae_position_ids.append(vae_position_ids)

            latent_height = height // self.latent_downsample
            latent_width = width // self.latent_downsample
            num_image_tokens = latent_height * latent_width
            packed_init_noises.append(
                torch.randn(
                    num_image_tokens,
                    self.latent_channel * self.latent_patch_size**2,
                )
            )
            packed_vae_token_indexes.extend(
                range(query_curr, query_curr + num_image_tokens)
            )
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids["end_of_image"])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(
                packed_vae_token_indexes,
                dtype=torch.long,
            ),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(
                packed_position_ids,
                dtype=torch.long,
            ),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes,
                dtype=torch.long,
            ),
        }
        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        self._require_visual_feature_extractors()
        packed_position_ids, packed_indexes, packed_key_value_indexes = [], [], []

        curr = 0
        for (height, width), curr_kvlen, curr_position_id in zip(
            image_sizes,
            curr_kvlens,
            curr_rope,
        ):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1

            latent_height = height // self.latent_downsample
            latent_width = width // self.latent_downsample
            num_image_tokens = latent_height * latent_width
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(
                packed_position_ids,
                dtype=torch.long,
            ),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(
                packed_indexes,
                dtype=torch.long,
            ),
            "cfg_packed_key_value_indexes": torch.tensor(
                packed_key_value_indexes,
                dtype=torch.long,
            ),
        }
        return generation_input

    def embed_bagel_vae_image(self, generation_input: dict[str, Any]) -> torch.Tensor:
        self._require_visual_feature_extractors()
        packed_sequence = self._bagel_base_text_sequence(generation_input)
        padded_images = generation_input["padded_images"].to(
            device=self.vae2llm.weight.device,
            dtype=self.vae2llm.weight.dtype,
        )
        padded_latent = self.vae_model.encode(padded_images)
        patch_size = self.latent_patch_size
        packed_latents = []
        for latent, (height, width) in zip(
            padded_latent,
            generation_input["patchified_vae_latent_shapes"],
        ):
            latent = latent[
                :,
                : height * patch_size,
                : width * patch_size,
            ].reshape(
                self.latent_channel,
                height,
                patch_size,
                width,
                patch_size,
            )
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(
                -1,
                patch_size * patch_size * self.latent_channel,
            )
            packed_latents.append(latent)
        packed_latent = torch.cat(packed_latents, dim=0)
        packed_latent = self._embed_bagel_latents(
            latent_tokens=packed_latent,
            timestep=generation_input["packed_timesteps"],
            latent_position_ids=generation_input["packed_vae_position_ids"],
        )
        packed_sequence = packed_sequence.to(device=packed_latent.device)
        packed_sequence[
            generation_input["packed_vae_token_indexes"].to(packed_sequence.device)
        ] = packed_latent.to(packed_sequence.dtype)
        return packed_sequence

    def embed_bagel_vit_image(self, generation_input: dict[str, Any]) -> torch.Tensor:
        self._require_visual_feature_extractors()
        packed_sequence = self._bagel_base_text_sequence(generation_input)
        device = self.vae2llm.weight.device
        vit_tokens = generation_input["packed_vit_tokens"].to(
            device=device,
            dtype=self.vae2llm.weight.dtype,
        )
        packed_vit_position_ids = generation_input["packed_vit_position_ids"].to(
            device=device,
            dtype=torch.long,
        )
        vit_token_seqlens = generation_input["vit_token_seqlens"].to(device=device)
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(vit_token_seqlens, dim=0),
            (1, 0),
        ).to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()
        packed_vit_token_embed = self.vit_model(
            packed_pixel_values=vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = self.connector(packed_vit_token_embed)
        packed_vit_token_embed = packed_vit_token_embed + self.vit_pos_embed(
            packed_vit_position_ids
        )
        packed_sequence = packed_sequence.to(device=packed_vit_token_embed.device)
        packed_sequence[
            generation_input["packed_vit_token_indexes"].to(packed_sequence.device)
        ] = packed_vit_token_embed.to(packed_sequence.dtype)
        return packed_sequence

    def decode_bagel_image(
        self,
        latent_tokens: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> Any:
        self._require_visual_feature_extractors()
        from PIL import Image

        height, width = image_shape
        latent_height = height // self.latent_downsample
        latent_width = width // self.latent_downsample
        latent_tokens = latent_tokens.to(
            device=self.vae2llm.weight.device,
            dtype=self.vae2llm.weight.dtype,
        )
        latent = latent_tokens.reshape(
            1,
            latent_height,
            latent_width,
            self.latent_patch_size,
            self.latent_patch_size,
            self.latent_channel,
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1,
            self.latent_channel,
            latent_height * self.latent_patch_size,
            latent_width * self.latent_patch_size,
        )
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        return Image.fromarray(image.to(torch.uint8).cpu().numpy())

    def _bagel_base_text_sequence(
        self,
        generation_input: dict[str, Any],
    ) -> torch.Tensor:
        packed_text_ids = generation_input["packed_text_ids"].to(
            device=self.model.embed_tokens.weight.device,
            dtype=torch.long,
        )
        packed_text_embedding = self.model.embed_tokens(packed_text_ids)
        seq_len = int(generation_input["packed_seqlens"].sum().item())
        packed_sequence = packed_text_embedding.new_zeros(
            (seq_len, self.config.hidden_size)
        )
        packed_sequence[
            generation_input["packed_text_indexes"].to(packed_sequence.device)
        ] = packed_text_embedding
        return packed_sequence

    def _flattened_position_ids(
        self,
        height: int,
        width: int,
        patch_size: int,
        *,
        max_num_patches_per_side: int,
    ) -> torch.Tensor:
        data_utils = _bagel_data_utils()
        getter_name = (
            "get_flattened_position_ids_interpolate"
            if bool(getattr(self.config, "bagel_interpolate_pos", False))
            else "get_flattened_position_ids_extrapolate"
        )
        getter = getattr(data_utils, getter_name)
        return getter(
            height,
            width,
            patch_size,
            max_num_patches_per_side=max_num_patches_per_side,
        )


class BAGEL2DPositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side: int, hidden_size: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(max_num_patch_per_side**2, hidden_size),
            requires_grad=False,
        )

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pos_embed[position_ids]


class BAGELMLPConnector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


def is_bagel_visual_feature_key(name: str) -> bool:
    return name.startswith(
        (
            "connector.",
            "vit_pos_embed.",
            "vit_model.",
        )
    )


def _bagel_checkpoint_dir_from_config(config) -> Path | None:
    checkpoint_dir = getattr(config, "bagel_checkpoint_dir", None)
    if checkpoint_dir:
        return Path(str(checkpoint_dir)).expanduser()
    name_or_path = getattr(config, "_name_or_path", None)
    if not name_or_path:
        return None
    candidate = Path(str(name_or_path)).expanduser()
    if (candidate / "vit_config.json").exists() and (
        candidate / "ae.safetensors"
    ).exists():
        return candidate
    return None


def _import_bagel_visual_loader_symbols() -> dict[str, Any]:
    try:
        autoencoder = importlib.import_module("modeling.autoencoder")
        bagel = importlib.import_module("modeling.bagel")
    except ImportError as exc:
        raise RuntimeError(
            "BAGEL SRT visual feature extractor loading requires the official "
            "BAGEL Python modules on PYTHONPATH"
        ) from exc
    return {
        "load_ae": getattr(autoencoder, "load_ae"),
        "SiglipVisionConfig": getattr(bagel, "SiglipVisionConfig"),
        "SiglipVisionModel": getattr(bagel, "SiglipVisionModel"),
    }


def _bagel_data_utils():
    try:
        return importlib.import_module("data.data_utils")
    except ImportError as exc:
        raise RuntimeError(
            "BAGEL SRT image packing requires official BAGEL data.data_utils "
            "on PYTHONPATH"
        ) from exc
