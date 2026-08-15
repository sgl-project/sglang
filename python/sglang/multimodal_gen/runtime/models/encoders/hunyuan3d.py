# Copied and adapted from: https://github.com/Tencent-Hunyuan/Hunyuan3D-2

import glob
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from torchvision import transforms
from transformers.models.clip.configuration_clip import (
    CLIPVisionConfig as TransformersCLIPVisionConfig,
)
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config

from sglang.multimodal_gen.configs.models.encoders.clip import (
    CLIPVisionArchConfig,
    CLIPVisionConfig,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.encoders.clip import (
    CLIPVisionModel as NativeCLIPVisionModel,
)
from sglang.multimodal_gen.runtime.models.encoders.dinov2 import (
    Dinov2Model as NativeDinov2Model,
)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


class ImageEncoder(nn.Module, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    layer_names = [
        "model.encoder.layer",
        "model.vision_model.encoder.layers",
    ]
    MODEL_CLASS: type[nn.Module]
    MODEL_CONFIG_CLASS: type
    mean = []
    std = []

    def __init__(
        self,
        version=None,
        config=None,
        use_cls_token=True,
        image_size=224,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            if not version:
                raise ValueError("Image encoder requires either version or config")
            source_config = self.MODEL_CONFIG_CLASS.from_pretrained(version)
        else:
            source_config = self.MODEL_CONFIG_CLASS.from_dict(config)
        self.model = self._build_model(source_config)
        if config is None:
            self._load_pretrained_weights(version)
        self.model.eval()
        self.model.requires_grad_(False)
        self.use_cls_token = use_cls_token
        patch_size = self.model.config.patch_size
        if not isinstance(patch_size, int):
            patch_size = patch_size[0]
        self.size = image_size // patch_size
        self.num_patches = self.size**2
        if self.use_cls_token:
            self.num_patches += 1

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, transforms.InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.CenterCrop(image_size),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

    @classmethod
    def _build_model(cls, config: Any) -> nn.Module:
        return cls.MODEL_CLASS(config)

    def _load_pretrained_weights(self, version: str) -> None:
        local_path = snapshot_download(
            repo_id=version,
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )
        safetensors_files = sorted(glob.glob(os.path.join(local_path, "*.safetensors")))
        if safetensors_files:
            weights = safetensors_weights_iterator(safetensors_files)
        else:
            pt_files = sorted(glob.glob(os.path.join(local_path, "*.bin")))
            if not pt_files:
                raise FileNotFoundError(
                    f"No safetensors or PyTorch weights found for {version}"
                )
            weights = pt_weights_iterator(pt_files)
        self.load_weights((f"model.{name}", tensor) for name, tensor in weights)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        def model_weights():
            for name, tensor in weights:
                if not name.startswith("model."):
                    raise ValueError(f"Unexpected image encoder weight: {name}")
                yield name.removeprefix("model."), tensor

        loaded = self.model.load_weights(model_weights())
        expected = set(dict(self.model.named_parameters()))
        missing = expected - loaded
        if missing:
            examples = sorted(missing)[:8]
            raise RuntimeError(
                f"Image encoder checkpoint is missing {len(missing)} parameters: "
                f"{examples}"
            )
        return {f"model.{name}" for name in loaded}

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def forward(self, image, mask=None, value_range=(-1, 1), **kwargs):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.device, dtype=self.dtype)
        inputs = self.transform(image)
        outputs = self.model(inputs)

        last_hidden_state = outputs.last_hidden_state
        if not self.use_cls_token:
            last_hidden_state = last_hidden_state[:, 1:, :]

        return last_hidden_state

    def unconditional_embedding(self, batch_size, **kwargs):
        zero = torch.zeros(
            batch_size,
            self.num_patches,
            self.model.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )

        return zero


class CLIPImageEncoder(ImageEncoder):
    MODEL_CLASS = NativeCLIPVisionModel
    MODEL_CONFIG_CLASS = TransformersCLIPVisionConfig
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    @classmethod
    def _build_model(cls, config: TransformersCLIPVisionConfig) -> nn.Module:
        arch_config = CLIPVisionArchConfig()
        for name, value in config.to_dict().items():
            setattr(arch_config, name, value)
        native_config = CLIPVisionConfig(
            arch_config=arch_config,
            require_post_norm=False,
            prefix="clip",
        )
        return cls.MODEL_CLASS(native_config)


class DinoImageEncoder(ImageEncoder):
    MODEL_CLASS = NativeDinov2Model
    MODEL_CONFIG_CLASS = Dinov2Config
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


class DinoImageEncoderMV(DinoImageEncoder):
    _aliases = [
        "hy3dshape.models.conditioner.DinoImageEncoderMV",
    ]

    def __init__(
        self,
        version=None,
        config=None,
        use_cls_token=True,
        image_size=224,
        view_num=4,
        **kwargs,
    ):
        super().__init__(version, config, use_cls_token, image_size, **kwargs)
        self.view_num = view_num
        pos = np.arange(self.view_num, dtype=np.float32)
        view_embedding = torch.from_numpy(
            get_1d_sincos_pos_embed_from_grid(self.model.config.hidden_size, pos)
        ).float()

        view_embedding = view_embedding.unsqueeze(1).repeat(1, self.num_patches, 1)
        self.register_buffer(
            "view_embed",
            view_embedding.unsqueeze(0),
            persistent=False,
        )

    def forward(self, image, mask=None, value_range=(-1, 1), view_idxs=None, **kwargs):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.device, dtype=self.dtype)

        bs, num_views, c, h, w = image.shape
        image = image.view(bs * num_views, c, h, w)

        inputs = self.transform(image)
        outputs = self.model(inputs)

        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = last_hidden_state.view(
            bs, num_views, last_hidden_state.shape[-2], last_hidden_state.shape[-1]
        )

        view_embedding = self.view_embed.to(last_hidden_state.dtype).to(
            last_hidden_state.device
        )
        if view_idxs is not None:
            assert len(view_idxs) == bs
            view_embeddings = []
            for i in range(bs):
                view_idx = view_idxs[i]
                assert num_views == len(view_idx)
                view_embeddings.append(self.view_embed[:, view_idx, ...])
            view_embedding = (
                torch.cat(view_embeddings, 0)
                .to(last_hidden_state.dtype)
                .to(last_hidden_state.device)
            )

        if num_views != self.view_num:
            view_embedding = view_embedding[:, :num_views, ...]
        last_hidden_state = last_hidden_state + view_embedding
        last_hidden_state = last_hidden_state.view(
            bs, num_views * last_hidden_state.shape[-2], last_hidden_state.shape[-1]
        )
        return last_hidden_state

    def unconditional_embedding(self, batch_size, view_idxs, **kwargs):
        zero = torch.zeros(
            batch_size,
            self.num_patches * len(view_idxs[0]),
            self.model.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        return zero


def build_image_encoder(config):
    encoder_classes = {
        "CLIPImageEncoder": CLIPImageEncoder,
        "DinoImageEncoder": DinoImageEncoder,
        "DinoImageEncoderMV": DinoImageEncoderMV,
    }
    encoder_type = config["type"]
    try:
        encoder_cls = encoder_classes[encoder_type]
    except KeyError as exc:
        raise ValueError(f"Unknown image encoder type: {encoder_type}") from exc
    return encoder_cls(**config["kwargs"])


class DualImageEncoder(nn.Module, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    layer_names = [
        "main_image_encoder.model.encoder.layer",
        "main_image_encoder.model.vision_model.encoder.layers",
        "additional_image_encoder.model.encoder.layer",
        "additional_image_encoder.model.vision_model.encoder.layers",
    ]

    def __init__(
        self,
        main_image_encoder,
        additional_image_encoder,
    ):
        super().__init__()
        self.main_image_encoder = build_image_encoder(main_image_encoder)
        self.additional_image_encoder = build_image_encoder(additional_image_encoder)

    def forward(self, image, mask=None, **kwargs):
        outputs = {
            "main": self.main_image_encoder(image, mask=mask, **kwargs),
            "additional": self.additional_image_encoder(image, mask=mask, **kwargs),
        }
        return outputs

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        grouped: dict[str, list[tuple[str, torch.Tensor]]] = {
            "main_image_encoder": [],
            "additional_image_encoder": [],
        }
        for name, tensor in weights:
            component_name, separator, nested_name = name.partition(".")
            if not separator or component_name not in grouped:
                raise ValueError(f"Unexpected dual image encoder weight: {name}")
            grouped[component_name].append((nested_name, tensor))

        loaded = set()
        components = (
            ("main_image_encoder", self.main_image_encoder),
            ("additional_image_encoder", self.additional_image_encoder),
        )
        for component_name, component in components:
            component_weights = grouped[component_name]
            loaded.update(
                f"{component_name}.{name}"
                for name in component.load_weights(component_weights)
            )
        return loaded

    def unconditional_embedding(self, batch_size, **kwargs):
        outputs = {
            "main": self.main_image_encoder.unconditional_embedding(
                batch_size, **kwargs
            ),
            "additional": self.additional_image_encoder.unconditional_embedding(
                batch_size, **kwargs
            ),
        }
        return outputs


class SingleImageEncoder(nn.Module, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    layer_names = [
        "main_image_encoder.model.encoder.layer",
        "main_image_encoder.model.vision_model.encoder.layers",
    ]

    def __init__(
        self,
        main_image_encoder,
    ):
        super().__init__()
        self.main_image_encoder = build_image_encoder(main_image_encoder)

    def forward(self, image, mask=None, **kwargs):
        outputs = {
            "main": self.main_image_encoder(image, mask=mask, **kwargs),
        }
        return outputs

    def unconditional_embedding(self, batch_size, **kwargs):
        outputs = {
            "main": self.main_image_encoder.unconditional_embedding(
                batch_size, **kwargs
            ),
        }
        return outputs

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        prefix = "main_image_encoder."
        nested_weights = []
        for name, tensor in weights:
            if not name.startswith(prefix):
                raise ValueError(f"Unexpected single image encoder weight: {name}")
            nested_weights.append((name.removeprefix(prefix), tensor))
        return {
            f"{prefix}{name}"
            for name in self.main_image_encoder.load_weights(nested_weights)
        }


# Entry class for model registry
EntryClass = [
    SingleImageEncoder,
    DualImageEncoder,
    DinoImageEncoder,
    DinoImageEncoderMV,
]
