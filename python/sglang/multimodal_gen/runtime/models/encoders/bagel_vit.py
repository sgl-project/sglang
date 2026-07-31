# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""BAGEL SigLIP NaViT image encoder and connector.

The component owns every image-understanding weight from ``ema.safetensors``:
``vit_model.*``, ``connector.*``, and ``vit_pos_embed.*``. Keeping these
weights outside the denoiser makes the request boundary explicit and lets the
pipeline account for the component independently.

Source: https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/bagel/siglip_navit.py
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import functional as TF

from sglang.multimodal_gen.configs.models.encoders.bagel_vit import (
    BagelImageEncoderConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.bagel import (
    calculate_bagel_resize_dimensions,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)

# Latest Torch SDPA is faster for BAGEL's shorter vision sequences on SM100,
# while FA4 wins decisively for large inputs. At 2,304 tokens the measured gain
# is marginal; 2,500 tokens and above show a clear win. Keep that policy scoped
# to the architecture on which it was measured.
_FA4_MIN_SEQUENCE_LENGTH = 2500


def _uses_sm100_fa4_crossover(backend: AttentionBackendEnum) -> bool:
    """Return whether a selected backend uses the measured SM100 policy."""
    if backend != AttentionBackendEnum.FA:
        return False
    device_capability = current_platform.get_device_capability()
    return device_capability is not None and device_capability.to_int() == 100


def _run_siglip_attention(
    attention: LocalAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    use_sm100_fa4_crossover: bool,
) -> Tensor:
    """Use SDPA below the validated SM100 FA4 sequence-length crossover."""
    if use_sm100_fa4_crossover and query.shape[1] < _FA4_MIN_SEQUENCE_LENGTH:
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=attention.softmax_scale,
        )
        return attended.transpose(1, 2)
    return attention(query, key, value)


class _SiglipAttention(nn.Module):
    def __init__(self, config: BagelImageEncoderConfig) -> None:
        super().__init__()
        arch = config.arch_config
        self.num_heads = arch.num_attention_heads
        self.head_dim = arch.hidden_size // arch.num_attention_heads
        self.q_proj = nn.Linear(arch.hidden_size, arch.hidden_size)
        self.k_proj = nn.Linear(arch.hidden_size, arch.hidden_size)
        self.v_proj = nn.Linear(arch.hidden_size, arch.hidden_size)
        self.out_proj = nn.Linear(arch.hidden_size, arch.hidden_size)
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            causal=False,
            supported_attention_backends=arch._supported_attention_backends,
        )
        self._use_sm100_fa4_crossover = _uses_sm100_fa4_crossover(self.attn.backend)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply non-causal self-attention to one packed image."""
        sequence_length = hidden_states.shape[0]
        query = self.q_proj(hidden_states).view(
            1, sequence_length, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            1, sequence_length, self.num_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            1, sequence_length, self.num_heads, self.head_dim
        )
        attended = _run_siglip_attention(
            self.attn,
            query,
            key,
            value,
            self._use_sm100_fa4_crossover,
        ).squeeze(0)
        attended = attended.reshape(sequence_length, -1)
        return self.out_proj(attended)


class _SiglipMLP(nn.Module):
    def __init__(self, config: BagelImageEncoderConfig) -> None:
        super().__init__()
        arch = config.arch_config
        self.fc1 = nn.Linear(arch.hidden_size, arch.intermediate_size)
        self.fc2 = nn.Linear(arch.intermediate_size, arch.hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class _SiglipEncoderLayer(nn.Module):
    def __init__(self, config: BagelImageEncoderConfig) -> None:
        super().__init__()
        arch = config.arch_config
        self.layer_norm1 = nn.LayerNorm(arch.hidden_size, eps=arch.layer_norm_eps)
        self.self_attn = _SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(arch.hidden_size, eps=arch.layer_norm_eps)
        self.mlp = _SiglipMLP(config)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class _SiglipEmbeddings(nn.Module):
    def __init__(self, config: BagelImageEncoderConfig) -> None:
        super().__init__()
        arch = config.arch_config
        patch_width = arch.patch_size * arch.patch_size * 3
        self.patch_embedding = nn.Linear(patch_width, arch.hidden_size)
        self.position_embedding = nn.Embedding(
            arch.position_embedding_rows, arch.hidden_size
        )

    def forward(self, patches: Tensor, position_ids: Tensor) -> Tensor:
        return self.patch_embedding(patches) + self.position_embedding(position_ids)


class _SiglipEncoder(nn.Module):
    def __init__(self, config: BagelImageEncoderConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _SiglipEncoderLayer(config)
                for _ in range(config.arch_config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _SiglipVisionTransformer(nn.Module):
    def __init__(self, config: BagelImageEncoderConfig) -> None:
        super().__init__()
        arch = config.arch_config
        self.embeddings = _SiglipEmbeddings(config)
        self.encoder = _SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(arch.hidden_size, eps=arch.layer_norm_eps)

    def forward(self, patches: Tensor, position_ids: Tensor) -> Tensor:
        hidden_states = self.embeddings(patches, position_ids)
        hidden_states = self.encoder(hidden_states)
        return self.post_layernorm(hidden_states)


class _SiglipVisionModel(nn.Module):
    def __init__(self, config: BagelImageEncoderConfig) -> None:
        super().__init__()
        self.vision_model = _SiglipVisionTransformer(config)

    def forward(self, patches: Tensor, position_ids: Tensor) -> Tensor:
        return self.vision_model(patches, position_ids)


class _Connector(nn.Module):
    def __init__(self, config: BagelImageEncoderConfig) -> None:
        super().__init__()
        arch = config.arch_config
        self.fc1 = nn.Linear(arch.hidden_size, arch.llm_hidden_size)
        self.fc2 = nn.Linear(arch.llm_hidden_size, arch.llm_hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class _PositionEmbedding(nn.Module):
    def __init__(self, config: BagelImageEncoderConfig) -> None:
        super().__init__()
        arch = config.arch_config
        self.pos_embed = nn.Parameter(
            torch.empty(arch.position_embedding_rows, arch.llm_hidden_size),
            requires_grad=False,
        )

    def forward(self, position_ids: Tensor) -> Tensor:
        return F.embedding(position_ids, self.pos_embed)


def preprocess_bagel_vit_image(
    image: Image.Image,
    config: BagelImageEncoderConfig,
) -> tuple[Tensor, Tensor, tuple[int, int]]:
    """Resize, normalize, and patchify one image for BAGEL's ViT.

    Args:
        image: Input PIL image. The pipeline normally passes the VAE-resized
            Editing image so the two image contexts share the same source crop.
        config: ViT architecture and resize constraints.

    Returns:
        Raw normalized patches, flattened position IDs, and ``(height, width)``
        in patch units.

    Raises:
        ValueError: If the transformed geometry cannot be represented by the
            checkpoint position table.
    """
    arch = config.arch_config
    image = image.convert("RGB")
    width, height = calculate_bagel_resize_dimensions(
        image.width,
        image.height,
        max_size=arch.max_image_size,
        min_size=arch.min_image_size,
        stride=arch.patch_size,
    )
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    pixels = TF.to_tensor(image).mul_(2.0).sub_(1.0)
    channels, pixel_height, pixel_width = pixels.shape
    patch_height = pixel_height // arch.patch_size
    patch_width = pixel_width // arch.patch_size
    if (
        patch_height > arch.max_num_patches_per_side
        or patch_width > arch.max_num_patches_per_side
    ):
        raise ValueError(
            "BAGEL ViT image exceeds the checkpoint position table: "
            f"{patch_width}x{patch_height} patches"
        )
    patches = pixels.reshape(
        channels,
        patch_height,
        arch.patch_size,
        patch_width,
        arch.patch_size,
    )
    patches = torch.einsum("chpwq->hwpqc", patches).reshape(
        -1, arch.patch_size * arch.patch_size * channels
    )
    rows = torch.arange(patch_height).unsqueeze(1)
    columns = torch.arange(patch_width).unsqueeze(0)
    position_ids = (rows * arch.max_num_patches_per_side + columns).reshape(-1)
    return patches, position_ids, (patch_height, patch_width)


class BagelImageEncoder(nn.Module):
    """Encode one BAGEL Editing image into LLM-width semantic tokens.

    Args:
        config: ViT, connector, and position-table architecture.
    """

    def __init__(self, config: BagelImageEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or BagelImageEncoderConfig()
        self.vit_model = _SiglipVisionModel(self.config)
        self.connector = _Connector(self.config)
        self.vit_pos_embed = _PositionEmbedding(self.config)
        self.requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, patches: Tensor, position_ids: Tensor) -> Tensor:
        """Project normalized patches into BAGEL's LLM hidden space.

        Args:
            patches: Tensor shaped ``[tokens, patch_size * patch_size * 3]``.
            position_ids: Flattened two-dimensional checkpoint positions.

        Returns:
            Semantic image tokens shaped ``[tokens, llm_hidden_size]``.

        Note:
            Callers must establish ``set_forward_context`` before invoking the
            encoder because native attention backends consume that context.

        Raises:
            ValueError: If patch and position counts differ or a position is out
                of range.
        """
        if patches.ndim != 2 or position_ids.ndim != 1:
            raise ValueError("BAGEL ViT expects 2D patches and 1D position IDs")
        if patches.shape[0] != position_ids.shape[0]:
            raise ValueError("BAGEL ViT patch and position counts must match")
        if position_ids.numel() and (
            int(position_ids.min()) < 0
            or int(position_ids.max())
            >= self.config.arch_config.position_embedding_rows
        ):
            raise ValueError("BAGEL ViT position ID is outside the checkpoint table")
        patches = patches.to(device=self.device, dtype=self.dtype)
        position_ids = position_ids.to(device=self.device, dtype=torch.long)
        hidden_states = self.vit_model(patches, position_ids)
        return self.connector(hidden_states) + self.vit_pos_embed(position_ids)

    def encode_image(self, image: Image.Image) -> Tensor:
        """Preprocess and encode one PIL image.

        Args:
            image: RGB-compatible PIL image.

        Returns:
            LLM-width semantic image tokens on this component's device.

        Note:
            Callers must establish ``set_forward_context`` before invoking the
            encoder because native attention backends consume that context.

        Raises:
            ValueError: If image geometry exceeds the checkpoint tables.
        """
        patches, position_ids, _ = preprocess_bagel_vit_image(image, self.config)
        return self.forward(patches, position_ids)

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
        *,
        strict: bool = True,
    ) -> set[str]:
        """Stream ViT, connector, and position weights from ``ema.safetensors``.

        Args:
            weights: Iterator of checkpoint tensor names and values.
            strict: Require complete component coverage and reject unknown keys
                within the image-encoder namespaces.

        Returns:
            Target parameter names populated by the iterator.

        Raises:
            ValueError: If a required tensor is missing, has the wrong shape, or
                uses an unknown image-encoder key.
        """
        params = dict(self.named_parameters())
        required = set(params)
        loaded: set[str] = set()
        unexpected: list[str] = []
        prefixes = ("vit_model.", "connector.", "vit_pos_embed.")

        for source_name, tensor in weights:
            if not source_name.startswith(prefixes):
                continue
            parameter = params.get(source_name)
            if parameter is None:
                unexpected.append(source_name)
                continue
            self._load_parameter(source_name, parameter, tensor)
            loaded.add(source_name)

        missing = sorted(required - loaded)
        if strict and (missing or unexpected):
            details = []
            if missing:
                details.append(f"missing image encoder weights: {missing}")
            if unexpected:
                details.append(
                    f"unexpected image encoder weights: {sorted(unexpected)}"
                )
            raise ValueError("; ".join(details))
        return loaded

    def _load_parameter(
        self, name: str, parameter: nn.Parameter, tensor: Tensor
    ) -> None:
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise ValueError(
                f"image encoder weight shape mismatch for {name}: expected "
                f"{tuple(parameter.shape)}, got {tuple(tensor.shape)}"
            )
        if parameter.is_meta:
            parent: nn.Module = self
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(
                parent,
                parts[-1],
                nn.Parameter(tensor.to(dtype=parameter.dtype), requires_grad=False),
            )
            return
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


EntryClass = BagelImageEncoder
