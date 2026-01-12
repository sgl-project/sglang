# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
OpenVLA model implementation for SGLang inference.

OpenVLA (Open Vision-Language-Action) is a 7B VLA model for robotic manipulation.
Architecture: DINOv2 + SigLIP (fused) -> MLP Projector -> Llama-2-7B -> Action Tokens

References:
    - Paper: https://arxiv.org/abs/2406.09246
    - Model: https://huggingface.co/openvla/openvla-7b
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.utils import add_prefix, logger


class PrismaticVisionBackbone(nn.Module):
    """Fused vision backbone combining DINOv2 and SigLIP.

    OpenVLA uses a fused dual-encoder vision backbone that concatenates
    features from DINOv2 (structural/geometric) and SigLIP (semantic).
    Both are ViT models loaded via TIMM.
    """

    def __init__(
        self,
        timm_model_ids: List[str],
        image_sizes: List[int],
        use_fused_vision_backbone: bool = True,
    ):
        super().__init__()
        self.use_fused = use_fused_vision_backbone
        self.timm_model_ids = timm_model_ids
        self.image_sizes = image_sizes

        # Will be loaded via load_weights
        self.featurizer = None
        self.fused_featurizer = None
        self.embed_dim = None

    def _init_timm_models(self):
        """Initialize TIMM models. Called after config is loaded.

        Important: Must specify img_size to match OpenVLA's training configuration.
        DINOv2 defaults to 518x518 but OpenVLA uses 224x224.
        """
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for OpenVLA. Install with: pip install timm")

        # Get image size from config (default 224 for OpenVLA)
        img_size = self.image_sizes[0] if self.image_sizes else 224

        # Primary encoder (DINOv2)
        # Must specify img_size=224 because DINOv2 defaults to 518x518
        self.featurizer = timm.create_model(
            self.timm_model_ids[0],
            pretrained=False,  # weights loaded separately
            num_classes=0,
            img_size=img_size,
        )
        self.embed_dim = self.featurizer.embed_dim

        # Secondary encoder (SigLIP) for fused backbone
        if self.use_fused and len(self.timm_model_ids) > 1:
            self.fused_featurizer = timm.create_model(
                self.timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=img_size,
            )
            self.embed_dim += self.fused_featurizer.embed_dim

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """Extract and fuse features from both vision encoders.

        Args:
            pixel_values: Images of shape (batch, channels, height, width).

        Returns:
            Patch features of shape (batch, num_patches, embed_dim).
        """
        # Get features from primary encoder
        features = self.featurizer.forward_features(pixel_values)

        # Remove CLS token if present (keep only patch tokens)
        if features.shape[1] > (pixel_values.shape[-1] // 14) ** 2:
            features = features[:, 1:, :]

        # Fuse with secondary encoder
        if self.use_fused and self.fused_featurizer is not None:
            fused_features = self.fused_featurizer.forward_features(pixel_values)
            if fused_features.shape[1] > (pixel_values.shape[-1] // 14) ** 2:
                fused_features = fused_features[:, 1:, :]
            features = torch.cat([features, fused_features], dim=-1)

        return features


class PrismaticProjector(nn.Module):
    """MLP projector to align vision features with LLM embedding space.

    For fused vision backbone (OpenVLA default):
        vision_dim -> 4*vision_dim -> llm_dim -> llm_dim
        with GELU activations between layers.
    """

    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        use_fused: bool = True,
    ):
        super().__init__()
        self.use_fused = use_fused

        if use_fused:
            # Fused projector: 3-layer MLP
            intermediate_dim = 4 * vision_dim
            self.fc1 = nn.Linear(vision_dim, intermediate_dim)
            self.act_fn1 = nn.GELU()
            self.fc2 = nn.Linear(intermediate_dim, llm_dim)
            self.act_fn2 = nn.GELU()
            self.fc3 = nn.Linear(llm_dim, llm_dim)
        else:
            # Simple 2-layer MLP
            self.fc1 = nn.Linear(vision_dim, llm_dim)
            self.act_fn1 = nn.GELU()
            self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to LLM embedding space.

        Args:
            vision_features: Shape (batch, num_patches, vision_dim).

        Returns:
            Projected features of shape (batch, num_patches, llm_dim).
        """
        x = self.fc1(vision_features)
        x = self.act_fn1(x)
        x = self.fc2(x)

        if self.use_fused:
            x = self.act_fn2(x)
            x = self.fc3(x)

        return x


class OpenVLAForActionPrediction(nn.Module):
    """OpenVLA model for action prediction via SGLang.

    Architecture follows Prismatic VLM with fused vision backbone:
    - Vision: DINOv2 + SigLIP (concatenated features)
    - Projector: 3-layer MLP with GELU
    - LLM: Llama-2-7B

    Action prediction:
    - 7D action space: [dx, dy, dz, drx, dry, drz, gripper]
    - 256 bins per dimension (tokens 32000-32255 in Llama vocab)
    - Autoregressive generation of 7 action tokens
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        # Vision backbone config
        self.timm_model_ids = getattr(
            config,
            "timm_model_ids",
            ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
        )
        self.image_sizes = getattr(config, "image_sizes", [224, 224])
        self.use_fused_vision_backbone = getattr(config, "use_fused_vision_backbone", True)

        # Vision backbone
        self.vision_backbone = PrismaticVisionBackbone(
            timm_model_ids=self.timm_model_ids,
            image_sizes=self.image_sizes,
            use_fused_vision_backbone=self.use_fused_vision_backbone,
        )

        # Projector (initialized after vision backbone loads)
        self.projector = None

        # Language model
        self.language_model = LlamaForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        # Image token handling
        # OpenVLA uses pad_token_id=32000 as the image placeholder
        self.image_token_id = getattr(config, "pad_token_id", 32000)

        # Action token config
        self.n_action_bins = getattr(config, "n_action_bins", 256)
        self.action_dim = 7  # 6 DoF pose + gripper

        # Number of image patches (224/14)^2 = 256
        self.num_patches = (self.image_sizes[0] // 14) ** 2

    def pad_input_ids(
        self,
        input_ids: List[int],
        image_inputs: MultimodalInputs,
    ) -> List[int]:
        """Replace image placeholder tokens with appropriate padding.

        OpenVLA inserts vision features after the BOS token. The number of
        image tokens equals num_patches (256 for 224x224 images).

        Args:
            input_ids: Token IDs with image placeholders.
            image_inputs: Multimodal input data.

        Returns:
            Padded input IDs with expanded image positions.
        """
        if not image_inputs or not image_inputs.mm_items:
            return input_ids

        pad_values = [item.pad_value for item in image_inputs.mm_items]
        image_inputs.image_pad_len = []
        image_inputs.image_offsets = []

        for idx, item in enumerate(image_inputs.mm_items):
            # Find image token position
            try:
                offset = input_ids.index(self.image_token_id)
            except ValueError:
                # No image token found, assume after BOS (position 1)
                offset = 1

            # Number of patches for this image
            num_patches = self.num_patches

            # Replace single image token with num_patches padding tokens
            pad_value = pad_values[idx % len(pad_values)]
            input_ids = (
                input_ids[:offset]
                + [pad_value] * num_patches
                + input_ids[offset + 1:]
            )

            image_inputs.image_offsets.append(offset)
            image_inputs.image_pad_len.append(num_patches)

        return input_ids

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images through vision backbone and projector.

        Args:
            pixel_values: Images of shape (batch, C, H, W).

        Returns:
            Projected features of shape (batch, num_patches, llm_dim).
        """
        # Get vision features
        vision_features = self.vision_backbone(pixel_values)

        # Project to LLM space
        projected = self.projector(vision_features)

        return projected

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Forward pass for action prediction.

        Handles both:
        1. Prefill: Process images + text together
        2. Decode: Generate action tokens autoregressively

        Args:
            input_ids: Token IDs.
            positions: Position IDs.
            forward_batch: Batch info including multimodal inputs.

        Returns:
            Logits for next token prediction.
        """
        image_inputs = forward_batch.mm_inputs

        if forward_batch.forward_mode.is_extend():
            # Prefill: embed text and insert vision features
            input_ids.clamp_(min=0, max=self.config.vocab_size - 1)
            input_embeds = self.language_model.model.embed_tokens(input_ids)

            # Check if we need to process images
            max_image_offset = []
            for im in image_inputs:
                if im and im.image_offsets:
                    max_image_offset.append(
                        np.max(np.array(im.image_offsets) + np.array(im.image_pad_len))
                    )
                else:
                    max_image_offset.append(-1)

            start_positions = positions[forward_batch.extend_start_loc].cpu().numpy()
            need_vision = start_positions <= np.array(max_image_offset)

            if need_vision.any():
                bs = forward_batch.batch_size

                # Gather pixel values
                pixel_values = []
                for i in range(bs):
                    if need_vision[i] and image_inputs[i]:
                        for item in image_inputs[i].mm_items:
                            pixel_values.append(item.feature)

                if pixel_values:
                    # Stack and encode images
                    pixel_values = torch.tensor(
                        np.stack(pixel_values, axis=0),
                        device=input_embeds.device,
                        dtype=input_embeds.dtype,
                    )
                    image_features = self.encode_images(pixel_values)

                    # Insert image features into embeddings
                    extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
                    extend_seq_lens = forward_batch.extend_seq_lens.cpu().numpy()
                    prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu

                    feat_idx = 0
                    for i in range(bs):
                        if not need_vision[i] or not image_inputs[i]:
                            continue

                        start_idx = extend_start_loc_cpu[i]
                        seq_len = extend_seq_lens[i]
                        prefix_len = prefix_lens_cpu[i]

                        for img_idx, image_offset in enumerate(image_inputs[i].image_offsets):
                            pad_len = image_inputs[i].image_pad_len[img_idx]

                            if image_offset + pad_len <= prefix_len:
                                continue
                            if image_offset >= prefix_len + seq_len:
                                break

                            img_feature = image_features[feat_idx]
                            feat_idx += 1

                            # Calculate insertion indices
                            input_offset = image_offset - prefix_len
                            left_idx = start_idx + max(0, input_offset)
                            right_idx = left_idx + pad_len

                            # Handle boundary conditions
                            if input_offset < 0:
                                img_feature = img_feature[-input_offset:]
                            if right_idx > start_idx + seq_len:
                                img_feature = img_feature[:start_idx + seq_len - right_idx]
                                right_idx = start_idx + seq_len

                            try:
                                input_embeds[left_idx:right_idx] = img_feature
                            except RuntimeError as e:
                                logger.warning(f"Error inserting image features: {e}")

            return self.language_model(
                input_ids, positions, forward_batch, input_embeds=input_embeds
            )

        elif forward_batch.forward_mode.is_decode():
            # Decode: standard LLM forward
            return self.language_model(input_ids, positions, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load OpenVLA weights from HuggingFace checkpoint.

        Weight mapping:
        - vision_backbone.featurizer.* -> DINOv2 weights
        - vision_backbone.fused_featurizer.* -> SigLIP weights
        - projector.fc1/fc2/fc3.* -> Projector weights
        - language_model.* -> Llama weights
        """
        # Initialize vision backbone TIMM models
        self.vision_backbone._init_timm_models()

        # Initialize projector after we know vision dimensions
        vision_dim = self.vision_backbone.embed_dim
        llm_dim = self.config.hidden_size
        self.projector = PrismaticProjector(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            use_fused=self.use_fused_vision_backbone,
        )

        # Create parameter lookup
        params_dict = dict(self.named_parameters())

        # Weight name mappings for Prismatic checkpoint format
        weight_mappings = {
            "vision_backbone.featurizer": "vision_backbone.featurizer",
            "vision_backbone.fused_featurizer": "vision_backbone.fused_featurizer",
            "projector": "projector",
            "language_model.model": "language_model.model",
            "language_model.lm_head": "language_model.lm_head",
        }

        for name, loaded_weight in weights:
            # Handle language model weights
            if name.startswith("language_model."):
                lm_name = name[len("language_model."):]
                self.language_model.load_weights([(lm_name, loaded_weight)])
                continue

            # Handle vision and projector weights
            matched = False
            for src_prefix, tgt_prefix in weight_mappings.items():
                if name.startswith(src_prefix):
                    tgt_name = name.replace(src_prefix, tgt_prefix)
                    if tgt_name in params_dict:
                        param = params_dict[tgt_name]

                        # Handle potential size mismatch for positional embeddings
                        if "pos_embed" in tgt_name and param.shape != loaded_weight.shape:
                            logger.warning(
                                f"Position embedding size mismatch for {tgt_name}: "
                                f"param={param.shape}, weight={loaded_weight.shape}. "
                                f"Interpolating..."
                            )
                            loaded_weight = self._interpolate_pos_embed(
                                loaded_weight, param.shape
                            )

                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        matched = True
                    break

            if not matched:
                # Try direct parameter loading
                if name in params_dict:
                    param = params_dict[name]

                    # Handle potential size mismatch for positional embeddings
                    if "pos_embed" in name and param.shape != loaded_weight.shape:
                        logger.warning(
                            f"Position embedding size mismatch for {name}: "
                            f"param={param.shape}, weight={loaded_weight.shape}. "
                            f"Interpolating..."
                        )
                        loaded_weight = self._interpolate_pos_embed(
                            loaded_weight, param.shape
                        )

                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)

    def _interpolate_pos_embed(
        self,
        pos_embed: torch.Tensor,
        target_shape: torch.Size,
    ) -> torch.Tensor:
        """Interpolate positional embeddings to match target size.

        Args:
            pos_embed: Source positional embedding of shape (1, src_len, dim).
            target_shape: Target shape (1, tgt_len, dim).

        Returns:
            Interpolated positional embedding matching target_shape.
        """
        import torch.nn.functional as F

        src_len = pos_embed.shape[1]
        tgt_len = target_shape[1]

        if src_len == tgt_len:
            return pos_embed

        # Reshape for 2D interpolation: (1, dim, sqrt(src_len), sqrt(src_len))
        src_size = int(src_len ** 0.5)
        tgt_size = int(tgt_len ** 0.5)

        # Reshape: (1, src_len, dim) -> (1, dim, src_size, src_size)
        pos_embed_2d = pos_embed.permute(0, 2, 1).reshape(1, -1, src_size, src_size)

        # Interpolate
        pos_embed_2d = F.interpolate(
            pos_embed_2d.float(),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        )

        # Reshape back: (1, dim, tgt_size, tgt_size) -> (1, tgt_len, dim)
        pos_embed_out = pos_embed_2d.reshape(1, -1, tgt_len).permute(0, 2, 1)

        return pos_embed_out.to(pos_embed.dtype)

    def decode_action_tokens(
        self,
        action_tokens: torch.Tensor,
        action_token_offset: int = 32000,
    ) -> torch.Tensor:
        """Convert action tokens to continuous action values.

        OpenVLA uses 256-bin discretization per action dimension.
        Tokens are in range [action_token_offset, action_token_offset + 255].
        Values are mapped to [-1, 1].

        Args:
            action_tokens: Token IDs of shape (batch, 7).
            action_token_offset: Starting token ID for action tokens.

        Returns:
            Continuous actions of shape (batch, 7) in [-1, 1].
        """
        # Convert token IDs to bin indices
        bin_indices = action_tokens - action_token_offset

        # Map to continuous values: bin / 255 * 2 - 1
        actions = (bin_indices.float() / 255.0) * 2.0 - 1.0

        return actions


# SGLang model registration
EntryClass = [OpenVLAForActionPrediction]
