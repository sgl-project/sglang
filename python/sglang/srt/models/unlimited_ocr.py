"""Standalone UNLIMITED-OCR model (SAM + CLIP vision encoders, Deepseek backbone)."""

import logging
from typing import Iterable, List, Optional, Set, Tuple, TypeAlias, Union

import torch
from torch import Tensor, nn

from sglang.srt.configs.unlimited_ocr import UnlimitedVLConfig
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek import DeepseekForCausalLM
from sglang.srt.models.deepseek_ocr import (
    MlpProjector,
    build_clip_l,
    build_sam_vit_b,
    merge_multimodal_embeddings,
)
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.utils import cpu_has_amx_support, is_cpu

_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()

NestedTensors: TypeAlias = Union[
    list["NestedTensors"],
    list["torch.Tensor"],
    "torch.Tensor",
    tuple["torch.Tensor", ...],
]

MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]

logger = logging.getLogger(__name__)


class UnlimitedOCRForCausalLM(nn.Module):
    """Standalone UNLIMITED-OCR model (SAM + CLIP ViT) with prefill-aware SWA."""

    def __init__(
        self,
        *,
        config: UnlimitedVLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        """Initialize UnlimitedOCRForCausalLM with vision encoders, projector, and LM."""
        super().__init__()

        self.config = config
        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        n_embed = getattr(self.projector_config, "n_embed", 1280)

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

        self.model = DeepseekForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "language"),
        )

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()

        self.projector = MlpProjector(
            projector_type=self.projector_config.projector_type,
            input_dim=self.projector_config.input_dim,
            n_embed=n_embed,
            depth=self.projector_config.depth,
            mlp_ratio=self.projector_config.mlp_ratio,
            downsample_ratio=self.projector_config.downsample_ratio,
        )

        self.image_token_id = None

    def get_attention_sliding_window_size(self) -> Optional[int]:
        """Return the sliding window size from the model config, or None."""
        return getattr(self.config, "sliding_window_size", None)

    def is_prefill_aware_swa(self) -> bool:
        """Prefill tokens are always retained in KV cache during decode."""
        return True

    def _encode_ocr1_features(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images through SAM and CLIP encoders, then project features."""
        features_1 = self.sam_model(images)
        features_2 = self.vision_model(images, features_1)
        features = torch.cat(
            (
                features_2[:, 1:],
                features_1.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )
        return self.projector(features)

    def _format_ocr1_global_features(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape global features into a flat sequence with newline tokens."""
        _, hw, n_dim = features.shape
        h = w = int(hw**0.5)
        features = features.view(h, w, n_dim)
        features = torch.cat(
            [features, self.image_newline[None, None, :].expand(h, 1, n_dim)],
            dim=1,
        )
        return features.view(-1, n_dim)

    def _format_ocr1_local_features(
        self, features: torch.Tensor, crop_shape: torch.Tensor
    ) -> torch.Tensor:
        """Reshape local crop features into a flat sequence with newline tokens."""
        _, hw2, n_dim2 = features.shape
        h2 = w2 = int(hw2**0.5)
        width_crop_num, height_crop_num = int(crop_shape[0]), int(crop_shape[1])
        features = (
            features.view(height_crop_num, width_crop_num, h2, w2, n_dim2)
            .permute(0, 2, 1, 3, 4)
            .reshape(height_crop_num * h2, width_crop_num * w2, n_dim2)
        )
        features = torch.cat(
            [
                features,
                self.image_newline[None, None, :].expand(
                    height_crop_num * h2, 1, n_dim2
                ),
            ],
            dim=1,
        )
        return features.view(-1, n_dim2)

    @staticmethod
    def _collect_mm_flag(
        items: List[MultimodalDataItem], flag_name: str
    ) -> Optional[List[bool]]:
        """Collect a boolean multimodal flag from all data items."""
        values = []
        for item in items:
            value = getattr(item, flag_name, None)
            if value is None:
                return None
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(bool(value))
        return values

    def _parse_and_validate_image_input(self, **kwargs: object):
        """Parse and validate pixel values, spatial crops, and image crops."""
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)
        has_images = kwargs.pop("has_images", None)

        if pixel_values is None:
            return None
        if has_images is not None:
            if not has_images:
                return None
        elif torch.sum(pixel_values).item() == 0:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
                )
            if not isinstance(images_spatial_crop, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image sizes. "
                    f"Got type: {type(images_spatial_crop)}"
                )
            if not isinstance(images_crop, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image crop. " f"Got type: {type(images_crop)}"
                )
            return [pixel_values, images_crop, images_spatial_crop]

        raise AssertionError("This line should be unreachable.")

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_crop: torch.Tensor,
        images_spatial_crop: torch.Tensor,
        has_local_crops: Optional[List[bool]] = None,
    ) -> NestedTensors:
        """Encode pixel values into per-image embedding sequences."""
        images_in_this_batch = []

        with torch.no_grad():
            for jdx in range(images_spatial_crop.size(0)):
                patches = images_crop[jdx][0].to(torch.bfloat16)
                image_ori = pixel_values[jdx]
                crop_shape = images_spatial_crop[jdx][0]
                use_local_crops = (
                    has_local_crops[jdx]
                    if has_local_crops is not None
                    else torch.sum(patches).item() != 0
                )

                global_features = self._encode_ocr1_features(image_ori)
                global_features = self._format_ocr1_global_features(global_features)

                if use_local_crops:
                    local_features = self._encode_ocr1_features(patches)
                    local_features = self._format_ocr1_local_features(
                        local_features, crop_shape
                    )
                    global_local_features = torch.cat(
                        [
                            local_features,
                            global_features,
                            self.view_seperator[None, :],
                        ],
                        dim=0,
                    )
                else:
                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :]], dim=0
                    )

                images_in_this_batch.append(global_local_features)

        return images_in_this_batch

    def _process_image_input(self, mm_items: List[MultimodalDataItem]) -> torch.Tensor:
        """Process multimodal data items into concatenated vision features."""
        target_dtype = self.vision_model.dtype
        has_local_crops = self._collect_mm_flag(mm_items, "has_local_crops")
        pixel_values = torch.stack([item.feature for item in mm_items], dim=0).type(
            target_dtype
        )

        images_crop = (
            torch.stack([item.images_crop for item in mm_items], dim=0)
            .type(target_dtype)
            .to(device=pixel_values.device)
        )
        images_spatial_crop = (
            torch.cat([item.images_spatial_crop for item in mm_items], dim=0)
            .type(torch.long)
            .to(device=pixel_values.device)
        )
        pixel_values = pixel_values.view(
            pixel_values.shape[0] * pixel_values.shape[1], 1, *pixel_values.shape[2:]
        )
        images_crop = images_crop.view(
            images_crop.shape[0] * images_crop.shape[1], 1, *images_crop.shape[2:]
        )
        images_spatial_crop = images_spatial_crop.view(
            images_spatial_crop.shape[0] * images_spatial_crop.shape[1],
            1,
            *images_spatial_crop.shape[2:],
        )

        assert images_crop.dim() == 6
        assert images_spatial_crop.dim() == 3

        vision_feature_lists = self._pixel_values_to_embedding(
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            has_local_crops=has_local_crops,
        )
        vision_features = torch.cat(vision_feature_lists, dim=0).type(target_dtype)
        return vision_features

    def get_language_model(self) -> torch.nn.Module:
        """Return the underlying language model."""
        return self.model

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        """Compute multimodal embeddings from image inputs, if present."""
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        """Get text embeddings and merge in multimodal embeddings if provided."""
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, self.image_token_id
            )
        return inputs_embeds

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """Pad input token IDs with multimodal placeholder tokens."""
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract vision features from multimodal data items."""
        vision_embeddings = self._process_image_input(items)
        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ):
        """Run the full multimodal forward pass (embed, encode, decode)."""
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load and remap checkpoint weights into the model parameters."""
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name == "lm_head.weight":
                name = "model.lm_head.weight"
            elif name.startswith("model."):
                if (
                    "image_newline" in name
                    or ".projector" in name
                    or "vision_model" in name
                    or "sam_model" in name
                    or "view_seperator" in name
                ):
                    name = name[len("model.") :]
                elif not (
                    ".projector" in name
                    or "vision_model" in name
                    or "sam_model" in name
                    or "image_newline" in name
                ):
                    name = name.replace("model.", "model.model.")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if (
                    "mlp.experts." in name or "mlp.shared_experts." in name
                ) and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            raise RuntimeError(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )
        self.post_load_weights()

    def post_load_weights(self):
        """Apply post-loading weight transformations (e.g., AMX repacking on CPU)."""
        if _is_cpu and _is_cpu_amx_available:
            from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading

            layer_ids = int(self.config.num_hidden_layers)
            first_k_dense_replace_id = (
                self.config.first_k_dense_replace
                if hasattr(self.config, "first_k_dense_replace")
                else -1
            )
            moe_layer_freq_id = (
                self.config.moe_layer_freq
                if hasattr(self.config, "moe_layer_freq")
                else 1
            )
            for layer_id in range(0, layer_ids):
                if (
                    layer_id >= first_k_dense_replace_id
                    and layer_id % moe_layer_freq_id == 0
                ):
                    if (
                        hasattr(self.model, "model")
                        and hasattr(self.model.model, "layers")
                        and hasattr(self.model.model.layers[layer_id], "mlp")
                    ):
                        self_moe = self.model.model.layers[layer_id].mlp
                        if hasattr(self_moe, "w1") and hasattr(self_moe, "w2"):
                            _amx_process_weight_after_loading(self_moe, ["w1", "w2"])


EntryClass = [UnlimitedOCRForCausalLM]
