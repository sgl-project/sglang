from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from sglang.srt.configs.deepseekvl2 import (
    DeepseekVL2Config,
    DeepseekVL2MlpProjectorConfig,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek import DeepseekForCausalLM
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM


class DeepseekVL2MlpProjector(nn.Module):
    def __init__(
        self,
        config: DeepseekVL2MlpProjectorConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):

        super().__init__()

        self.config = config

        if config.projector_type == "identity":
            modules = nn.Identity()

        elif config.projector_type == "linear":
            self.layers = nn.ModuleList(
                [
                    ReplicatedLinear(
                        config.input_dim,
                        config.n_embed,
                        quant_config=quant_config,
                    )
                ]
            )

        elif config.projector_type == "mlp_gelu":
            mlp_depth = config.depth
            self.layers = nn.ModuleList(
                [
                    ReplicatedLinear(
                        config.input_dim,
                        config.n_embed,
                        quant_config=quant_config,
                    )
                ]
            )
            for _ in range(1, mlp_depth):
                self.layers.append(nn.GELU())
                self.layers.append(
                    ReplicatedLinear(
                        config.n_embed,
                        config.n_embed,
                        quant_config=quant_config,
                    )
                )

        elif config.projector_type == "downsample_mlp_gelu":
            mlp_depth = config.depth
            mlp_ratio = config.mlp_ratio
            self.layers = nn.ModuleList(
                [
                    ReplicatedLinear(
                        config.input_dim
                        * config.downsample_ratio
                        * config.downsample_ratio,
                        config.n_embed * mlp_ratio,
                        quant_config=quant_config,
                    )
                ]
            )
            for _ in range(1, mlp_depth - 1):
                self.layers.append(nn.GELU())
                self.layers.append(
                    ReplicatedLinear(
                        config.n_embed * mlp_ratio,
                        config.n_embed * mlp_ratio,
                        quant_config=quant_config,
                    )
                )
            self.layers.append(nn.GELU())
            self.layers.append(
                ReplicatedLinear(
                    config.n_embed * mlp_ratio,
                    config.n_embed,
                    quant_config=quant_config,
                )
            )

        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

        if config.token_pooling:
            self.token_pooling_layer = ReplicatedLinear(
                config.input_dim * 4, config.input_dim, quant_config=quant_config
            )

    def forward(self, x):
        if self.config.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)

            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            patches = patches.contiguous().view(
                batch_size, channels, h_patches * w_patches, -1
            )
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)[0]

        elif self.config.projector_type == "downsample_mlp_gelu":
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.config.downsample_ratio:
                pad = self.config.downsample_ratio - h % self.config.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(
                x,
                kernel_size=self.config.downsample_ratio,
                stride=self.config.downsample_ratio,
                padding=0,
            )  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        return x


class DeepseekVL2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: DeepseekVL2Config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        # ----------- vision encoder ------------
        vision_config = config.vision_config
        self.vision = self._init_vision_module(vision_config, quant_config)

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = DeepseekVL2MlpProjector(projector_config, quant_config)

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        embed_std = 1 / torch.sqrt(
            torch.tensor(projector_config.n_embed, dtype=torch.float32)
        )
        if self.tile_tag == "2D":
            self.image_newline = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
            self.view_seperator = nn.Parameter(
                torch.randn(projector_config.n_embed) * embed_std
            )
        else:
            raise ValueError(f"tile tag should be 2D, but got {self.tile_tag}")

        # ----------- language model ------------
        language_config = config.language_config
        if language_config.use_mla:
            self.language_model = DeepseekV2ForCausalLM(language_config)
        else:
            # deepseek-vl2-tiny forbids mla
            self.language_model = DeepseekForCausalLM(language_config)

    def _init_vision_module(
        self, vision_config, quant_config: Optional[QuantizationConfig]
    ) -> nn.Module:
        # TODO: refactor vision model through timm wrapper from transformers
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm") from ImportError

        model = timm.create_model(
            "vit_so400m_patch14_siglip_384.webli",
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )

        model = model.to(dtype=torch.get_default_dtype())
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ):
        hs = general_mm_embed_routine(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            image_data_embedding_func=self.get_image_feature,
            language_model=self.language_model,
        )

        return hs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters())
        weights = list(weights)
        for name, loaded_weight in weights:
            if "language" in name:
                name = name.replace("language.", "")
                self.language_model.load_weights([(name, loaded_weight)])
            else:
                param = params_dict[name]
                weights_loader = getattr(param, "weight_loader", default_weight_loader)
                weights_loader(param, loaded_weight)

    def pad_input_ids(self, input_ids: List[int], image_inputs: MultimodalInputs):
        helper = MultiModalityDataPaddingPatternMultimodalTokens(
            [image_inputs.im_token_id]
        )
        return helper.pad_input_tokens(input_ids, image_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]):

        images_spatial_crop = torch.cat(
            [item.image_spatial_crop for item in items], dim=0
        )

        assert images_spatial_crop.dim() == 3

        # TODO: can it be batched ?
        images_in_this_batch = []
        for item in items:
            assert item.pixel_values.dim() == 4
            image_feature = self.vision.forward_features(
                item.pixel_values.type(next(self.vision.parameters()).dtype).to(
                    device=next(self.vision.parameters()).device
                )
            )
            images_embeds = self.projector(image_feature)
            _, hw, n_dim = images_embeds.shape
            h = w = int(hw**0.5)
            tile_index = 0
            for jdx in range(item.image_spatial_crop.shape[1]):
                num_width_tiles, num_height_tiles = item.image_spatial_crop[0, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embeds[tile_index]

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[
                    tile_index + 1 : tile_index + 1 + num_tiles_in_image
                ]
                tile_index += num_tiles_in_image + 1

                # format global and local features
                # ----------------- global view add newline -----------------
                # [hw, D] -> [h, w, D]
                global_features = global_features.view(h, w, n_dim)

                # [D]     -> [h, 1, D]
                new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)

                # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                global_features = torch.cat(
                    [global_features, new_lines_in_global], dim=1
                )

                # [h, w + 1, D] -> [h * (w + 1), D]
                global_features = global_features.view(-1, n_dim)

                # ----------------- local view add newline -----------------
                # [num_height_tiles * num_width_tiles, h * w, D] ->
                # [num_height_tiles * h, num_width_tiles * w, D]
                local_features = rearrange(
                    local_features,
                    "(th tw) (h w) d -> (th h) (tw w) d",
                    th=num_height_tiles,
                    tw=num_width_tiles,
                    h=h,
                    w=w,
                )

                # [D] -> [num_height_tiles * h, 1, D]
                new_lines_in_local = repeat(
                    self.image_newline,
                    "d -> (th h) 1 d",
                    th=num_height_tiles,
                    h=h,
                )

                # [num_height_tiles * h, num_width_tiles * w + 1, D]
                local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                # [num_height_tiles * h, num_width_tiles * w + 1, D]
                #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                local_features = local_features.view(-1, n_dim)

                # merge global and local tiles
                if self.global_view_pos == "head":
                    global_local_features = torch.cat(
                        [
                            global_features,
                            self.view_seperator[None, :],
                            local_features,
                        ]
                    )
                else:
                    global_local_features = torch.cat(
                        [
                            local_features,
                            self.view_seperator[None, :],
                            global_features,
                        ]
                    )

                images_in_this_batch.append(global_local_features)

        return torch.cat(images_in_this_batch, dim=0)


EntryClass = DeepseekVL2ForCausalLM
