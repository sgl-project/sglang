# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
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
import warnings

import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel, PretrainedConfig
from transformers.models.clip.modeling_clip import CLIPAttention
from transformers.utils import logging

try:
    from flash_attn import flash_attn_func
except ImportError:
    pass

logger = logging.get_logger(__name__)


MAX_INPUT_ID = int(1e9)

CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(
  attention_dropout=0.0,
  dropout=0.0,
  hidden_act="quick_gelu",
  hidden_size=1024,
  image_size=336,
  initializer_factor=1.0,
  initializer_range=0.02,
  intermediate_size=4096,
  layer_norm_eps=1e-05,
  num_attention_heads=16,
  num_channels=3,
  num_hidden_layers=24,
  patch_size=14,
  projection_dim=768
)

class CLIPAttentionFA2(CLIPAttention):
    """Add flash attention 2 to CLIPAttention. (This is only used in the vision encoder)"""

    def forward(self,
        hidden_states,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=False,
    ):
        """Input shape: Batch x Time x Channel"""

        assert attention_mask is None, "CLIPAttentionFA2 does not support attention_mask"
        assert causal_attention_mask is None, "CLIPAttentionFA2 does not support causal_attention_mask"
        assert output_attentions is False, "CLIPAttentionFA2 does not support output_attentions"

        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states).reshape(bsz, tgt_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(bsz, tgt_len, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(bsz, tgt_len, self.num_heads, self.head_dim)

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        ).reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output, None


class Phi3ImageEmbedding(nn.Module):
    """Phi3 Image embedding."""

    def __init__(self, config: PretrainedConfig, wte=None, **kwargs) -> None:
        super().__init__()

        # n_embed or hidden_size
        hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size
        if hasattr(config, 'embd_pdrop') or hasattr(config, 'embed_pdrop'):
            embd_drop = config.embd_pdrop if hasattr(config, 'embd_pdrop') else config.embed_pdrop
            self.drop = nn.Dropout(embd_drop)
        else:
            self.drop = None

        self.wte = wte

        if isinstance(config.img_processor, dict) and config.img_processor.get('name', None) == 'clip_vision_model':
            assert 'model_name' in config.img_processor, 'model_name must be provided for CLIPVisionModel'
            assert 'image_dim_out' in config.img_processor, 'image_dim_out must be provided for CLIPVisionModel'
            assert 'num_img_tokens' in config.img_processor, 'num_img_tokens must be provided for CLIPVisionModel'
            assert config.img_processor['model_name'] == 'openai/clip-vit-large-patch14-336'
            clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
            self.img_processor = CLIPVisionModel(clip_config)
            image_dim_out = config.img_processor['image_dim_out']
            self.num_img_tokens = config.img_processor['num_img_tokens']

            # FA2 in CLIP
            if config._attn_implementation == 'flash_attention_2':
                for layer in self.img_processor.vision_model.encoder.layers:
                    clip_fa2 = CLIPAttentionFA2(clip_config)
                    del layer.self_attn
                    layer.self_attn = clip_fa2
        else:
            raise NotImplementedError(f'img_processor = {config.img_processor}, not implemented')

        self.image_dim_out = image_dim_out
        self.img_sizes = None

        # global_gn and sub_gn for hd transform, serves as line separator
        self.use_hd_transform = kwargs.get('use_hd_transform', False)
        self.with_learnable_separator = kwargs.get('with_learnable_separator', False)
        self.hd_transform_order = kwargs.get('hd_transform_order', 'glb_sub')
        # with_hd_transform and with_learnable_separator should have same value
        assert self.use_hd_transform == self.with_learnable_separator, 'use_hd_transform and with_learnable_separator should have same value'
        if self.with_learnable_separator:
            assert self.use_hd_transform, 'learnable separator is only for hd transform'
            # 1024 * 4, merge spatial to channel dimension
            self.glb_GN = nn.Parameter(torch.zeros([1, 1, self.image_dim_out * 4]))
            self.sub_GN = nn.Parameter(torch.zeros([1, 1, 1, self.image_dim_out * 4]))
            logger.info(f'learnable separator enabled for hd transform, hd_transform_order = {self.hd_transform_order}')

        projection_cls = kwargs.get('projection_cls', 'linear')
        if projection_cls == 'linear':
            self.img_projection = nn.Linear(image_dim_out, hidden_size)
        elif projection_cls == 'mlp' and self.use_hd_transform:
            dim_projection = hidden_size
            depth = 2
            layers = [nn.Linear(image_dim_out * 4, dim_projection)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(),
                                nn.Linear(dim_projection, dim_projection)])
            self.img_projection = nn.Sequential(*layers)
        elif projection_cls == 'mlp':
            dim_projection = hidden_size
            depth = 2
            layers = [nn.Linear(image_dim_out, dim_projection)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(),
                                nn.Linear(dim_projection, dim_projection)])
            self.img_projection = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f'projection_cls = {projection_cls}, not implemented')

        self.vocab_size = config.vocab_size
        self.img_features = None

        if isinstance(config.img_processor, dict):
            self.layer_idx = config.img_processor.get('layer_idx', -2)
            self.type_feature = config.img_processor.get('type_feature', 'patch')
        else:
            self.layer_idx = -2
            self.type_feature = 'patch'


    def set_img_features(self, img_features: torch.FloatTensor) -> None:
        self.img_features = img_features

    def set_img_sizes(self, img_sizes: torch.LongTensor) -> None:
        self.img_sizes = img_sizes

    def get_img_features(self, img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        LAYER_IDX = self.layer_idx
        TYPE_FEATURE = self.type_feature

        img_processor_output = self.img_processor(img_embeds, output_hidden_states=True)
        img_feature = img_processor_output.hidden_states[LAYER_IDX]

        if TYPE_FEATURE == "patch":
            patch_feature = img_feature[:, 1:]
            return patch_feature

        raise NotImplementedError

    def forward(
        self, input_ids: torch.LongTensor, pixel_values: torch.FloatTensor, image_sizes=None
    ) -> torch.FloatTensor:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        # positions for image tokens
        positions = torch.nonzero((input_ids < 0) & (input_ids > -MAX_INPUT_ID), as_tuple=True)
        has_image = len(positions[0].tolist()) > 0
        # input_ids = input_ids.clamp_min(0).clamp_max(self.vocab_size).detach()
        input_ids.clamp_min_(0).clamp_max_(self.vocab_size)
        warnings.warn(
            "Phi-3-V modifies `input_ids` in-place and the tokens indicating images will be "
            "removed after model forward. If your workflow requires multiple forward passes on "
            "the same `input_ids`, please make a copy of `input_ids` before passing it to the "
            "model."
        )

        hidden_states = self.wte(input_ids)

        if has_image:
            assert self.use_hd_transform
            num_images, num_crops, c, h, w = pixel_values.shape
            assert c == 3 and h == w == 336
            img_features = self.get_img_features(pixel_values.flatten(0, 1)).reshape(
                num_images, num_crops, -1, self.image_dim_out
            )
            image_features_proj = self.hd_feature_transform(img_features, image_sizes)
            hidden_states = hidden_states.index_put(
                positions, image_features_proj, accumulate=False
            )

        if self.drop is not None:
            hidden_states = self.drop(hidden_states)

        return hidden_states

    def hd_feature_transform(self, image_features, image_sizes):
        """
        image_features: (num_images, num_crops+1, 24*24, 1024)
        """
        assert (
            self.hd_transform_order == 'sub_glb'
        ), f'hd_transform_order `{self.hd_transform_order}` not implemented'
        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        global_image_features = image_features[:, 0]  # (num_images, 24*24, 1024)
        # global feature can be viewed as a special HD case with num_crops 1x1
        global_image_features_hd = self.reshape_hd_patches_2x2merge(global_image_features, 1, 1)
        global_image_features_hd_newline = self.add_image_newline(global_image_features_hd)

        all_image_embeddings = []
        # need a for loop to process each image because of different image sizes
        # (patch arrangement is different for each image)
        for i, img_size in enumerate(image_sizes):
            h, w = img_size
            h_crop = h // 336
            w_crop = w // 336
            num_crops = h_crop * w_crop

            # NOTE: real num_crops is padded
            # (num_crops, 24*24, 1024)
            sub_image_features = image_features[i, 1 : 1 + num_crops]
            sub_image_features_hd = self.reshape_hd_patches_2x2merge(
                sub_image_features, h_crop, w_crop
            )
            sub_image_features_hd_newline = self.add_image_newline(sub_image_features_hd)

            # [sub features, separator, global features]
            all_image_embeddings.extend(
                [
                    sub_image_features_hd_newline.squeeze(0),  # (h_crop*12*(w_crop*12+1), 4096)
                    self.glb_GN.squeeze(0),
                    global_image_features_hd_newline[i],
                ]
            )

        image_features_proj = self.img_projection(
            torch.cat(all_image_embeddings, dim=0).to(target_device).to(target_dtype)
        )

        return image_features_proj

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        """
        image_features: (num_images*num_crops, 24*24, 1024)
        output: (num_images, h_crop*12, w_crop*12, 4096), h_crop*w_crop == num_crops
        """
        N, L, C = image_features.shape
        assert L == 24 * 24 and C == 1024 and N % (h_crop * w_crop) == 0
        num_images = N // (h_crop * w_crop)
        H = int(L**0.5)
        image_features_hd = (
            image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
            .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
            .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
            .reshape(N, -1, 4 * C)  # N, 144, 4096
            .reshape(
                num_images, h_crop, w_crop, H // 2, H // 2, -1
            )  # n_img, h_crop, w_crop, 12, 12, 4096
            .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
            .reshape(
                num_images, h_crop * H // 2, w_crop * H // 2, 4 * C
            )  # n_img, h_crop*12, w_crop*12, 4096
        )

        # alternative implementation using einops
        # from einops import rearrange
        # image_features_nhwc = rearrange(
        #     image_features,
        #     'N (H W) c -> N H W c',
        #     H=H,
        #     W=H,
        # )
        # image_features_2x2merge = rearrange(
        #     image_features_nhwc,
        #     'N (h h_pool) (w w_pool) c -> N h w (h_pool w_pool c)',
        #     h_pool=2,
        #     w_pool=2,
        # )
        # image_features_hd = rearrange(
        #     image_features_2x2merge,
        #     '(n_img h_crop w_crop) h w C -> n_img (h_crop h) (w_crop w) C',
        #     h_crop=h_crop,
        #     w_crop=w_crop,
        # )

        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        # add the newline token to the HD image feature patches
        newline_embeddings = self.sub_GN.expand(num_images, h, -1, -1)  # (n_img, h, 1, hid_dim)
        image_features_hd_newline = torch.cat(
            [image_features_hd, newline_embeddings], dim=2
        ).reshape(num_images, -1, hid_dim)
        return image_features_hd_newline