from typing import List,Optional,Tuple,Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.configs import DeepseekVLV2Config
from sglang.srt.layers.layernorm import RMSNorm

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (ParallelLMHead,VocabParallelEmbedding)

from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

class DeepseekVLV2VisionTransformer(nn.Module):
    def __init__(self,config):
        return
    

class DeepseekVLV2MlpProjector(nn.Module):
    def __init__(self,config):
        super.__init__()
        self.config=config
        
        if config.projector_type == "downsample_mlp_gelu":
            mlp_depth = config.depth
            mlp_ratio = config.mlp_ratio
            modules = [nn.Linear(config.input_dim * config.downsample_ratio * config.downsample_ratio, config.n_embed * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed * mlp_ratio, config.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.n_embed * mlp_ratio, config.n_embed))
            modules = nn.Sequential(*modules)
        
        else:
            raise NotImplementedError(
                f"Unsupported projector type: {config.projector_type}")

        self.layers = modules
    
    def forward(self, x):
        bs, hw, input_dim = x.shape
        h = w = int((hw) ** 0.5)

        """compute padding"""
        if h % self.cfg.downsample_ratio:
            pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
        else:
            pad = 0
        x = x.reshape(bs, h, w, input_dim)
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

        """4 to 1 concat"""
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        x = F.unfold(x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio,
                        padding=0)  # B, C*4, HW // 4
        x = x.permute(0, 2, 1)

        return self.layers(x)


# todo
class DeepseekVLV2ForCausalLM(nn.Module):

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__(config)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # ----------- vision encoder ------------
        vision_config = config.vision_config
        self.vision = DeepseekVLV2VisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.heads,
            mlp_ratio=vision_config.mlp_ratio,
            class_token=vision_config.class_token,
            global_pool=vision_config.global_pool,
            ignore_head=vision_config.ignore_head,
            weight_init=vision_config.weight_init,
            num_classes=0,
            deterministic=vision_config.deterministic,
            num_recomputing_layers=vision_config.num_recomputing_layers
        )

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = DeepseekVLV2MlpProjector(projector_config)

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # TODO format these code
        embed_std = 1 / torch.sqrt(torch.tensor(projector_config.n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
            # fix the typo: view_seperater
            self.view_seperator = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
        elif self.tile_tag == "1D":
            # <|tile_x|>, <|tile_global|>
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f"len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}")
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, config.aligner.params.n_embed)) * embed_std
            )
        else:
            raise ValueError(f"tile tag should be either 1D or 2D, but got {self.tile_tag}")

        # ----------- language model ------------
        language_config = config.language_config
        self.language = DeepseekV2ForCausalLM(language_config)
        
EntryClass= DeepseekVLV2ForCausalLM