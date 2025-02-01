from typing import List,Optional,Tuple,Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig
import math
from functools import partial
from einops import rearrange, repeat

from sglang.srt.configs import DeepseekVL2Config
from sglang.srt.layers.layernorm import RMSNorm

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (ParallelLMHead,VocabParallelEmbedding)

from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

def _no_grad_tunc_normal_(tensor,mean,std,a,b):
    def norm_cdf(x):
        return (1.0+math.erf(x/math.sqrt(2.0)))/2.0
    
    with torch.no_grad():
        l=norm_cdf((a-mean)/std)
        u=norm_cdf((b-mean)/std)
        
        tensor.uniform_(2*l-1,2*u-1)
        
        tensor.erfinv_()
        
        tensor.mul_(std*math.sqrt(2.0))
        tensor.add_(mean)
        
        tensor.clamp_(min=a,max=b)
        return tensor

def trunc_normal_(tensor,mean=0.0,std=1.0,a=-2.0,b=2.0):
    with torch.no_grad():
        dtype=tensor.dtype
        tensor_fp32=tensor.float()
        tensor_fp32=_no_grad_tunc_normal_(tensor_fp32,mean,std,a,b)
        tensor_dtpye=tensor_fp32.to(dtype=dtype)
        tensor.copy_(tensor_dtpye)

def init_weights(self):
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed,std=self.pos_embed.shape[1]**-0.5)
    trunc_normal_(self.latent,std=self.latent_dim**-0.5)

def init_weights_vit_timm(module:nn.Module,)->None:
    if isinstance(module,nn.Linear):
        trunc_normal_(module.weight,std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module,'init_weights'):
        module.init_weights()


class DeepseekVL2VisionTransformer(nn.Module):
    def __init__(self,img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str='map',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            ignore_head: bool = False,
            deterministic: bool = False,
            num_recomputing_layers: int = 0):
        
        super.__init__()
        norm_layer=partial(nn.LayerNorm,eps=1e-6)
        act_layer=partial(nn.GELU,approximate='tanh')
        
        self.num_classes=num_classes #always 0
        self.global_pool=global_pool #always map
        self.num_features = self.embed_dim = embed_dim #diff
        self.num_prefix_tokens = 1
        self.num_prefix_tokens += reg_tokens #always 0
        self.has_class_token = class_token #always true
        self.no_embed_class = no_embed_class  #always false
        self.dynamic_img_size = dynamic_img_size #always false
        self.grad_checkpointing = False #always false
        self.ignore_head = ignore_head #always false
        
        embed_args = {}
        self.patch_embed=PatchEmbed(img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,)
        
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        
        self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()  #always false
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                deterministic=deterministic,
            )
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        if global_pool == 'map':
            AttentionPoolLatent.init_weights = init_weights
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # 初始化权重
        if weight_init != 'skip':
            self.init_weights(weight_init)
    
    def init_weights(self, mode = '') -> None:
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)
    
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:

        to_cat = []
        pos_embed = self.pos_embed
        #need check
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "is_first_stage", True):
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
        x = self.blocks(x)
        if getattr(self, "is_last_stage", True):
            x = self.norm(x)
        return x

class DeepseekVL2MlpProjector(nn.Module):
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
class DeepseekVL2ForCausalLM(nn.Module):

    def __init__(self, config: DeepseekVL2Config):
        super().__init__(config)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # ----------- vision encoder ------------
        vision_config = config.vision_config
        self.vision = DeepseekVL2VisionTransformer(
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
        self.projector = DeepseekVL2MlpProjector(projector_config)

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        embed_std = 1 / torch.sqrt(torch.tensor(projector_config.n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            self.image_newline = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
        else:
            raise ValueError(f"tile tag should be 2D, but got {self.tile_tag}")

        # ----------- language model ------------
        language_config = config.language_config
        self.language = DeepseekV2ForCausalLM(language_config)
        
    def forward(self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
                
        outputs = self.language.forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch
        )

        return outputs

    def prepare_inputs_embeds(
            self,
            input_ids: torch.LongTensor,
            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,
            **ignore_kwargs
    ):
        #todo: find the get_input_embeddings api
        if images is None or images_spatial_crop.sum() == 0:
            return self.language.get_input_embeddings()(input_ids)
        
        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += (1 + num_width_tiles * num_height_tiles)

            total_tiles.append(images[idx, :batch_num_tiles[idx]])

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)
        if total_tiles.shape[0] == 0:
            return self.language.get_input_embeddings()(input_ids)

        # [batch_all_tiles, vit_seq_len, c]
        images_feature = self.vision(total_tiles)

        # [batch_all_tiles, hw, D]
        images_embeds = self.projector(images_feature)
        _, hw, n_dim = images_embeds.shape
        h = w = int(hw ** 0.5)

        # put image tokens into the input_embeds, [b, T, D]
        input_embeds = self.language.get_input_embeddings()(input_ids)

        # 根据self.tile_tag & self.global_view_pos填充image token sequence
        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):

                # extra global & local features
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embeds[tile_index]

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[tile_index + 1: tile_index + 1 + num_tiles_in_image]

                tile_index += num_tiles_in_image + 1


                # ----------------- global view add newline -----------------
                # [hw, D] -> [h, w, D]
                global_features = global_features.view(h, w, n_dim)
                # [D]     -> [h, 1, D]
                new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                global_features = torch.cat([global_features, new_lines_in_global], dim=1)
                # [h, w + 1, D] -> [h * (w + 1), D]
                global_features = global_features.view(-1, n_dim)

                # ----------------- local view add newline -----------------
                # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                local_features = rearrange(
                    local_features,
                    "(th tw) (h w) d -> (th h) (tw w) d",
                    th=num_height_tiles,
                    tw=num_width_tiles,
                    h=h,
                    w=w
                )

                # [D] -> [num_height_tiles * h, 1, D]
                new_lines_in_local = repeat(
                    self.image_newline,
                    "d -> (th h) 1 d",
                    th=num_height_tiles,
                    h=h
                )

                # [num_height_tiles * h, num_width_tiles * w + 1, D]
                local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                # [num_height_tiles * h, num_width_tiles * w + 1, D]
                #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                local_features = local_features.view(-1, n_dim)

                # ----------------- merge global and local tiles -----------------
                if self.global_view_pos == "head":
                    global_local_features = torch.cat(
                        [global_features, self.view_seperator[None, :], local_features], dim=0)
                else:
                    global_local_features = torch.cat(
                        [local_features, self.view_seperator[None, :], global_features], dim=0)

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                input_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1), images_in_this_batch)

        return input_embeds
    
EntryClass= DeepseekVL2ForCausalLM