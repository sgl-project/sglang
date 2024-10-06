import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import einops
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2Config
from vllm.model_executor.layers.activation import QuickGELU

from sglang.srt.model_executor.forward_batch_info import InputMetadata
from sglang.srt.models.qwen2 import Qwen2ForCausalLM


# TODO: This should be a config in huggingface...
@dataclass
class MolmoVisionBackboneConfig:
    image_default_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14
    image_pos_patch_size: int = 14
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_key_value_heads: int = 16
    image_num_layers: int = 23
    image_mlp_dim: int = 4096
    image_mlp_activations: str = "quick_gelu"
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


@dataclass
class PlaceholderConfig:
    hidden_size: int = 3584
    intermediate_size: int = 37888


class MolmoViTMLP(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.image_emb_dim, config.image_mlp_dim, bias=False)
        self.w2 = nn.Linear(config.image_mlp_dim, config.image_emb_dim, bias=False)
        self.act = QuickGELU()

    def forward(self, x: torch.Tensor):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x


class MolmoMultiHeadDotProductAttention(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig, num_vit_layers: int = 1):
        super().__init__()
        self.config = config
        self.num_heads = config.image_num_heads
        self.num_kv_heads = config.image_num_key_value_heads
        self.head_dim = config.image_emb_dim // config.image_num_heads
        self.num_vit_layers = num_vit_layers

        self.wq = nn.Linear(
            config.image_emb_dim * num_vit_layers,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            config.image_emb_dim * num_vit_layers,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            config.image_emb_dim * num_vit_layers,
            self.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.out = nn.Linear(
            self.num_heads * self.head_dim, config.image_emb_dim, bias=False
        )

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.view(
            hidden_states.shape[:2] + (self.num_heads, self.head_dim)
        )

    def _merge_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.view(
            hidden_states.shape[:2] + (self.num_heads * self.head_dim,)
        )

    def forward(self, inputs_q: torch.Tensor, inputs_kv: torch.Tensor = None):
        if inputs_kv is None:
            inputs_k = inputs_q
            inputs_v = inputs_q
        else:
            inputs_k = inputs_kv
            inputs_v = inputs_kv

        # (bsz, seq_len, embed_dim)
        q = self.wq(inputs_q)
        k = self.wk(inputs_k)
        v = self.wv(inputs_v)

        # TODO: Implement flash attention or xformers
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # (bsz, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # (bsz, num_heads, seq_len, head_dim) -> transpose to (bsz, seq_len, num_heads, head_dim)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        ).transpose(1, 2)

        # (bsz, seq_len, num_heads, head_dim) -> (bsz, seq_len, embed_dim)
        attn_output = self._merge_heads(attn_output)
        output = self.out(attn_output)

        return output


class MolmoResidualAttentionBlock(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()
        self.config = config
        self.attention = MolmoMultiHeadDotProductAttention(config)
        self.feed_forward = MolmoViTMLP(config)
        self.attention_norm = nn.LayerNorm(
            config.image_emb_dim, eps=config.image_norm_eps
        )
        self.ffn_norm = nn.LayerNorm(config.image_emb_dim, eps=config.image_norm_eps)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states)
        )
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


class BlockCollection(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()
        self.config = config
        self.resblocks = nn.ModuleList(
            [
                MolmoResidualAttentionBlock(config)
                for _ in range(config.image_num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        hidden_states = []
        for block in self.resblocks:
            x = block(x)
            hidden_states.append(x)

        return hidden_states


class VisionTransformer(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()
        self.config = config
        self.scale = config.image_emb_dim**-0.5
        self.patch_num = config.image_num_patch
        self.num_prefix_tokens = 1
        self.class_embedding = nn.Parameter(torch.randn(config.image_emb_dim))
        self.positional_embedding = nn.Parameter(
            torch.randn(config.image_num_pos, config.image_emb_dim)
        )
        self.patch_embedding = nn.Linear(
            config.image_patch_size * config.image_patch_size * 3,
            config.image_emb_dim,
        )
        self.transformer = BlockCollection(config)
        self.pre_ln = nn.LayerNorm(config.image_emb_dim, eps=config.image_norm_eps)

    def _expand_token(self, token: torch.Tensor, batch_size: int) -> torch.Tensor:
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def _add_pos_embed(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        cls_embed = self.positional_embedding[0:1]
        pos_embed = self.positional_embedding[1:]

        pos_embed = pos_embed.reshape(
            (
                int(math.sqrt(pos_embed.shape[0])),
                int(math.sqrt(pos_embed.shape[0])),
                pos_embed.shape[1],
            )
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_embed.shape[0] != patch_num_0 or pos_embed.shape[1] != patch_num_1:
            # (1, patch_h, patch_w, emb_dim) -> (1, emb_dim, patch_h, patch_w)
            pos_embed = pos_embed.unsqueeze(0).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed,
                size=(patch_num_0, patch_num_1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

            # (1, emb_dim, patch_h, patch_w) -> (patch_h, patch_w, emb_dim)
            pos_embed = pos_embed.permute(0, 2, 3, 1).squeeze(0)

        pos_embed = pos_embed.reshape(-1, pos_embed.shape[-1])

        # cat: (1, embed_dim) + (patch_h * patch_w, embed_dim) -> (1, patch_h * patch_w + 1, embed_dim)
        x = x + torch.cat([cls_embed[None, :, :], pos_embed[None, :, :]], dim=1).to(
            x.dtype
        )
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> List[torch.Tensor]:
        if patch_num is None:
            patch_num = self.patch_num

        B, N, D = x.shape
        x = self.patch_embedding(x)

        # Expand the class embedding to batch size then concate along the patch dimension
        x = torch.cat(
            [self._expand_token(self.class_embedding, x.shape[0]).to(x.device), x],
            dim=1,
        )
        x = self._add_pos_embed(x, patch_num)
        x = self.pre_ln(x)
        hidden_states = self.transformer(x)
        return hidden_states


class MolmoMultiModalProjector(nn.Module):
    def __init__(
        self,
        config: PlaceholderConfig,
        input_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(
            input_dim or config.hidden_size, config.intermediate_size // 2, bias=False
        )
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(
            input_dim or config.hidden_size, config.intermediate_size // 2, bias=False
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        # TODO: Impelement efficient fused SiluAndMul
        intermediate = torch.cat([self.w1(x), self.w3(x)], dim=-1)
        x = self.act(intermediate)
        x = self.w2(x)
        return x


class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()

        self.config = config
        self.vit_layers = [-2, -9]
        self.image_vit = VisionTransformer(config)
        self.image_projector = MolmoMultiModalProjector(
            PlaceholderConfig(),
            input_dim=config.image_emb_dim,
        )
        self.pad_embed = nn.Parameter(
            torch.zeros((2, config.image_emb_dim * len(self.vit_layers)))
        )
        self.image_pooling_2d = MolmoMultiHeadDotProductAttention(
            config, num_vit_layers=len(self.vit_layers)
        )

        self.image_num_patch = config.image_num_patch
        self.llm_patches_per_crop = (
            (self.image_num_patch[0] + 1) // 2,
            (self.image_num_patch[1] + 1) // 2,
        )

    def encode_image(self, images: torch.Tensor):
        B, T, N, D = images.shape

        # Mask checks if all tokens are equal to -1
        mask = ~torch.all(images.view(B * T, N, D) == -1, dim=(1, 2), keepdim=True)
        images = images.view(B * T, N, D)

        hidden_states = self.image_vit(images)

        features = []
        for layer in self.vit_layers:
            features.append(hidden_states[layer])
            image_features = torch.cat(features, dim=-1)

        # Since num_prefix_tokens is always 1, we can just take the first token
        cls_embed = image_features[:, 0]
        image_features = image_features[:, 1:]

        image_features = image_features * mask
        image_features = image_features.view(B, T, N, -1)

        cls_embed = cls_embed.view(B, T, -1)

        return image_features, cls_embed

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor):
        batch_size, num_images = images.shape[:2]
        image_features, cls_embed = self.encode_image(images)

        assert image_masks is not None
        pad_embed = self.pad_embed[:, None, None, None, :]
        all_pad = image_masks == 0
        partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(
            dtype=torch.float32
        )
        all_pad = all_pad.to(dtype=torch.float32)
        image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
        image_features = image_features + pad_embed[1] * torch.unsqueeze(
            partial_pad, -1
        )

        # Dropout was none
        image_features = image_features.reshape(
            (batch_size, num_images) + self.config.image_num_patch + (-1,)
        )

        if self.image_num_patch[0] % 2 == 1:
            # Pad so we can still pool 2x2 patches
            image_features = F.pad(
                image_features,
                (0, 0, 0, 1, 0, 1, 0, 0, 0, 0),
            )

        print(image_features.shape)
        # Divide into 2x2 patches
        image_features = einops.rearrange(
            image_features,
            "b n (h dh) (w dw) c -> (b n h w) (dh dw) c",
            dh=2,
            dw=2,
        )
        print(image_features.shape)

        query = image_features.mean(-2, keepdim=True)
        print(query.shape)
        image_features = self.image_pooling_2d(query, image_features)

        h, w = self.llm_patches_per_crop
        image_features = image_features.reshape(batch_size, num_images, h * w, -1)

        # MLP layer to map the feature.
        image_features = self.image_projector(image_features)

        # image_features: (batch_size, num_image, num_patch, d_model)
        # cls_embed: (batch_size, num_image, d_model)
        return image_features, cls_embed


class MolmoForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config, vision_config: MolmoVisionBackboneConfig):
        super().__init__()
        self.config = config
        self.transformer = Qwen2ForCausalLM(config)
        self.vision_backbone = MolmoVisionBackbone(vision_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        image_inputs = input_metadata.image_inputs

        has_image = input_metadata.image_inputs is not None

        if input_metadata.forward_mode.is_extend():
            x = self.transformer.embed_tokens(input_ids)

            if has_image:
                images = image_inputs["images"]
                image_masks = image_inputs["image_masks"]
                image_input_idx = image_inputs["image_input_idx"]
                seq_len = len(input_ids)

                images = images.to(device=x.device, dtype=x.dtype)
                image_features, cls_embed = self.vision_backbone(
                    images=images, image_masks=image_masks
                )
                batch_size = images.shape[0]
                num_image, num_patch = image_features.shape[1:3]

                assert image_input_idx.shape == (batch_size, num_image, num_patch)

                image_features = image_features.to(x.device)
                image_features = image_features.view(
                    batch_size, num_image * num_patch, -1
                )
                image_input_idx = image_input_idx.view(batch_size, num_image, num_patch)

                valid = image_input_idx >= 0
                image_features = image_features * valid[:, :, None].to(
                    dtype=image_features.dtype
                )
                image_features = image_features.view(
                    batch_size, num_image * num_patch, -1
                ).contiguous()

                image_input_idx = image_input_idx * valid.to(dtype=image_features.dtype)
                image_input_idx = image_input_idx * valid.to(image_input_idx.dtype)
                offset = torch.cat(
                    [seq_len.new_zeros((1)), seq_len.cumsum(dim=0)[:-1]], dim=0
                )[:, None]
                image_input_idx = image_input_idx + offset.to(image_input_idx.dtype)
                image_input_idx = image_input_idx.flatten()[:, None]
                mat = (
                    image_input_idx
                    == torch.arange(seq_len.sum().item(), device=x.device)[None, :]
                )
                mat = mat.to(image_features.dtype)
                x = x + torch.einsum("nd,nm->md", image_features, mat)

            input_embeds = x
            input_ids = None

            return self.transformer(
                input_ids, positions, input_metadata, input_embeds=input_embeds
            )
        elif input_metadata.forward_mode.is_decode():
            return self.transformer(input_ids, positions, input_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            print(name)
            if "vision_backbone" in name:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            else:
                if "ln_f.weight" in name:
                    name = "model.norm.weight"

                if "transformer.blocks" in name:
                    name = name.replace("transformer.blocks", "layers")

                if "attn_out" in name:
                    name = name.replace("attn_out", "self_attn.o_proj")

                if "att_proj" in name:
                    name = name.replace("att_proj", "self_attn.qkv_proj")

                if "q_norm" in name:
                    name = name.replace("q_norm", "self_attn.q_norm")

                if "k_norm" in name:
                    name = name.replace("k_norm", "self_attn.k_norm")

                if "ff_proj" in name:
                    name = name.replace("ff_proj", "mlp.gate_up_proj")
                    assert "weight" in name
                    up_weight, gate_weight = loaded_weight.chunk(2, dim=0)
                    loaded_weight = torch.cat([gate_weight, up_weight], dim=0)

                if "ff_out" in name:
                    if "layers" in name:
                        name = name.replace("ff_out", "mlp.down_proj")
                    else:
                        # lm head
                        name = name.replace("model.transformer.ff_out", "lm_head")

                if "attn_norm" in name:
                    name = name.replace("attn_norm", "input_layernorm")

                if "ff_norm" in name:
                    name = name.replace("ff_norm", "post_attention_layernorm")
                self.transformer.load_weights([(name, loaded_weight)])


EntryClass = [MolmoForCausalLM]

if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     "allenai/Molmo-7B-D-0924",
    #     trust_remote_code=True,
    #     torch_dtype="auto",
    #     device_map="auto",
    # )
    # model = CLIPVisionModel.from_pretrained(
    #         "openai/clip-vit-large-patch14-336", torch_dtype=torch.float16
    #     )
    # breakpoint()
    inputs = processor.process(
        # images=[
        #     Image.open(
        #         requests.get("https://picsum.photos/id/237/536/354", stream=True).raw
        #     )
        # ],
        text="Describe this image.",
    )
    print(inputs)
    # qwen2config = Qwen2Config.from_pretrained("allenai/Molmo-7B-D-0924")
    # vision_config = MolmoVisionBackboneConfig()
    # model = MolmoModel(qwen2config, vision_config)
    # model.eval()
    # print(model)

    # input_metadata = InputMetadata(
    #     forward_mode=ForwardMode.EXTEND,
    #     batch_size=1,
    #     image_inputs=inputs,
    # )
    # Takes in bs, num_crops, num_patches_h * num_patches_w, patch_size * patch_size * 3
    #                  -> bs, num_crops, num_patches_h * num_patches_w, image_embed_dim * 2
    # input_embeds = model(
    #     inputs["input_ids"],
    #     None,
    #     input_metadata,
    # )
    # print(input_embeds.shape)
    # print(cls_embed.shape)
