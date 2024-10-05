from dataclasses import dataclass, field
from typing import List, Tuple

import einops
import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, CLIPVisionModel
from vllm.model_executor.layers.activation import QuickGELU


# TODO: This should be a config in huggingface...
@dataclass
class VisionBackboneConfig:
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


class MolmoViTMLP(nn.Module):
    def __init__(self, config: VisionBackboneConfig):
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
    def __init__(self, config: VisionBackboneConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.image_num_heads
        self.num_kv_heads = config.image_num_key_value_heads
        self.head_dim = config.image_emb_dim // config.image_num_heads

        self.wq = nn.Linear(
            config.image_emb_dim, self.num_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(
            config.image_emb_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.image_emb_dim, self.num_kv_heads * self.head_dim, bias=False
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

    def forward(self, inputs_q: torch.Tensor, inputs_kv: torch.Tensor):
        if inputs_kv is None:
            inputs_k = inputs_q
            inputs_v = inputs_q
        else:
            inputs_k = inputs_kv
            inputs_v = inputs_kv

        # (bsz, seq_len, embed_dim)
        q = self.wq(inputs_q)
        k = self.wk(inputs_kv)
        v = self.wv(inputs_kv)

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
    def __init__(self, config: VisionBackboneConfig):
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
    def __init__(self, config: VisionBackboneConfig):
        super().__init__()
        self.config = config
        self.resblocks = nn.ModuleList(
            [
                MolmoResidualAttentionBlock(config)
                for _ in range(config.image_num_layers)
            ]
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = []
        for block in self.resblocks:
            hidden_states.append(block(hidden_states))

        return hidden_states


class VisionTransformer(nn.Module):
    def __init__(self, config: VisionBackboneConfig):
        super().__init__()
        self.config = config
        self.scale = config.image_emb_dim**-0.5
        self.patch_num = config.image_num_patch
        self.num_prefix_tokens = 1
        self.class_embedding = nn.Parameter(torch.randn(config.image_emb_dim))
        self.positional_embedding = nn.Parameter(
            torch.randn(config.image_num_pos, config.image_emb_dim)
        )
        self.patch_embedding = nn.Parameter(
            torch.randn(
                config.image_patch_size * config.image_patch_size * 3,
                config.image_emb_dim,
            )
        )
        self.transformer = BlockCollection(config)
        self.pre_ln = nn.LayerNorm(config.image_emb_dim, eps=config.image_norm_eps)

    def _expand_token(token: torch.Tensor, batch_size: int) -> torch.Tensor:
        return token.view(1, 1, -1).expand(batch_size, -1, -1)

    def _add_pos_embed(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        cls_embed = self.positional_embedding[0]
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
            self._expand_token(self.class_embedding, x.shape[0]).to(x.device), dim=1
        )
        x = self._add_pos_embed(x, patch_num)
        x = self.pre_ln(x)
        hidden_states = self.transformer(x)
        return hidden_states


class MolmoMultiModalProjector(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(
            config.image_embed_dim, self.config.hidden_size * 2, bias=False
        )
        self.w2 = nn.Linear(
            self.config.hidden_size * 2, self.config.hidden_size, bias=False
        )
        self.w3 = nn.Linear(
            config.image_embed_dim, self.config.hidden_size * 2, bias=False
        )
        self.act = QuickGELU()

    def forward(self, image_features: torch.Tensor):
        image_features = self.w2(
            self.act(self.w1(image_features), self.w3(image_features))
        )
        return image_features


class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()

        self.config = config
        self.image_vit = VisionTransformer(config)
        self.pad_embed = nn.Parameter(torch.zeros((2, config.image_dim)))

    def encode_image(self, images: torch.Tensor):
        B, T, N, D = images.shape

        # Mask checks if all tokens are equal to -1
        mask = ~torch.all(images.view(B * T, N, D) == -1, dim=(1, 2), keepdim=True)

        hidden_states = self.image_vit(images, output_hidden_states=True)[2]

        features = []
        for layer in self.config.vit_feature_select_layers:
            features.append(hidden_states[layer])
            image_features = torch.cat(features, dim=-1)
        cls_embed = image_features[:, 0]
        image_features = image_features[:, 1:]

        image_features = image_features * mask
        image_features = image_features.view(B, T, N, -1)

        cls_embed = cls_embed.view(B, T, -1) if cls_embed is not None else None

        return image_features, cls_embed

    def forward(self, images: torch.Tensor, image_masks: torch.Tensor):
        batch_size, num_images = images.shape[:2]
        image_features, cls_embed = self.encode_image(images)

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
        image_features = einops.rearrange(
            image_features,
            "b n (h dh) (w dw) c -> (b n h w) (dh dw) c",
            dh=2,
            dw=2,
        )
        image_features = self.image_pooling_2d(image_features[:, :1, :], image_features)

        h, w = self.config.image_num_patch
        h, w = (h + 2 - 1) // 2, (w + 2 - 1) // 2
        image_features = image_features.reshape(batch_size, num_images, h * w, -1)

        # MLP layer to map the feature.
        image_features = self.image_projector(image_features)

        # image_features: (batch_size, num_image, num_patch, d_model)
        # cls_embed: (batch_size, num_image, d_model)
        return image_features, cls_embed


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    # model = CLIPVisionModel.from_pretrained(
    #         "openai/clip-vit-large-patch14-336", torch_dtype=torch.float16
    #     )
    # breakpoint()
    inputs = processor.process(
        images=[
            Image.open(
                requests.get("https://picsum.photos/id/237/536/354", stream=True).raw
            )
        ],
        text="Describe this image.",
    )
    # model = MolmoVisionBackbone(MolmoVisionBackboneConfig()).to("cuda")
    # model.eval()
    print(model)

    # Takes in bs, num_crops, num_patches_h * num_patches_w, patch_size * patch_size * 3
    #                  -> bs, num_crops, num_patches_h * num_patches_w, image_embed_dim * 2
    image_features, cls_embed = model.model.vision_backbone.encode_image(
        inputs["images"].unsqueeze(0).to("cuda")
    )
    print(image_features.shape)
    print(cls_embed.shape)
