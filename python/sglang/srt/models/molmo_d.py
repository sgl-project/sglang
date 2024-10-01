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
class MolmoVisionBackboneConfig:
    image_dim: int = 2048
    vit_feature_select_layers: List[int] = field(default_factory=lambda: [-2, -9])
    num_prefix_tokens: int = 1
    image_num_patch: Tuple[int, int] = (14, 14)
    image_pooling_h: int = 2
    image_pooling_w: int = 2
    hidden_size: int = 3584
    image_embed_dim: int = 1024


class MolmoMLP(nn.Module):
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
        self.image_vit = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336", torch_dtype=torch.float16
        )
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
            dh=self.config.image_pooling_h,
            dw=self.config.image_pooling_w,
        )
        image_features = self.image_pooling_2d(image_features[:, :1, :], image_features)

        h, w = self.config.image_num_patch
        h, w = (h + self.config.image_pooling_h - 1) // self.config.image_pooling_h, (
            w + self.config.image_pooling_w - 1
        ) // self.config.image_pooling_w
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
