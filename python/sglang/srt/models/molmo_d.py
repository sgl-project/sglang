from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from transformers import CLIPVisionModel


# TODO: This should be a config in huggingface...
@dataclass
class MolmoVisionBackboneConfig:
    image_dim: int = 2048
    vit_feature_select_layers: List[int] = field(default_factory=lambda: [-2, -9])
    num_prefix_tokens: int = 1


class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: MolmoVisionBackboneConfig):
        super().__init__()

        self.config = config
        self.image_vit = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336", torch_dtype=torch.float16
        )
        self.pad_embed = nn.Parameter(torch.zeros((2, config.image_dim)))

    def encode_images(self, images: torch.Tensor):
        breakpoint()
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
        image_features, cls_embed = self.encode_images(images)

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
            (batch_size, num_image) + image_features.shape[-2] + (-1,)
        )
        # TODO: continue forward


if __name__ == "__main__":
    model = MolmoVisionBackbone(MolmoVisionBackboneConfig()).to("cuda")
    model.eval()
    images = torch.randn(1, 3, 336, 336).to("cuda")
    image_features, cls_embed = model.encode_images(images)
    print(image_features.shape)
    print(cls_embed.shape)
