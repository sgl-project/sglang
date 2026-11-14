import math

import torch
import torch.nn.functional as F
from torch import nn


@torch.compile(dynamic=True, fullgraph=True)
def _renormalize(x, mean, std):
    rgb = x * std.to(x.dtype) + mean.to(x.dtype)
    return 2.0 * rgb - 1.0


def _position_embeddings(grid, y, x, heights, widths):
    if grid.is_cuda:
        from sglang.kernels.ops.inkling2.vision import (
            bilinear_interpolate_rows,
        )

        return bilinear_interpolate_rows(grid, y, x, heights, widths)

    def weights(indices, targets, size):
        center = (size / targets.float()) * (indices.float() + 0.5)
        distance = torch.arange(size, device=grid.device)[None, :] + 0.5
        weights = (1 - (distance - center[:, None]).abs()).clamp(min=0)
        return weights / weights.sum(dim=1, keepdim=True)

    result = torch.einsum(
        "ni,nj,ijd->nd",
        weights(y, heights, grid.shape[0]),
        weights(x, widths, grid.shape[1]),
        grid.float(),
    )
    return result.to(grid.dtype)


def _rotary(q, k, positions):
    if q.is_cuda:
        from sglang.kernels.ops.inkling2.vision import apply_rotary_qk

        return apply_rotary_qk(q, k, positions)
    dim = q.shape[-1]
    frequencies = 1.0 / (
        10000 ** (torch.arange(0, dim, 4, device=q.device, dtype=torch.float32) / dim)
    )
    angles = torch.cat(
        (positions[:, 1, None] * frequencies, positions[:, 0, None] * frequencies),
        dim=-1,
    )
    phase = torch.polar(torch.ones_like(angles), angles)
    cos, sin = phase.real[:, None, :], phase.imag[:, None, :]

    def rotate(value):
        pairs = value.float().reshape(*value.shape[:-1], -1, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        return (
            torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
            .flatten(-2)
            .to(value.dtype)
        )

    return rotate(q), rotate(k)


class Inkling2VisionAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.num_heads = heads
        self.head_dim = width // heads
        self.q_proj = nn.Linear(width, width, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(width, width, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(width, width, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(width, width, dtype=torch.bfloat16)

    def forward(self, hidden, positions, lengths, cu_seqlens):
        shape = (-1, self.num_heads, self.head_dim)
        q, k, v = (
            projection(hidden).reshape(shape)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        q, k = _rotary(q, k, positions)
        if hidden.is_cuda:
            from sglang.kernels.ops.attention.flash_attn.cute import (
                flash_attn_varlen_func,
            )

            if torch.cuda.get_device_capability(hidden.device)[0] not in (9, 10):
                raise ValueError(
                    "Inkling2 vision attention requires Hopper or Blackwell"
                )
            max_length = max(257, sum(lengths) - len(lengths) + 1)
            output, _ = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_length,
                max_seqlen_k=max_length,
                softmax_scale=self.head_dim**-0.5,
                causal=False,
                deterministic=True,
            )
        else:
            output = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        qi.transpose(0, 1).unsqueeze(0),
                        ki.transpose(0, 1).unsqueeze(0),
                        vi.transpose(0, 1).unsqueeze(0),
                        scale=self.head_dim**-0.5,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                    for qi, ki, vi in zip(
                        q.split(lengths), k.split(lengths), v.split(lengths)
                    )
                ]
            )
        return self.out_proj(output.reshape(hidden.shape))


class Inkling2VisionMLP(nn.Module):
    def __init__(self, width, intermediate_size):
        super().__init__()
        self.c_fc = nn.Linear(width, intermediate_size, dtype=torch.bfloat16)
        self.c_proj = nn.Linear(intermediate_size, width, dtype=torch.bfloat16)

    def forward(self, hidden):
        return self.c_proj(F.gelu(self.c_fc(hidden)))


class Inkling2VisionLayerScale(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))

    def forward(self, hidden):
        return hidden * self.gamma


class Inkling2VisionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, eps = config["hidden_size"], config["layer_norm_eps"]
        self.ln_1 = nn.LayerNorm(width, eps=eps, dtype=torch.bfloat16)
        self.self_attn = Inkling2VisionAttention(width, config["num_attention_heads"])
        self.ln_2 = nn.LayerNorm(width, eps=eps, dtype=torch.bfloat16)
        self.mlp = Inkling2VisionMLP(width, int(width * config["mlp_ratio"]))
        self.ls_1 = Inkling2VisionLayerScale(width)
        self.ls_2 = Inkling2VisionLayerScale(width)

    def forward(self, hidden, positions, lengths, cu_seqlens):
        hidden = hidden + self.ls_1(
            self.self_attn(self.ln_1(hidden), positions, lengths, cu_seqlens)
        )
        return hidden + self.ls_2(self.mlp(self.ln_2(hidden)))


class Inkling2VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["hidden_size"]
        self.patch_embedding = nn.Linear(
            3 * config["native_patch_size"] ** 2,
            width,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.position_embedding = nn.Parameter(
            torch.empty(config["posemb_grid_size"] ** 2, width, dtype=torch.bfloat16)
        )
        self.class_embedding = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))
        self.cls_position_embedding = nn.Parameter(
            torch.empty(width, dtype=torch.bfloat16)
        )

    def forward(self, pixels):
        return self.patch_embedding(pixels)


class Inkling2VisionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(
            config["hidden_size"],
            eps=config["layer_norm_eps"],
            dtype=torch.bfloat16,
        )
        self.layers = nn.ModuleList(
            [Inkling2VisionLayer(config) for _ in range(config["num_hidden_layers"])]
        )


class Inkling2VisionEncoder(nn.Module):
    def __init__(self, vision_config: dict, text_hidden_size: int):
        super().__init__()
        for key in (
            "temporal_patch_size",
            "native_patch_size",
            "adapter_merge_size",
            "projection_hidden_dim",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "posemb_grid_size",
        ):
            value = vision_config[key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"vision_config.{key} must be a positive integer")
        if type(text_hidden_size) is not int or text_hidden_size <= 0:
            raise ValueError("text_hidden_size must be a positive integer")
        for key in ("mlp_ratio", "layer_norm_eps"):
            value = vision_config[key]
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"vision_config.{key} must be finite and positive")
        width, heads = (
            vision_config["hidden_size"],
            vision_config["num_attention_heads"],
        )
        if width % heads or width // heads not in (4, 8, 16, 32, 64, 128):
            raise ValueError(
                "Vision head dimension must be one of 4, 8, 16, 32, 64, 128"
            )
        intermediate_size = width * vision_config["mlp_ratio"]
        if not math.isfinite(intermediate_size) or intermediate_size != int(
            intermediate_size
        ):
            raise ValueError("Vision MLP intermediate size must be an integer")
        if type(vision_config["projector_bias"]) is not bool:
            raise ValueError("vision_config.projector_bias must be a boolean")
        self.config = vision_config
        self.text_hidden_size = text_hidden_size
        self.patch_size = vision_config["native_patch_size"]
        self.merge_size = vision_config["adapter_merge_size"]
        self.temporal_patch_size = vision_config["temporal_patch_size"]
        self.backbone = Inkling2VisionBackbone(vision_config)
        hidden_size = vision_config["projection_hidden_dim"]
        self.proj_in = nn.Linear(
            self.merge_size**2 * vision_config["hidden_size"],
            hidden_size,
            bias=vision_config["projector_bias"],
            dtype=torch.bfloat16,
        )
        self.proj_out = nn.Linear(
            hidden_size,
            text_hidden_size,
            bias=vision_config["projector_bias"],
            dtype=torch.bfloat16,
        )
        for name, values in (
            ("_clip_mean", (0.48145466, 0.4578275, 0.40821073)),
            ("_clip_std", (0.26862954, 0.26130258, 0.27577711)),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, 1, 3),
                persistent=False,
            )

    def _preprocess_patches(self, patches):
        x = _renormalize(patches, self._clip_mean, self._clip_std)
        x = x.mean(dim=1)
        return x.permute(0, 3, 1, 2).contiguous().to(self.proj_out.weight.dtype)

    def _embed_images(self, patches, grid):
        m, p = self.merge_size, self.patch_size
        pixels, positions, heights, widths = [], [], [], []
        for image, (_, height, width) in zip(
            patches.split([h * w for _, h, w in grid]), grid
        ):
            native = image.reshape(height, width, 3, m, p, m, p)
            native = native.permute(0, 3, 1, 5, 4, 6, 2).reshape(-1, p * p * 3)
            pixels.extend((native.new_zeros((1, native.shape[1])), native))
            rows = torch.arange(height * width * m * m, device=patches.device)
            coordinates = torch.stack((rows // (width * m), rows % (width * m)), -1)
            positions.extend((coordinates.new_zeros((1, 2)), coordinates))
            heights.append(rows.new_full((rows.numel() + 1,), height * m))
            widths.append(rows.new_full((rows.numel() + 1,), width * m))
        pixels = torch.cat(pixels)
        positions = torch.cat(positions)
        lengths = [h * w * m * m + 1 for _, h, w in grid]
        cu_seqlens = torch.tensor(
            [0, *lengths], dtype=torch.int32, device=patches.device
        ).cumsum(0, dtype=torch.int32)
        embeddings = self.backbone.embeddings
        hidden = embeddings(pixels)
        grid_size = self.config["posemb_grid_size"]
        hidden = hidden + _position_embeddings(
            embeddings.position_embedding.reshape(grid_size, grid_size, -1),
            positions[:, 0].contiguous(),
            positions[:, 1].contiguous(),
            torch.cat(heights),
            torch.cat(widths),
        )
        cls_rows = cu_seqlens[:-1].long()
        hidden[cls_rows] = (
            embeddings.class_embedding + embeddings.cls_position_embedding
        )
        positions = (positions + 1).to(torch.int32)
        positions[cls_rows] = 0
        return self.backbone.ln_pre(hidden), positions, lengths, cu_seqlens

    def _pool(self, hidden, grid, lengths):
        m, width = self.merge_size, hidden.shape[-1]
        return torch.cat(
            [
                image[1:]
                .reshape(h, m, w, m, width)
                .permute(0, 2, 1, 3, 4)
                .reshape(h * w, m * m * width)
                for image, (_, h, w) in zip(hidden.split(lengths), grid)
            ]
        )

    def forward(self, vision_patches_bthwc, vision_grid_thw):
        if vision_patches_bthwc.dtype != torch.bfloat16:
            raise ValueError("Inkling2 image patches must use bfloat16")
        side = self.patch_size * self.merge_size
        if vision_patches_bthwc.ndim != 5 or tuple(vision_patches_bthwc.shape[1:]) != (
            self.temporal_patch_size,
            side,
            side,
            3,
        ):
            raise ValueError("Unexpected Inkling2 image patch shape")
        if (
            vision_grid_thw.device.type != "cpu"
            or vision_grid_thw.ndim != 2
            or vision_grid_thw.shape[1] != 3
        ):
            raise ValueError(
                "vision_grid_thw must be a CPU tensor with shape [num_images, 3]"
            )
        if vision_grid_thw.dtype not in (torch.int32, torch.int64):
            raise ValueError("vision_grid_thw must contain integer dimensions")
        grid = vision_grid_thw.tolist()
        if any(t != 1 or h <= 0 or w <= 0 for t, h, w in grid):
            raise ValueError(
                "Inkling2 vision requires nonempty image grids with temporal size 1"
            )
        if sum(h * w for _, h, w in grid) != vision_patches_bthwc.shape[0]:
            raise ValueError("Image grids do not match the number of image patches")
        if not grid:
            return self.proj_out.weight.new_empty((0, self.text_hidden_size))
        patches = self._preprocess_patches(vision_patches_bthwc)
        hidden, positions, lengths, cu_seqlens = self._embed_images(patches, grid)
        for layer in self.backbone.layers:
            hidden = layer(hidden, positions, lengths, cu_seqlens)
        hidden = self._pool(hidden, grid, lengths)
        return self.proj_out(F.gelu(self.proj_in(hidden)))
