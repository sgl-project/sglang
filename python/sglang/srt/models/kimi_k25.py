import logging
from copy import deepcopy
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import activations

from sglang.srt.configs.kimi_k25 import KimiK25Config, KimiK25VisionConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)

try:
    from transformers.activations import PytorchGELUTanh
except ImportError:
    from transformers.activations import GELUTanh

    activations.PytorchGELUTanh = GELUTanh
    PytorchGELUTanh = GELUTanh

from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM
from sglang.srt.models.kimi_vl_moonvit import MLP2
from sglang.srt.utils import add_prefix

KIMIV_VT_INFER_MAX_PATCH_NUM = 16328
logger = logging.getLogger(__name__)


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, x_shape=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """

    freqs_cis = freqs_cis.unsqueeze(-2)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xq.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


def tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    d_model = x.size(-1)

    outputs = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        # Get the current sequence
        seq = x[pre_sum : pre_sum + t * h * w]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = h // kernel_height, w // kernel_width
        reshaped_seq = seq.view(
            t, new_height, kernel_height, new_width, kernel_width, d_model
        )
        reshaped_seq = (
            reshaped_seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        )  # temporal pooling
        padded_seq = reshaped_seq.view(
            new_height * new_width, kernel_height * kernel_width, -1
        )
        outputs.append(padded_seq)
        pre_sum += t * h * w

    return outputs


class MoonViTEncoderLayer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        activation=F.gelu,
        attn_bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads

        self.norm0 = nn.LayerNorm(hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)

        self.attn = VisionAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            projection_size=hidden_dim,
            use_qkv_parallel=True,
            qkv_bias=attn_bias,
            proj_bias=attn_bias,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            use_data_parallel=use_data_parallel,
            customized_position_embedding_applier=apply_rope,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        rope_freqs_cis: torch.Tensor | None = None,
    ):
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)

        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=rope_freqs_cis,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    From:
    https://github.com/OpenGVLab/InternVideo/blob/421f6d2361fc8f61a3394244571f2601a4e99e29/InternVideo2/multi_modality/models/backbones/internvideo2/pos_embed.py#L86
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


@get_rope_shape_decorate
@torch.compile(dynamic=True)
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """
    t_size: int of the temporal size
    return:
    pos_embed: [t_size, embed_dim] or [1+t_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
                )

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support.
    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.
    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py
    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis", self._precompute_freqs_cis(device), persistent=False
            )

        shapes = grid_thws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis


class MoonVision3dPatchEmbed(nn.Module):

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
    ):
        super().__init__()
        assert isinstance(
            patch_size, int | Sequence
        ), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert (
            len(patch_size) == 2
        ), f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_dim, out_dim, kernel_size=patch_size, stride=patch_size
        )

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 3): temporal, height and width
        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.size(0), -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_thws)
        return x


class MoonViT3dEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = "spatial_temporal",
    ) -> None:
        super().__init__()

        assert (
            video_attn_type == "spatial_temporal"
        ), f'video_attn_type must be "spatial_temporal", got {video_attn_type}'
        self.video_attn_type = video_attn_type
        self.rope_2d = Rope2DPosEmbRepeated(
            block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512
        )
        self.blocks = nn.ModuleList(
            [MoonViTEncoderLayer(**block_cfg) for _ in range(num_layers)]
        )
        self.final_layernorm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws=grid_thws, device=hidden_states.device
        )

        lengths = torch.cat(
            (
                torch.zeros(1, dtype=grid_thws.dtype, device=grid_thws.device),
                grid_thws[:, 0] * grid_thws[:, 1] * grid_thws[:, 2],
            )
        )

        max_seqlen = lengths.max()
        cu_seqlens = lengths.to(hidden_states.device).cumsum(dim=0, dtype=torch.int32)

        for block in self.blocks:
            hidden_states = block(
                hidden_states, cu_seqlens, max_seqlen, rope_freqs_cis=rope_freqs_cis
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class MoonViT3dPretrainedModel(nn.Module):
    model_type = "moonvit3d"
    _no_split_modules = ["PackingTransformer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": PytorchGELUTanh(),
                "attn_bias": True,
            },
            video_attn_type=config.video_attn_type,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def forward(
        self, pixel_values: torch.Tensor, grid_thws: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws (torch.Tensor): Temporal, height and width.
        Returns:
            torch.Tensor: The output tokens.
        """
        assert grid_thws.ndim == 2, f"grid_thws should be 2D, got {grid_thws.ndim}"
        assert grid_thws.size(1) == 3, f"No support for _thw: {grid_thws}"
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        hidden_states = hidden_states.squeeze(0)
        # spatial downsampling 2x with temporal pooling all
        hidden_states = tpool_patch_merger(
            hidden_states, grid_thws, merge_kernel_size=self.merge_kernel_size
        )

        return hidden_states


class K2VLMultiModalProjector(nn.Module):
    """Multi-modal projector with patch merging for K2-VL."""

    def __init__(
        self,
        config: KimiK25VisionConfig,
        use_data_parallel: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.use_data_parallel = use_data_parallel

        # Hidden size after patch merging
        merge_h, merge_w = config.merge_kernel_size
        self.hidden_size = config.vt_hidden_size * merge_h * merge_w

        self.pre_norm = torch.nn.LayerNorm(config.vt_hidden_size, eps=1e-5)
        self.linear_1 = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            prefix=add_prefix(prefix, "linear_1"),
        )
        self.linear_2 = ReplicatedLinear(
            self.hidden_size,
            config.text_hidden_size,
            bias=True,
            prefix=add_prefix(prefix, "linear_2"),
        )
        self.act = nn.GELU()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


@torch.inference_mode()
def mm_projection_auto(
    mm_projector: torch.nn.Module | None, vt_output: list[torch.Tensor]
):
    """Apply MM projector to vision tower outputs."""
    if mm_projector is None:
        return vt_output

    num_embedding_list = [x.shape[0] for x in vt_output]
    batched = torch.cat(vt_output, dim=0)
    proj_out = mm_projector(batched) if mm_projector else batched
    proj_out = proj_out.reshape(-1, proj_out.shape[-1])
    proj_out = torch.split(proj_out, num_embedding_list)
    return proj_out


@torch.inference_mode()
def vision_tower_forward_auto(
    vision_tower: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    mm_projector: torch.nn.Module | None = None,
) -> list[torch.Tensor]:
    """Auto-batched vision tower forward."""
    assert isinstance(
        pixel_values, torch.Tensor
    ), "expect pixel_values to be a tensor, get {}".format(type(pixel_values))
    n = grid_thw.shape[0]
    n_patches_each_media = grid_thw.prod(-1)
    max_infer_batch = max(n_patches_each_media.max(), KIMIV_VT_INFER_MAX_PATCH_NUM)
    logger.debug(
        "vt max_infer_batch: %s, KIMIV_VT_INFER_MAX_PATCH_NUM: %s",
        max_infer_batch,
        KIMIV_VT_INFER_MAX_PATCH_NUM,
    )
    tensors = []
    pre_sum = 0
    current_group_start = 0
    current_group_patches = 0

    for i in range(n):
        current_media_patches = n_patches_each_media[i].item()
        if current_group_patches + current_media_patches <= max_infer_batch:
            current_group_patches += current_media_patches
        else:
            if current_group_start < i:
                group_grid_thw = grid_thw[current_group_start:i]
                group_n_patches = n_patches_each_media[current_group_start:i].sum()
                group_input = pixel_values[pre_sum : pre_sum + group_n_patches]
                group_output = vision_tower(group_input, group_grid_thw)
                proj_out = mm_projection_auto(mm_projector, group_output)
                tensors.extend(proj_out)
                pre_sum += group_n_patches

            current_group_start = i
            current_group_patches = current_media_patches

    # Process the last group
    if current_group_start < n:
        group_grid_thw = grid_thw[current_group_start:n]
        group_n_patches = n_patches_each_media[current_group_start:n].sum()
        group_input = pixel_values[pre_sum : pre_sum + group_n_patches]
        group_output = vision_tower(group_input, group_grid_thw)
        proj_out = mm_projection_auto(mm_projector, group_output)
        tensors.extend(proj_out)

    return tensors


class KimiK25ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: KimiK25Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,  # fix init_tts argument error
    ) -> None:
        super().__init__()
        self.config = config
        # Create vision tower
        self.vision_tower = MoonViT3dPretrainedModel(config.vision_config)
        # Create mm projector
        self.mm_projector = K2VLMultiModalProjector(config.vision_config)

        self.language_model = DeepseekV3ForCausalLM(config.text_config, quant_config)

        # Ensure that the dtype of the vision_tower and mm_projector matches that of the language_model.
        # This solves the dtype mismatch issue when using device_map="auto" and torch_dtype.
        if hasattr(self.language_model, "dtype"):
            target_dtype = self.language_model.dtype
            self.vision_tower = self.vision_tower.to(dtype=target_dtype)
            self.mm_projector = self.mm_projector.to(dtype=target_dtype)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.vision_tower.dtype
        )
        grid_thws = torch.concat([item.grid_thws for item in items], dim=0).to(
            self.vision_tower.device
        )

        target_dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(target_dtype)
        image_features = vision_tower_forward_auto(
            self.vision_tower,
            pixel_values,
            grid_thws,
            mm_projector=self.mm_projector,
        )
        image_features = torch.cat(image_features, dim=0)
        return image_features

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for the model, separating vision and language weights"""
        weights = list(weights)

        # Separate vision tower weights and language model weights
        vision_weights = []
        language_weights = []

        for name, loaded_weight in weights:
            if "vision_tower" in name or "mm_projector" in name:
                name = name.replace(r"wqkv.", r"attn.qkv_proj.")
                name = name.replace(r"wo.", r"attn.proj.")
                name = name.replace("mm_projector.proj.0", "mm_projector.linear_1")
                name = name.replace("mm_projector.proj.2", "mm_projector.linear_2")
                vision_weights.append((name, loaded_weight))
            else:
                name = name.replace("language_model.", "")
                # All other weights go to language model
                language_weights.append((name, loaded_weight))

        # Load vision tower weights
        vision_state_dict = dict(vision_weights)
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in vision_state_dict.items():
            if name not in params_dict:
                raise ValueError(f"Weight {name} not found in params_dict")
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            # loaded_weight = self._pad_vit_attn_dummy_heads(name, loaded_weight)
            weight_loader(param, loaded_weight)

        # Load language model weights
        if language_weights:
            self.language_model.load_weights(language_weights)


EntryClass = [KimiK25ForConditionalGeneration]
