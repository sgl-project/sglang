"""
LFM3-VL vision tower (the LFM2.5-230M-class ViT) and vision->language projector.

Not a SigLIP2 NaFlex tower like LFM2-VL's: this encoder is a Liquid hybrid built
from the LFM3 operator alphabet -- 8 gated depthwise 3x3 conv2d layers over the
(h, w) patch grid interleaved with 6 bidirectional GQA layers (16 q / 8 kv, per-head
q/k RMSNorm, 2D RoPE). Patch embedding is a plain `Linear` over Qwen2-VL-preprocessed
patch vectors, and there are no learned position embeddings.

Three facts that are load-bearing for correctness:

  1. The conv layers need a true row-major (raster) h x w grid, but Qwen's
     patchifier emits *block-major* order (2x2 merge blocks contiguous), so
     `_block_major_to_raster` undoes that permutation up front.
  2. There is no cross-frame attention -- every temporal patch is an independent
     `h*w`-token sequence. The HF reference loops per (grid entry, temporal patch);
     we batch frames sharing an (h, w) instead, collapsing a video's `grid_t`
     sequential tower passes into one.
  3. RMSNorm uses `cast_x_before_out_mul=True` to match the reference's
     rounding point: cast to the input dtype before multiplying by the weight.
     The native path preserves the eager reduction used by the reference.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.lfm3_vl import Lfm3VlConfig
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, set_weight_attrs


def _apply_rope_2d(
    hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply half-split 2D RoPE to (batch, tokens, heads, head_dim)."""
    head_dim = hidden_states.shape[-1]
    first, second = (
        hidden_states[..., : head_dim // 2],
        hidden_states[..., head_dim // 2 :],
    )
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class Lfm3VlVisionMLP(nn.Module):
    """SwiGLU FFN with fused gate_up, matching the checkpoint's `[gate; up]` layout."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up", prefix),
        )
        self.w2 = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("w2", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up(hidden_states)
        out, _ = self.w2(self.act_fn(gate_up))
        return out


class Lfm3VlVisionAttention(nn.Module):
    """Full bidirectional GQA over one frame, with per-head q/k RMSNorm + 2D RoPE.

    No causal or padding mask exists: each frame is a complete `h*w` sequence and
    frames never attend to each other, so batched SDPA needs no mask at all.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        total_num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            total_num_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )
        self.q_layernorm = RMSNorm(
            self.head_dim,
            eps=config.norm_eps,
            cast_x_before_out_mul=True,
            force_native=True,
        )
        self.k_layernorm = RMSNorm(
            self.head_dim,
            eps=config.norm_eps,
            cast_x_before_out_mul=True,
            force_native=True,
        )

        # Read the post-sharding head counts off the layer rather than
        # re-deriving them: QKVParallelLinear owns the KV-replication policy for
        # tp_size > num_kv_heads, and a second copy of that arithmetic here would
        # mis-slice the qkv split silently rather than raise.
        self.num_local_heads = self.qkv_proj.num_heads
        self.num_local_kv_heads = self.qkv_proj.num_kv_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_local_heads * self.head_dim
        kv_size = self.num_local_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q = self.q_layernorm(q.view(batch, tokens, self.num_local_heads, self.head_dim))
        k = self.k_layernorm(
            k.view(batch, tokens, self.num_local_kv_heads, self.head_dim)
        )
        v = v.view(batch, tokens, self.num_local_kv_heads, self.head_dim)

        cos, sin = rope
        q = _apply_rope_2d(q, cos, sin)
        k = _apply_rope_2d(k, cos, sin)

        # (batch, tokens, heads, dim) -> (batch, heads, tokens, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

        attn_out = attn_out.transpose(1, 2).reshape(batch, tokens, q_size)
        out, _ = self.out_proj(attn_out)
        return out


class Lfm3VlVisionConv2d(nn.Module):
    """Gated depthwise 3x3 conv2d over the patch grid: out_proj(C * conv2d(B * x)).

    Sharded channel-wise, the 2D analogue of how `Lfm3ShortConv` splits its causal
    conv: `in_proj` is a 3-way merged column-parallel projection so each rank owns
    the same channel slice of all three gates (`[B | C | x]`), `spatial_conv` is
    sharded on its channel dim, and `out_proj` reduces back to the full hidden
    size. Because the depthwise conv never mixes channels, the split is exact --
    each rank convolves its own channels and the row-parallel reduce reassembles
    the residual stream.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_parallel().tp_size
        self.hidden_size_per_partition = hidden_size // tp_size

        self.in_proj = MergedColumnParallelLinear(
            hidden_size,
            [hidden_size] * 3,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("in_proj", prefix),
        )
        self.out_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )
        self.spatial_conv = nn.Parameter(
            torch.empty(self.hidden_size_per_partition, 1, 3, 3)
        )
        set_weight_attrs(self.spatial_conv, {"weight_loader": sharded_weight_loader(0)})

    def forward(
        self, hidden_states: torch.Tensor, *, height: int, width: int
    ) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        projected, _ = self.in_proj(hidden_states)
        # Each rank's slice is [B_local | C_local | x_local], so a plain 3-way
        # chunk of the local output keeps the gates aligned with the conv channels.
        first, middle, third = projected.chunk(3, dim=-1)
        # (batch, tokens, channels) -> (batch, channels, height, width)
        gated = (first * third).permute(0, 2, 1).reshape(batch, -1, height, width)
        conv_out = F.conv2d(
            gated, self.spatial_conv, padding=1, groups=self.hidden_size_per_partition
        )
        conv_out = conv_out.reshape(batch, -1, tokens).permute(0, 2, 1)
        out, _ = self.out_proj(middle * conv_out)
        return out


class Lfm3VlVisionLayer(nn.Module):
    """Pre-norm hybrid layer. Note: no post-norms here, unlike the LFM3 LM layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        layer_type: str,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.operator_norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            cast_x_before_out_mul=True,
            force_native=True,
        )
        self.ffn_norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            cast_x_before_out_mul=True,
            force_native=True,
        )
        if layer_type == "conv":
            self.conv = Lfm3VlVisionConv2d(
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("conv", prefix),
            )
        elif layer_type == "full_attention":
            self.self_attn = Lfm3VlVisionAttention(
                config,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            raise ValueError(f"unsupported vision layer type: {layer_type!r}")
        self.feed_forward = Lfm3VlVisionMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        height: int,
        width: int,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        operator_input = self.operator_norm(hidden_states)
        if self.layer_type == "conv":
            operator_output = self.conv(operator_input, height=height, width=width)
        else:
            operator_output = self.self_attn(operator_input, rope=rope)
        hidden_states = hidden_states + operator_output
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class Lfm3VlVisionEncoder(nn.Module):
    """The 14-layer hybrid tower, batched over frames that share a patch grid."""

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_parameters["rope_theta"]
        self.patchifier = nn.Linear(
            (config.in_channels * config.temporal_patch_size * config.patch_size**2),
            config.hidden_size,
            bias=False,
        )
        self.embedding_norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            cast_x_before_out_mul=True,
            force_native=True,
        )
        self.layers = nn.ModuleList(
            [
                Lfm3VlVisionLayer(
                    config,
                    layer_type=layer_type,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{idx}", prefix),
                )
                for idx, layer_type in enumerate(config.layer_types)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patchifier.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patchifier.weight.device

    @staticmethod
    def _block_major_to_raster(
        hidden_states: torch.Tensor,
        *,
        temporal: int,
        height: int,
        width: int,
        merge_size: int,
    ) -> torch.Tensor:
        """Undo Qwen's block-major patch order so the conv layers see a real grid."""
        return (
            hidden_states.view(
                temporal,
                height // merge_size,
                width // merge_size,
                merge_size,
                merge_size,
                -1,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(temporal, height * width, -1)
        )

    def _rope_tables(
        self, *, height: int, width: int, like: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (cos, sin) for interleaved 2D RoPE over a raster h x w grid.

        Row frequencies occupy the first quarter of the head, column frequencies
        the second; the rotation pairs dim j with j + head_dim/2. Positions restart
        at (0, 0) for every frame -- there is no temporal RoPE component.

        Shaped (1, h*w, 1, head_dim/2) to broadcast over frame-batch and heads.
        """
        device, dtype = like.device, like.dtype
        quarter = self.head_dim // 4
        frequencies = self.rope_theta ** (
            -(
                2.0
                * torch.arange(quarter, device=device, dtype=torch.float32)
                / (self.head_dim // 2)
            )
        )
        rows = torch.arange(height, device=device).repeat_interleave(width)
        cols = torch.arange(width, device=device).repeat(height)
        angles = torch.cat(
            (rows[:, None] * frequencies, cols[:, None] * frequencies), dim=-1
        )
        return (
            angles.cos().to(dtype)[None, :, None, :],
            angles.sin().to(dtype)[None, :, None, :],
        )

    def _split_frames(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[list[torch.Tensor], dict[tuple[int, int], list[int]]]:
        """Flatten (grid entry, temporal patch) into per-frame raster sequences.

        Returns the frames in reference order (grid entries in order, temporal
        patches within each) plus, for every distinct patch grid, the frame
        indices carrying it -- so the caller can batch a shape without the frames
        having to remember their own (h, w).
        """
        frames: list[torch.Tensor] = []
        groups: dict[tuple[int, int], list[int]] = {}
        offset = 0
        for temporal, height, width in grid_thw.tolist():
            temporal, height, width = int(temporal), int(height), int(width)
            tokens = temporal * height * width
            raster = self._block_major_to_raster(
                pixel_values[offset : offset + tokens],
                temporal=temporal,
                height=height,
                width=width,
                merge_size=self.spatial_merge_size,
            )
            offset += tokens
            for index in range(temporal):
                groups.setdefault((height, width), []).append(len(frames))
                frames.append(raster[index])
        if offset != pixel_values.shape[0]:
            raise ValueError(
                f"pixel_values has {pixel_values.shape[0]} patches but grid_thw "
                f"accounts for {offset}"
            )
        return frames, groups

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> list[tuple[torch.Tensor, int, int]]:
        """Encode Qwen-preprocessed patches into per-frame token sequences.

        Returns one `(tokens, height, width)` triple per frame, in reference
        order. Frames sharing an (h, w) are encoded as a single batch.
        """
        frames, groups = self._split_frames(pixel_values, grid_thw)

        encoded: dict[int, tuple[torch.Tensor, int, int]] = {}
        for (height, width), indices in groups.items():
            hidden_states = self.embedding_norm(
                self.patchifier(torch.stack([frames[index] for index in indices]))
            )
            # Positions depend only on (height, width), so every attention layer
            # in this group shares one pair of tables.
            rope = self._rope_tables(height=height, width=width, like=hidden_states)
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states, height=height, width=width, rope=rope
                )
            for position, index in enumerate(indices):
                encoded[index] = (hidden_states[position], height, width)
        return [encoded[index] for index in range(len(frames))]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load `model.vision_tower.*` weights (prefix already stripped by the caller).

        The checkpoint ships unfused q/k/v and w1 (gate) / w3 (up); the shard
        loaders fuse them into `qkv_proj` / `gate_up`. The raw `spatial_conv`
        Parameter is stored as `spatial_conv.weight`.
        """
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # Dotted so they only match the `feed_forward.w1` / `.w3` path components.
            (".gate_up.", ".w1.", 0),
            (".gate_up.", ".w3.", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.endswith(".conv.spatial_conv.weight"):
                name = name.removesuffix(".weight")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                if name not in params_dict:
                    raise KeyError(f"unexpected vision_encoder weight: {name}")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        return loaded_params


class Lfm3VlProjector(nn.Module):
    """2x2 spatial merge (concat, not pooling) then a 2-layer GELU MLP.

    LayerNorm runs per token on `vision_hidden_size` *before* the merge, which is
    the opposite order from LFM2-VL's projector; both linears carry bias and the
    activation is exact erf GELU.

    The two linears are sharded on the projector's inner dimension (column then
    row), the standard MLP split: the LayerNorm and the 2x2 merge both run on the
    full `vision_hidden_size` residual, and `linear_2` reduces back to the text
    hidden size.
    """

    def __init__(
        self,
        config: Lfm3VlConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.spatial_merge_size = config.downsample_factor
        self.layer_norm = nn.LayerNorm(
            config.vision_config.hidden_size,
            eps=config.projector_layer_norm_eps,
        )
        self.linear_1 = ColumnParallelLinear(
            self.spatial_merge_size**2 * config.vision_config.hidden_size,
            config.projector_hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
            prefix=add_prefix("linear_1", prefix),
        )
        self.linear_2 = RowParallelLinear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("linear_2", prefix),
        )

    def forward(
        self, hidden_states: torch.Tensor, *, height: int, width: int
    ) -> torch.Tensor:
        merge = self.spatial_merge_size
        if height % merge or width % merge:
            raise ValueError(
                f"projector requires a patch grid divisible by {merge}, got "
                f"{height}x{width}"
            )
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.view(
            height // merge, merge, width // merge, merge, -1
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            -1, merge**2 * hidden_states.shape[-1]
        )
        hidden_states, _ = self.linear_1(hidden_states)
        out, _ = self.linear_2(F.gelu(hidden_states, approximate="none"))
        return out
