# SPDX-License-Identifier: Apache-2.0
# Copied and adapted from: mossVG/mova/diffusion/models/interactionv2.py
"""
OpenVeo3跨模态交互控制器 v2
实现视觉塔和音频塔之间的条件交叉注意力交互机制

与v1版本的区别:
- v1版本: 使用简单的特征融合方法 (concat/add/cross_attn)
- v2版本: 使用标准的条件交叉注意力机制，类似DiT中的text条件

共同目标: 解决OpenVeo3训练中的维度不匹配问题，支持音频latents(~64维)与视觉特征的交互
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Use LocalAttention for cross-modal attention (different sequence lengths, no SP communication needed)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention

# Use SGLang's optimized RMSNorm instead of importing from mova_video_dit
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# modified from https://github.com/huggingface/transformers/blob/b0db5a02f39ebd2ccffd7f8eb77091fda61f9a1e/src/transformers/models/qwen3/modeling_qwen3.py#L299
class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, base: float, dim: int, device=None):
        super().__init__()
        self.base = base
        self.dim = dim
        self.attention_scaling = 1.0

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(fullgraph=True)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PerFrameAttentionPooling(nn.Module):
    """
    逐帧多头注意力池化：输入展平序列 [B, L, D] 与网格大小 (T, H, W)，
    对每个时间帧在其 H*W token 上进行单查询注意力池化，输出 [B, T, D]。

    参考 SigLIP 的 Multihead Attention Pooling 头（去掉 MLP 与残差叠加）。
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.num_heads = num_heads

        self.probe = nn.Parameter(torch.randn(1, 1, dim))
        nn.init.normal_(self.probe, std=0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]，L = T*H*W
            grid_size: (T, H, W)
        Returns:
            pooled: [B, T, D]
        """
        B, L, D = x.shape
        T, H, W = grid_size
        assert D == self.dim, f"输入通道维度 D={D} 与模块 dim={self.dim} 不一致"
        assert L == T * H * W, f"展平长度 L={L} 与 T*H*W={T*H*W} 不一致"

        S = H * W
        # 重排为按帧分组的 token
        x_bt_s_d = x.view(B, T, S, D).contiguous().view(B * T, S, D)  # [B*T, S, D]

        # 可学习 probe 作为查询，每帧一个查询
        probe = self.probe.expand(B * T, -1, -1)  # [B*T, 1, D]

        # 注意力池化：query=probe，key/value=该帧内的 H*W token
        pooled_bt_1_d = self.attention(probe, x_bt_s_d, x_bt_s_d, need_weights=False)[
            0
        ]  # [B*T, 1, D]
        pooled_bt_d = pooled_bt_1_d.squeeze(1)  # [B*T, D]

        # 还原回 [B, T, D]
        pooled = pooled_bt_d.view(B, T, D)
        pooled = self.layernorm(pooled)
        return pooled


class CrossModalInteractionController:
    """
    控制双塔交互的策略类
    管理视觉DiT(30层)和音频DiT(30层)之间的交互映射
    """

    def __init__(self, visual_layers: int = 30, audio_layers: int = 30):
        self.visual_layers = visual_layers
        self.audio_layers = audio_layers
        self.min_layers = min(visual_layers, audio_layers)

    def get_interaction_layers(
        self, strategy: str = "shallow_focus"
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        获取交互层的映射关系

        Args:
            strategy: 交互策略
                - "shallow_focus": 重点影响浅层，避免深层不对称
                - "distributed": 分布式影响
                - "progressive": 渐进式影响
                - "custom": 自定义影响层

        Returns:
            字典，包含 'v2a'(视觉->音频) 和 'a2v'(音频->视觉) 的映射关系
        """

        if strategy == "shallow_focus":
            # 重点影响前1/3层，避免深层不对称问题
            num_interact = min(10, self.min_layers // 3)
            interact_layers = list(range(0, num_interact))

        elif strategy == "distributed":
            # 在整个网络中分布式影响，每隔几层交互一次
            step = 3
            interact_layers = list(range(0, self.min_layers, step))

        elif strategy == "progressive":
            # 渐进式：浅层密集交互，深层稀疏交互
            shallow = list(range(0, min(8, self.min_layers)))  # 前8层密集
            if self.min_layers > 8:
                deep = list(range(8, self.min_layers, 3))  # 后面每3层一次
                interact_layers = shallow + deep
            else:
                interact_layers = shallow

        elif strategy == "custom":
            # 自定义策略，可以根据实际需求调整
            interact_layers = [0, 2, 4, 6, 8, 12, 16, 20]  # 指定层
            interact_layers = [i for i in interact_layers if i < self.min_layers]

        elif strategy == "full":
            interact_layers = list(range(0, self.min_layers))

        else:
            raise ValueError(f"未知的交互策略: {strategy}")

        # 构建双向映射
        mapping = {
            "v2a": [(i, i) for i in interact_layers],  # 视觉层i -> 音频层i
            "a2v": [(i, i) for i in interact_layers],  # 音频层i -> 视觉层i
        }

        return mapping

    def should_interact(
        self, layer_idx: int, direction: str, interaction_mapping: Dict
    ) -> bool:
        """
        判断指定层是否需要进行交互

        Args:
            layer_idx: 当前层索引
            direction: 交互方向 ('v2a' 或 'a2v')
            interaction_mapping: 交互映射表

        Returns:
            bool: 是否需要交互
        """
        if direction not in interaction_mapping:
            return False

        return any(src == layer_idx for src, _ in interaction_mapping[direction])


class ConditionalCrossAttention(nn.Module):
    """
    Cross-modal attention for dual-tower bridge.

    This module handles attention between video and audio hidden states,
    which have different sequence lengths. Uses LocalAttention because:
    - Cross-modal attention doesn't benefit from SP (different modalities)
    - The condition hidden states are replicated across all SP ranks
    """

    def __init__(self, dim: int, kv_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.q_dim = dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = self.q_dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        # Use LocalAttention for cross-modal attention (no SP communication needed)
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=None,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        if x_freqs is not None:
            x_cos, x_sin = x_freqs
            B, L, _ = q.shape
            q_view = rearrange(q, "b l (h d) -> b l h d", d=self.head_dim)
            x_cos = x_cos.to(q_view.dtype).to(q_view.device)
            x_sin = x_sin.to(q_view.dtype).to(q_view.device)
            # Expect x_cos/x_sin shape: [B or 1, L, head_dim]
            q_view, _ = apply_rotary_pos_emb(
                q_view, q_view, x_cos, x_sin, unsqueeze_dim=2
            )
            q = rearrange(q_view, "b l h d -> b l (h d)")
        if y_freqs is not None:
            y_cos, y_sin = y_freqs
            Bc, Lc, _ = k.shape
            k_view = rearrange(k, "b l (h d) -> b l h d", d=self.head_dim)
            y_cos = y_cos.to(k_view.dtype).to(k_view.device)
            y_sin = y_sin.to(k_view.dtype).to(k_view.device)
            # Expect y_cos/y_sin shape: [B or 1, L, head_dim]
            _, k_view = apply_rotary_pos_emb(
                k_view, k_view, y_cos, y_sin, unsqueeze_dim=2
            )
            k = rearrange(k_view, "b l h d -> b l (h d)")
        q = rearrange(q, "b l (h d) -> b l h d", d=self.head_dim)
        k = rearrange(k, "b l (h d) -> b l h d", d=self.head_dim)
        v = rearrange(v, "b l (h d) -> b l h d", d=self.head_dim)
        x = self.attn(q, k, v)
        x = rearrange(x, "b l h d -> b l (h d)")
        return self.o(x)


# from diffusers.models.attention import AdaLayerNorm
class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 2:
            scale, shift = temb.chunk(2, dim=2)
            # print(f"{x.shape = }, {scale.shape = }, {shift.shape = }")
        elif self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX and OmniGen for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x


class ConditionalCrossAttentionBlock(nn.Module):
    """
    在 ConditionalCrossAttention 外面包一层 Block，对条件输入 y 先做 LayerNorm。
    """

    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        pooled_adaln: bool = False,
    ):
        super().__init__()
        self.y_norm = nn.LayerNorm(kv_dim, eps=eps)
        self.inner = ConditionalCrossAttention(
            dim=dim, kv_dim=kv_dim, num_heads=num_heads, eps=eps
        )
        self.pooled_adaln = pooled_adaln
        if pooled_adaln:
            self.per_frame_pooling = PerFrameAttentionPooling(
                kv_dim, num_heads=num_heads, eps=eps
            )
            self.adaln = AdaLayerNorm(kv_dim, output_dim=dim * 2, chunk_dim=2)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if self.pooled_adaln:
            assert video_grid_size is not None, "video_grid_size 不能为空"
            pooled_y = self.per_frame_pooling(y, video_grid_size)
            # 对 pooled_y 的时间维进行插值以匹配 x 的序列长度
            if pooled_y.shape[1] != x.shape[1]:
                pooled_y = F.interpolate(
                    pooled_y.permute(0, 2, 1),  # [B, C, T]
                    size=x.shape[1],
                    mode="linear",
                    align_corners=False,
                ).permute(
                    0, 2, 1
                )  # [B, T, C]
            x = self.adaln(x, temb=pooled_y)
        y = self.y_norm(y)
        return self.inner(x=x, y=y, x_freqs=x_freqs, y_freqs=y_freqs)


from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class DualTowerConditionalBridge(ModelMixin, ConfigMixin):
    """
    双塔条件桥接模块 v2 (修正版)
    实现正确的架构：
    1. Audio latents [1, 64, 107] → Audio DiT → Audio hidden states [B, L, 1536]
    2. Visual latents [1, 48, 21, 45, 80] → Visual DiT → Visual hidden states [B, L, 3072]
    3. 两个DiT的hidden states之间进行CrossAttention交互
    """

    _repeated_blocks = ("ConditionalCrossAttentionBlock",)

    @register_to_config
    def __init__(
        self,
        visual_layers: int = 30,
        audio_layers: int = 30,
        visual_hidden_dim: int = 3072,  # 视觉DiT的hidden states维度
        audio_hidden_dim: int = 1536,  # 音频DiT的hidden states维度
        audio_fps: float = 44100.0 / 2048.0,
        head_dim: int = 128,  # 注意力头维度
        interaction_strategy: str = "shallow_focus",
        apply_cross_rope: bool = False,  # 是否在跨注意力中应用RoPE
        apply_first_frame_bias_in_rope: bool = False,  # 是否在 RoPE 对齐中考虑首帧 1/video_fps 偏差
        trainable_condition_scale: bool = False,
        pooled_adaln: bool = False,
    ):
        super().__init__()

        self.visual_hidden_dim = visual_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.audio_fps = audio_fps
        self.head_dim = head_dim
        self.apply_cross_rope = apply_cross_rope
        self.apply_first_frame_bias_in_rope = apply_first_frame_bias_in_rope
        self.trainable_condition_scale = trainable_condition_scale
        self.pooled_adaln = pooled_adaln
        if self.trainable_condition_scale:
            self.condition_scale = nn.Parameter(
                torch.tensor([1.0], dtype=torch.float32)
            )
        else:
            self.condition_scale = 1.0

        self.controller = CrossModalInteractionController(visual_layers, audio_layers)
        self.interaction_mapping = self.controller.get_interaction_layers(
            interaction_strategy
        )

        # 条件交叉注意力模块 - 在DiT hidden states层面交互
        self.audio_to_video_conditioners = (
            nn.ModuleDict()
        )  # 音频hidden states → 视觉DiT条件
        self.video_to_audio_conditioners = (
            nn.ModuleDict()
        )  # 视觉hidden states → 音频DiT条件

        # 为需要交互的层创建条件控制器
        # 音频DiT hidden states 条件调控视频DiT
        self.rotary = RotaryEmbedding(base=10000.0, dim=head_dim)
        for v_layer, _ in self.interaction_mapping["a2v"]:
            self.audio_to_video_conditioners[str(v_layer)] = (
                ConditionalCrossAttentionBlock(
                    dim=visual_hidden_dim,  # 3072 (视觉DiT hidden states)
                    kv_dim=audio_hidden_dim,  # 1536 (音频DiT hidden states)
                    num_heads=visual_hidden_dim // head_dim,  # 根据维度动态计算头数
                    pooled_adaln=False,  # a2v 应该无需 pooled adaln
                )
            )

        # 视觉DiT hidden states 条件控制音频DiT
        for a_layer, _ in self.interaction_mapping["v2a"]:
            self.video_to_audio_conditioners[str(a_layer)] = (
                ConditionalCrossAttentionBlock(
                    dim=audio_hidden_dim,  # 1536 (音频DiT hidden states)
                    kv_dim=visual_hidden_dim,  # 3072 (视觉DiT hidden states)
                    num_heads=audio_hidden_dim // head_dim,  # 安全的头数计算
                    pooled_adaln=self.pooled_adaln,
                )
            )

    @torch.no_grad()
    def build_aligned_freqs(
        self,
        video_fps: float,
        grid_size: Tuple[int, int, int],
        audio_steps: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        基于视频 fps 与网格大小 (f_v, h, w) 以及音频序列长度 audio_steps（对应固定音频 fps=44100/2048）
        生成对齐的 RoPE (cos, sin)。

        Returns:
            visual_freqs: (cos_v, sin_v) 形状 [1, f_v*h*w, head_dim]
            audio_freqs:  (cos_a, sin_a) 形状 [1, audio_steps, head_dim]
        """
        f_v, h, w = grid_size
        L_v = f_v * h * w
        L_a = int(audio_steps)

        device = device or next(self.parameters()).device
        dtype = dtype or torch.float32

        # 音频位置: 0,1,2,...,f-1  （以音频为基准）
        audio_pos = torch.arange(L_a, device=device, dtype=torch.float32).unsqueeze(0)

        # 视频位置: 将视频帧对齐到音频步长单位
        # FIXME(dhyu): 硬编码 VAE 时间倍数：4
        if self.apply_first_frame_bias_in_rope:
            # 考虑“首帧为 1/video_fps”的偏差
            video_effective_fps = float(video_fps) / 4.0
            if f_v > 0:
                t_starts = torch.zeros((f_v,), device=device, dtype=torch.float32)
                if f_v > 1:
                    t_starts[1:] = (1.0 / float(video_fps)) + torch.arange(
                        f_v - 1, device=device, dtype=torch.float32
                    ) * (1.0 / video_effective_fps)
            else:
                t_starts = torch.zeros((0,), device=device, dtype=torch.float32)
            # 转换到音频步长单位
            video_pos_per_frame = t_starts * float(self.audio_fps)
        else:
            # 不考虑首帧偏差：等间隔对齐
            scale = float(self.audio_fps) / float(video_fps / 4.0)
            video_pos_per_frame = (
                torch.arange(f_v, device=device, dtype=torch.float32) * scale
            )
        # 展平到 f*h*w，帧内的 h*w token 共用同一时间位置
        video_pos = video_pos_per_frame.repeat_interleave(h * w).unsqueeze(0)

        # print(f"video fps: {video_fps}, audio fps: {self.audio_fps}, scale: {scale}")
        # print(f"video pos: {video_pos.shape}, audio pos: {audio_pos.shape}")

        # 构造占位 x 以产生 cos/sin，dim=head_dim
        dummy_v = torch.zeros((1, L_v, self.head_dim), device=device, dtype=dtype)
        dummy_a = torch.zeros((1, L_a, self.head_dim), device=device, dtype=dtype)

        cos_v, sin_v = self.rotary(dummy_v, position_ids=video_pos)
        cos_a, sin_a = self.rotary(dummy_a, position_ids=audio_pos)

        return (cos_v, sin_v), (cos_a, sin_a)

    def should_interact(self, layer_idx: int, direction: str) -> bool:
        return self.controller.should_interact(
            layer_idx, direction, self.interaction_mapping
        )

    def apply_conditional_control(
        self,
        layer_idx: int,
        direction: str,
        primary_hidden_states: torch.Tensor,
        condition_hidden_states: torch.Tensor,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        condition_scale: Optional[float] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """
        应用条件控制（在DiT hidden states层面）

        Args:
            layer_idx: 当前层索引
            direction: 条件方向 ('a2v': 音频hidden states→视频DiT, 'v2a': 视觉hidden states→音频DiT)
            primary_hidden_states: 主DiT的hidden states [B, L, hidden_dim]
            condition_hidden_states: 条件DiT的hidden states [B, L, hidden_dim]
            condition_scale: 条件强度控制（类似CFG scale）

        Returns:
            条件调控后的主DiT hidden states [B, L, hidden_dim]
        """

        if not self.controller.should_interact(
            layer_idx, direction, self.interaction_mapping
        ):
            return primary_hidden_states

        if direction == "a2v":
            # 音频DiT hidden states 条件调控视频DiT
            conditioner = self.audio_to_video_conditioners[str(layer_idx)]

        elif direction == "v2a":
            # 视觉DiT hidden states 条件控制音频DiT
            conditioner = self.video_to_audio_conditioners[str(layer_idx)]
        else:
            raise ValueError(f"Invalid direction: {direction}")

        conditioned_features = conditioner(
            x=primary_hidden_states,
            y=condition_hidden_states,
            x_freqs=x_freqs,
            y_freqs=y_freqs,
            video_grid_size=video_grid_size,
        )

        if self.trainable_condition_scale and condition_scale is not None:
            logger.warning(
                "当前模型存在可训练的 condition_scale，但 condition_scale 参数被外部传递，将忽略可训练的 condition_scale，使用外部传递的 condition_scale=%s",
                condition_scale,
            )

        scale = condition_scale if condition_scale is not None else self.condition_scale

        primary_hidden_states = primary_hidden_states + conditioned_features * scale

        return primary_hidden_states

    def forward(
        self,
        layer_idx: int,
        visual_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        *,
        x_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        a2v_condition_scale: Optional[float] = None,
        v2a_condition_scale: Optional[float] = None,
        condition_scale: Optional[float] = None,
        video_grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        同时对视觉/音频塔进行双向条件调控。

        Args:
            layer_idx: 当前层索引。
            visual_hidden_states: 视觉DiT的hidden states。
            audio_hidden_states: 音频DiT的hidden states。
            x_freqs / y_freqs: 跨模态RoPE频率对 (cos, sin)。
                若传入，则假定 x_freqs 对应 primary（当前塔），y_freqs 对应 condition。
            a2v_condition_scale: 音频→视觉的条件强度，覆盖全局condition_scale。
            v2a_condition_scale: 视觉→音频的条件强度，覆盖全局condition_scale。
            condition_scale: 兜底的条件强度；当单向scale为None时使用。
            video_grid_size: (F, H, W)，在启用 pooled_adaln 时用于音频侧。

        Returns:
            (视觉hidden states, 音频hidden states)，均已应用对应方向的条件控制。
        """

        visual_conditioned = self.apply_conditional_control(
            layer_idx=layer_idx,
            direction="a2v",
            primary_hidden_states=visual_hidden_states,
            condition_hidden_states=audio_hidden_states,
            x_freqs=x_freqs,
            y_freqs=y_freqs,
            condition_scale=(
                a2v_condition_scale
                if a2v_condition_scale is not None
                else condition_scale
            ),
            video_grid_size=video_grid_size,
        )

        audio_conditioned = self.apply_conditional_control(
            layer_idx=layer_idx,
            direction="v2a",
            primary_hidden_states=audio_hidden_states,
            condition_hidden_states=visual_hidden_states,
            x_freqs=y_freqs,
            y_freqs=x_freqs,
            condition_scale=(
                v2a_condition_scale
                if v2a_condition_scale is not None
                else condition_scale
            ),
            video_grid_size=video_grid_size,
        )

        return visual_conditioned, audio_conditioned

    def get_interaction_info(self) -> Dict:
        """获取交互信息，用于调试和可视化"""
        return {
            "visual_layers": self.controller.visual_layers,
            "audio_layers": self.controller.audio_layers,
            "visual_hidden_dim": self.visual_hidden_dim,  # 3072
            "audio_hidden_dim": self.audio_hidden_dim,  # 1536
            "interaction_mapping": self.interaction_mapping,
            "total_interactions": len(self.interaction_mapping["v2a"])
            + len(self.interaction_mapping["a2v"]),
            "audio_to_video_conditioners_count": len(self.audio_to_video_conditioners),
            "video_to_audio_conditioners_count": len(self.video_to_audio_conditioners),
            "total_trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "architecture_note": "v2修正版: 在DiT hidden states层面进行交互，而非latents层面",
        }


class ContrastiveVideoAudioSimilarity(nn.Module):
    """
    对比学习相似度头：
    - 输入：
        video_x: [B, L1, Dv], 其中 L1 = T * H * W
        video_grid_size: (T, H, W)
        audio_x: [B, L2, Da]
    - 流程：
        1) video_x -> PerFrameAttentionPooling -> [B, T, Dv]
        2) 线性映射到音频维度 Da -> [B, T, Da]
        3) video 与 audio 分别通过无仿射 RMSNorm
        4) 计算相似度矩阵：sim = v @ a^T -> [B, T, L2]
    - 输出：
        sim: [B, T, L2]
    """

    DEFAULT_AUDIO_FPS = 44100.0 / 2048.0

    def __init__(
        self, video_dim: int, audio_dim: int, num_heads: int, eps: float = 1e-6
    ):
        super().__init__()
        assert video_dim > 0 and audio_dim > 0 and num_heads > 0
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.pool = PerFrameAttentionPooling(
            dim=video_dim, num_heads=num_heads, eps=eps
        )
        self.video_proj = nn.Linear(video_dim, audio_dim, bias=True)
        # 可学习的 logit 缩放与偏置（CLIP 风格）：sim * exp(logit_scale) + logit_bias
        self.logit_scale = nn.Parameter(torch.tensor([3.0]))
        self.logit_bias = nn.Parameter(torch.tensor([-3.9]))

    def forward(
        self,
        video_x: torch.Tensor,
        video_grid_size: Tuple[int, int, int],
        audio_x: torch.Tensor,
    ) -> torch.Tensor:
        Bv, L1, Dv = video_x.shape
        Ba, L2, Da = audio_x.shape
        assert Bv == Ba, f"batch 不一致: video B={Bv}, audio B={Ba}"
        assert Dv == self.video_dim, f"video dim 不匹配: {Dv} vs {self.video_dim}"
        assert Da == self.audio_dim, f"audio dim 不匹配: {Da} vs {self.audio_dim}"

        # 1) 每帧池化 -> [B, T, Dv]
        v_t = self.pool(video_x, video_grid_size)

        # 2) 映射到音频维度 -> [B, T, Da]
        v_t = self.video_proj(v_t)

        # 3) 归一化
        v_t = v_t / v_t.norm(dim=-1, keepdim=True)
        a = audio_x / audio_x.norm(dim=-1, keepdim=True)

        # 4) 相似度 [B, T, L2]
        sim = torch.matmul(v_t, a.transpose(1, 2))

        # 5) 可学习缩放与偏置
        logit_scale = self.logit_scale.to(device=sim.device, dtype=sim.dtype)
        logit_bias = self.logit_bias.to(device=sim.device, dtype=sim.dtype)
        sim = sim * logit_scale.exp() + logit_bias
        # print(f"{logit_scale=}, {logit_bias=}, {sim.shape=}")
        return sim

    @staticmethod
    @torch.no_grad()
    def build_contrastive_overlap_mask(
        video_fps: float,
        num_video_frames: int,
        num_audio_frames: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        use_vae_time_stride: bool = True,
        audio_fps: float = DEFAULT_AUDIO_FPS,
    ) -> torch.Tensor:
        """
        构建二维 0/1 掩码，表示视频与音频帧在时间轴上的重叠关系（首帧长度=1/video_fps，其余按 VAE 下采样）。

        Args:
            video_fps: 原始视频帧率 (Hz)。
            num_video_frames: 视频帧数量（通常为VAE后时间下采样的帧数）。
            num_audio_frames: 音频帧数量（基于 hop=2048, sr=44100）。
            device: 返回张量所在设备；默认使用 CPU。
            dtype: 返回张量数据类型；默认 float32。
            use_vae_time_stride: 若 True，则其余帧使用 video_fps/4 有效分辨率。
            audio_fps: 音频帧率；默认 44100/2048。

        Returns:
            torch.Tensor: [num_video_frames, num_audio_frames] 的 0/1 掩码。
        """
        if num_video_frames <= 0 or num_audio_frames <= 0:
            return torch.zeros(
                (max(num_video_frames, 0), max(num_audio_frames, 0)),
                dtype=dtype or torch.float32,
            )

        device = device or torch.device("cpu")
        dtype = dtype or torch.float32

        video_effective_fps = video_fps / 4.0 if use_vae_time_stride else video_fps

        # 视频帧起止时间（秒），首帧特殊处理
        if num_video_frames > 0:
            v_start = torch.zeros(
                (num_video_frames,), device=device, dtype=torch.float32
            )
            if num_video_frames > 1:
                v_start[1:] = (1.0 / video_fps) + torch.arange(
                    num_video_frames - 1, device=device, dtype=torch.float32
                ) * (1.0 / video_effective_fps)
        else:
            v_start = torch.zeros((0,), device=device, dtype=torch.float32)
        v_duration = torch.full(
            (num_video_frames,),
            1.0 / video_effective_fps if video_effective_fps > 0 else 0.0,
            device=device,
            dtype=torch.float32,
        )
        if num_video_frames > 0:
            v_duration[0] = 1.0 / video_fps
        v_end = v_start + v_duration

        # 音频帧起止时间（秒）
        a_idx = torch.arange(num_audio_frames, device=device, dtype=torch.float32)
        a_start = a_idx / audio_fps
        a_end = a_start + (1.0 / audio_fps)

        overlap = (v_start[:, None] < a_end[None, :]) & (
            a_start[None, :] < v_end[:, None]
        )
        return overlap.to(dtype=dtype, device=device)

    def training_loss(
        self,
        sim: torch.Tensor,
        target_mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        训练时使用的对比损失（sigmoid + BCE）。

        Args:
            sim: 相似度 logits，形状 [B, T, L2]（已应用可学习缩放与偏置）
            target_mask: 0/1 目标掩码，形状 [B, T, L2]
            valid_mask: 可选的有效元素掩码（0/1），同形状，用于忽略无效位置
            reduction: 'mean' | 'sum' | 'none'

        Returns:
            标量 loss（当 reduction='none' 时返回逐元素 loss 张量）
        """
        assert (
            sim.shape == target_mask.shape
        ), f"shape 不一致：sim={sim.shape}, target={target_mask.shape}"
        if valid_mask is not None:
            assert (
                valid_mask.shape == sim.shape
            ), f"valid_mask 形状不匹配：{valid_mask.shape} vs {sim.shape}"

        logits = sim
        target = target_mask.to(dtype=logits.dtype)
        video_frames = logits.shape[1]

        # 使用梯度等价的 KL(P||Q)：KL = BCE_with_logits(logits, target) - H(P)
        bce_el = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        with torch.no_grad():
            entropy_el = F.binary_cross_entropy(target, target, reduction="none")
        loss_el = bce_el - entropy_el
        if valid_mask is not None:
            loss_el = loss_el * valid_mask.to(dtype=loss_el.dtype)

        if reduction == "none":
            return loss_el
        elif reduction == "sum":
            return loss_el.sum()
        elif reduction == "mean":
            if valid_mask is not None:
                denom = valid_mask.sum().clamp_min(1).to(dtype=loss_el.dtype)
            else:
                denom = torch.tensor(
                    video_frames, device=loss_el.device, dtype=loss_el.dtype
                )
                return loss_el.sum() / denom
        else:
            raise ValueError(f"无效的 reduction: {reduction}")


EntryClass = DualTowerConditionalBridge
