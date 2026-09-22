"""Kimi K3 vision tower (MoonViT3d) and projector.

Faithful port of the checkpoint reference implementation
(modeling_kimi_k3.py). Dedicated to K3 — do not share with Kimi K2.5:
K3 uses qkv_hidden_size != hidden_size (head_dim = qkv_hidden_size //
num_heads), RMSNorm encoder norms, bias-free linears, and the
PatchMergerMLPV2 projector (no pre-norm, post RMSNorm), all of which
differ from the K2.5 vision code.

Weight names match the checkpoint exactly (wqkv/wo, mlp.fc0/fc1,
mm_projector.proj.0/proj.2, mm_projector.post_norm), so loading needs no
renames.
"""

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.attention.vision_rope import (
    apply_fused_qk_complex_rope,
    precompile_fused_qk_complex_rope,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import (
    FLASHINFER_WORKSPACE_SIZE_BYTES,
    QKV_BACKEND_IMPL,
    VisionAttentionMetadata,
    prepare_flashinfer_cudnn_vision_attention_metadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.models.kimi_vl_moonvit import tpool_patch_merger
from sglang.srt.multimodal.mm_utils import concat_or_single
from sglang.srt.runtime_context import get_mm
from sglang.srt.utils import get_bool_env_var, is_hip, print_info_once

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.ops.triton.conv.conv2d import conv2d as aiter_conv2d

_SM103_TRITON_MAX_SEQLEN = 1536
_SM103_FA4_MIN_ATTENTION_WORK = 3_000_000

GridTHW = Tuple[int, int, int]
SegmentBounds = Tuple[Tuple[int, int], ...]


def _resolve_grid_thw_list(
    grid_thws: torch.Tensor, grid_thw_list: Optional[Sequence[Sequence[int]]] = None
) -> Tuple[GridTHW, ...]:
    values = grid_thws.tolist() if grid_thw_list is None else grid_thw_list
    return tuple((int(t), int(h), int(w)) for t, h, w in values)


def _get_mm_attention_backend() -> str:
    try:
        return get_mm().mm_attention_backend or "auto"
    except ValueError:
        # config not published yet (import-time probes)
        return "auto"


def _is_fa4_available() -> bool:
    try:
        from sglang.kernels.ops.attention.flash_attention_v4 import (
            is_flash_attention_v4_available,
        )
    except ImportError:
        return False
    return is_flash_attention_v4_available()


def _resolve_mm_attention_backend(
    configured_backend: str,
    *,
    max_seqlen: int,
    total_tokens: int,
    device: torch.device,
    fa4_available: Optional[bool] = None,
) -> str:
    if configured_backend != "auto":
        return configured_backend
    if device.type != "cuda":
        return "sdpa"
    if torch.cuda.get_device_capability(device) != (10, 3):
        return "sdpa"

    use_fa4 = (
        max_seqlen > _SM103_TRITON_MAX_SEQLEN
        or max_seqlen * total_tokens >= _SM103_FA4_MIN_ATTENTION_WORK
    )
    if use_fa4:
        if fa4_available is None:
            fa4_available = _is_fa4_available()
        return "fa4" if fa4_available else "sdpa"
    return "triton_attn"


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _can_use_fused_rope(hidden_states: torch.Tensor, freqs_cis: torch.Tensor) -> bool:
    return _can_use_fused_rope_for_shape(
        dtype=hidden_states.dtype,
        device=hidden_states.device,
        freqs_cis=freqs_cis,
    )


def _can_use_fused_rope_for_shape(
    *,
    dtype: torch.dtype,
    device: torch.device,
    freqs_cis: torch.Tensor,
) -> bool:
    if not (
        device.type == "cuda"
        and freqs_cis.is_cuda
        and device == freqs_cis.device
        and dtype in (torch.bfloat16, torch.float16)
        and freqs_cis.dtype == torch.complex64
    ):
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 9


def sdpa_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    segment_bounds: Optional[SegmentBounds] = None,
) -> torch.Tensor:
    is_single_segment = (
        len(segment_bounds) == 1
        if segment_bounds is not None
        else cu_seqlens.numel() == 2
    )
    if is_single_segment:
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
        )
        return out.squeeze(0).transpose(0, 1).flatten(start_dim=-2)
    outputs = []
    if segment_bounds is None:
        bounds = cu_seqlens.tolist()
        segment_bounds = tuple(zip(bounds[:-1], bounds[1:]))
    for start, end in segment_bounds:
        seg_q = q[start:end].transpose(0, 1).unsqueeze(0)
        seg_k = k[start:end].transpose(0, 1).unsqueeze(0)
        seg_v = v[start:end].transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(seg_q, seg_k, seg_v)
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs).flatten(start_dim=-2)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def get_1d_sincos_pos_embed(embed_dim: int, t_size: int) -> np.ndarray:
    grid_t = np.arange(t_size, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)


def interpolate_pos_emb(
    weight: torch.Tensor, interpolation_mode: str, shape: Tuple[int, int]
) -> torch.Tensor:
    return (
        F.interpolate(
            weight.permute((2, 0, 1)).contiguous().unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


class Learnable2DInterpPosEmbDividedFixed(nn.Module):
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
            torch.from_numpy(get_1d_sincos_pos_embed(dim, num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )

    def position_embeddings(
        self,
        grid_thws: torch.Tensor,
        *,
        grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
    ) -> torch.Tensor:
        pos_embs = []
        for t, h, w in _resolve_grid_thw_list(grid_thws, grid_thw_list):
            assert t <= self.num_frames, f"t:{t} > num_frames:{self.num_frames}"
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = interpolate_pos_emb(
                    self.weight, self.interpolation_mode, (h, w)
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[
                    0:t
                ].to(pos_emb_2d.dtype)

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        return torch.cat(pos_embs)

    def forward(
        self,
        x: torch.Tensor,
        grid_thws: torch.Tensor,
        *,
        grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if position_embeddings is None:
            position_embeddings = self.position_embeddings(
                grid_thws, grid_thw_list=grid_thw_list
            )
        return x + position_embeddings


class MoonVision3dPatchEmbed(nn.Module):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: Union[int, Tuple[int, int]] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        pos_emb_interpolation_mode: str = "bicubic",
        patch_embed_proj_bias: bool = True,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=patch_embed_proj_bias,
        )

        if pos_emb_type != "divided_fixed":
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")
        self.pos_emb = Learnable2DInterpPosEmbDividedFixed(
            height=pos_emb_height,
            width=pos_emb_width,
            num_frames=pos_emb_time,
            dim=out_dim,
            interpolation_mode=pos_emb_interpolation_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_thws: torch.Tensor,
        *,
        grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # MIOpen can overflow grid_size for some patch shapes. Prefer AITER's
        # Triton convolution on AMD, with an equivalent linear fallback.
        if _use_aiter:
            x = aiter_conv2d(
                x,
                self.proj.weight,
                self.proj.bias,
                stride=self.patch_size,
                padding=(0, 0),
                dilation=(1, 1),
            ).view(x.size(0), -1)
        elif _is_hip:
            x = F.linear(x.flatten(1), self.proj.weight.flatten(1), self.proj.bias)
        else:
            x = self.proj(x).view(x.size(0), -1)
        return self.pos_emb(
            x,
            grid_thws,
            grid_thw_list=grid_thw_list,
            position_embeddings=position_embeddings,
        )


class Rope2DPosEmbRepeated(nn.Module):
    def __init__(
        self, dim: int, max_height: int, max_width: int, theta_base: float = 10000
    ):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4"
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()
        y_freqs = torch.outer(y_pos, freqs).float()
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        return freqs_cis.reshape(self.max_height, self.max_width, -1)

    def get_freqs_cis(
        self,
        grid_thws: torch.Tensor,
        device: torch.device,
        *,
        grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
    ) -> torch.Tensor:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis", self._precompute_freqs_cis(device), persistent=False
            )

        shapes = _resolve_grid_thw_list(grid_thws, grid_thw_list)
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ), (shapes, self.max_height, self.max_width)
        return torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )


class MLP2(nn.Module):
    def __init__(self, dims: List[int], activation, bias: bool = True):
        super().__init__()
        assert len(dims) == 3
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.activation(self.fc0(x)))


def _make_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    raise NotImplementedError(f"Not support norm_type: {norm_type}")


class MoonViTEncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        qkv_hidden_size: Optional[int] = None,
        norm_type: str = "layernorm",
        *,
        activation=F.gelu,
        attn_bias: bool = False,
        linear_bias: bool = True,
        attention_backend: str = "sdpa",
        attention_workspace: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.qkv_hidden_size = (
            hidden_dim if qkv_hidden_size is None else qkv_hidden_size
        )
        self.hidden_size_per_attention_head = self.qkv_hidden_size // self.num_heads

        self.norm0 = _make_norm(norm_type, hidden_dim)
        self.norm1 = _make_norm(norm_type, hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation, bias=linear_bias)
        self.wqkv = nn.Linear(hidden_dim, self.qkv_hidden_size * 3, bias=attn_bias)
        self.wo = nn.Linear(self.qkv_hidden_size, hidden_dim, bias=attn_bias)
        self.attention_backend = attention_backend
        if attention_backend == "auto":
            if not torch.cuda.is_available():
                implementation_backends = ()
            elif _is_hip:
                implementation_backends = ("triton_attn",)
            else:
                implementation_backends = ("triton_attn", "fa4")
        elif attention_backend == "sdpa":
            implementation_backends = ()
        else:
            implementation_backends = (attention_backend,)
        self.attention_backend_impls = nn.ModuleDict(
            {
                backend: QKV_BACKEND_IMPL[backend](
                    use_data_parallel=True,
                    workspace_buffer=attention_workspace,
                )
                for backend in implementation_backends
            }
        )

    def _attention(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        segment_bounds: SegmentBounds,
        rope_freqs_cis: torch.Tensor,
        forward_metadata: VisionAttentionMetadata,
        use_fused_rope: bool,
        selected_attention_backend: str,
    ) -> torch.Tensor:
        xqkv = self.wqkv(x)
        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        if use_fused_rope:
            xq, xk = apply_fused_qk_complex_rope(xq, xk, rope_freqs_cis)
        else:
            xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        if selected_attention_backend == "sdpa":
            attn_out = sdpa_varlen_attention(
                xq,
                xk,
                xv,
                cu_seqlens,
                segment_bounds=segment_bounds,
            )
        else:
            if selected_attention_backend == "flashinfer_cudnn":
                xv = xv.contiguous()
            attn_out = self.attention_backend_impls[selected_attention_backend](
                xq,
                xk,
                xv,
                cu_seqlens=cu_seqlens,
                bsz=1,
                seq_len=xq.shape[0],
                forward_metadata=forward_metadata,
            ).flatten(start_dim=-2)
        return self.wo(attn_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        segment_bounds: SegmentBounds,
        rope_freqs_cis: torch.Tensor,
        forward_metadata: VisionAttentionMetadata,
        use_fused_rope: bool,
        selected_attention_backend: str,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = self._attention(
            hidden_states,
            cu_seqlens,
            segment_bounds,
            rope_freqs_cis,
            forward_metadata,
            use_fused_rope,
            selected_attention_backend,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


@dataclass(frozen=True)
class KimiK3VisionForwardMetadata:
    grid_thw_list: Tuple[GridTHW, ...]
    segment_bounds: SegmentBounds
    rope_freqs_cis: torch.Tensor
    attention: VisionAttentionMetadata
    use_fused_rope: bool
    selected_attention_backend: str
    position_embeddings: Optional[torch.Tensor] = None


class MoonViT3dEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, block_cfg: dict) -> None:
        super().__init__()
        qkv_hidden_size = block_cfg.get("qkv_hidden_size") or block_cfg["hidden_dim"]
        attention_backend = _get_mm_attention_backend()
        if attention_backend != "auto" and attention_backend not in QKV_BACKEND_IMPL:
            raise ValueError(
                f"Unsupported Kimi-K3 vision attention backend: {attention_backend}"
            )
        attention_workspace = None
        if attention_backend == "flashinfer_cudnn" and torch.cuda.is_available():
            attention_workspace = torch.empty(
                FLASHINFER_WORKSPACE_SIZE_BYTES,
                dtype=torch.uint8,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        self.attention_backend = attention_backend
        if attention_backend == "auto":
            print_info_once(
                "Kimi-K3 vision attention uses shape-aware auto selection on "
                "B300/GB300 (Triton for small workloads, FA4 otherwise)."
            )
        self.attention_width = qkv_hidden_size
        self.rope_2d = Rope2DPosEmbRepeated(
            qkv_hidden_size // block_cfg["num_heads"], 512, 512
        )
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(
                    **block_cfg,
                    attention_backend=attention_backend,
                    attention_workspace=attention_workspace,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = _make_norm(
            block_cfg.get("norm_type", "layernorm"), hidden_dim
        )

    def precompile_fused_rope(self, dtype: torch.dtype, device: torch.device) -> bool:
        if not self.blocks:
            return False
        return precompile_fused_qk_complex_rope(
            num_heads=self.blocks[0].num_heads,
            head_dim=self.blocks[0].hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def precompile_attention_backend(
        self, dtype: torch.dtype, device: torch.device
    ) -> bool:
        if (
            self.attention_backend != "auto"
            or not self.blocks
            or device.type != "cuda"
            or torch.cuda.get_device_capability(device) != (10, 3)
            or not _is_fa4_available()
        ):
            return False

        block = self.blocks[0]
        if "fa4" not in block.attention_backend_impls:
            return False

        num_tokens = 256
        packed_qkv = torch.zeros(
            (
                num_tokens,
                3,
                block.num_heads,
                block.hidden_size_per_attention_head,
            ),
            dtype=dtype,
            device=device,
        )
        q = packed_qkv[:, 0].contiguous()
        k = packed_qkv[:, 1].contiguous()
        v = packed_qkv[:, 2]
        cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
        metadata = prepare_vision_attention_metadata(cu_seqlens, device=device)
        with torch.inference_mode():
            block.attention_backend_impls["fa4"](
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                bsz=1,
                seq_len=num_tokens,
                forward_metadata=metadata,
            )
        torch.cuda.synchronize(device)
        return True

    def prepare_forward_metadata(
        self,
        *,
        grid_thws: torch.Tensor,
        total_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
    ) -> KimiK3VisionForwardMetadata:
        shapes = _resolve_grid_thw_list(grid_thws, grid_thw_list)
        rope_freqs_cis = self.rope_2d.get_freqs_cis(
            grid_thws=grid_thws,
            device=device,
            grid_thw_list=shapes,
        )
        cumulative_lengths = [0]
        for t, h, w in shapes:
            cumulative_lengths.append(cumulative_lengths[-1] + t * h * w)
        cu_seqlens = torch.tensor(cumulative_lengths, dtype=torch.int32, device=device)

        if self.attention_backend == "flashinfer_cudnn":
            attention = prepare_flashinfer_cudnn_vision_attention_metadata(
                cu_seqlens,
                device=device,
                elem_per_token=self.attention_width,
            )
        else:
            attention = prepare_vision_attention_metadata(cu_seqlens, device=device)

        selected_attention_backend = _resolve_mm_attention_backend(
            self.attention_backend,
            max_seqlen=attention.max_seqlen,
            total_tokens=total_tokens,
            device=device,
        )
        return KimiK3VisionForwardMetadata(
            grid_thw_list=shapes,
            segment_bounds=tuple(zip(cumulative_lengths[:-1], cumulative_lengths[1:])),
            rope_freqs_cis=rope_freqs_cis,
            attention=attention,
            use_fused_rope=_can_use_fused_rope_for_shape(
                dtype=dtype,
                device=device,
                freqs_cis=rope_freqs_cis,
            ),
            selected_attention_backend=selected_attention_backend,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor,
        *,
        forward_metadata: Optional[KimiK3VisionForwardMetadata] = None,
        grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
    ) -> torch.Tensor:
        if forward_metadata is None:
            forward_metadata = self.prepare_forward_metadata(
                grid_thws=grid_thws,
                total_tokens=hidden_states.shape[0],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                grid_thw_list=grid_thw_list,
            )
            forward_metadata = replace(
                forward_metadata,
                use_fused_rope=_can_use_fused_rope(
                    hidden_states, forward_metadata.rope_freqs_cis
                ),
            )
        attention = forward_metadata.attention
        cu_seqlens = attention.cu_seqlens
        rope_freqs_cis = forward_metadata.rope_freqs_cis

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                forward_metadata.segment_bounds,
                rope_freqs_cis,
                attention,
                forward_metadata.use_fused_rope,
                forward_metadata.selected_attention_backend,
            )

        return self.final_layernorm(hidden_states)


class KimiK3VisionTower(nn.Module):
    def __init__(self, vision_config, **kwargs):
        super().__init__()
        config = vision_config
        self.config = config
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type
        if self.merge_type != "sd2_tpool":
            raise NotImplementedError(f"Not support merge_type: {self.merge_type}")

        hidden_size = getattr(config, "vt_hidden_size", None) or config.hidden_size
        num_heads = (
            getattr(config, "vt_num_attention_heads", None)
            or config.num_attention_heads
        )
        num_layers = (
            getattr(config, "vt_num_hidden_layers", None) or config.num_hidden_layers
        )
        intermediate_size = (
            getattr(config, "vt_intermediate_size", None) or config.intermediate_size
        )

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
            pos_emb_interpolation_mode=config.pos_emb_interpolation_mode,
            patch_embed_proj_bias=getattr(config, "patch_embed_proj_bias", True),
        )

        activation_func = getattr(config, "activation_func", "gelu_pytorch_tanh")
        if activation_func == "gelu_pytorch_tanh":
            activation = lambda x: F.gelu(x, approximate="tanh")
        elif activation_func == "gelu":
            activation = F.gelu
        else:
            raise NotImplementedError(f"Not support activation_func: {activation_func}")

        self.encoder = MoonViT3dEncoder(
            hidden_dim=hidden_size,
            num_layers=num_layers,
            block_cfg={
                "num_heads": num_heads,
                "hidden_dim": hidden_size,
                "qkv_hidden_size": getattr(config, "qkv_hidden_size", None),
                "mlp_dim": intermediate_size,
                "norm_type": getattr(config, "norm_type", "layernorm"),
                "activation": activation,
                "attn_bias": getattr(config, "attn_bias", True),
                "linear_bias": getattr(config, "linear_bias", True),
            },
        )
        self.cuda_graph_runner = None

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def precompile_fused_rope(self) -> bool:
        return self.encoder.precompile_fused_rope(self.dtype, self.device)

    def precompile_attention_backend(self) -> bool:
        return self.encoder.precompile_attention_backend(self.dtype, self.device)

    def prepare_forward_metadata(
        self,
        grid_thws: torch.Tensor,
        *,
        grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
        total_tokens: int,
        dtype: Optional[torch.dtype] = None,
    ) -> KimiK3VisionForwardMetadata:
        metadata = self.encoder.prepare_forward_metadata(
            grid_thws=grid_thws,
            total_tokens=total_tokens,
            dtype=self.dtype if dtype is None else dtype,
            device=self.device,
            grid_thw_list=grid_thw_list,
        )
        return replace(
            metadata,
            position_embeddings=self.patch_embed.pos_emb.position_embeddings(
                grid_thws, grid_thw_list=metadata.grid_thw_list
            ),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        *,
        grid_hw: Optional[torch.Tensor] = None,
        grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
        forward_metadata: Optional[KimiK3VisionForwardMetadata] = None,
    ) -> List[torch.Tensor]:
        # run_dp_sharded_mrope_vision_model calls rope_2d towers with
        # grid_hw=/max_seqlen= keywords (#30878); K3 grids are (t, h, w) and the
        # encoder derives its own varlen metadata, so max_seqlen is unused.
        if grid_thws is None:
            grid_thws = grid_hw
        assert grid_thws.ndim == 2 and grid_thws.size(1) == 3, grid_thws.shape
        if (
            forward_metadata is None
            and pixel_values.is_cuda
            and envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()
        ):
            if grid_thw_list is None:
                grid_thw_list = _resolve_grid_thw_list(grid_thws)
            else:
                grid_thw_list = _resolve_grid_thw_list(grid_thws, grid_thw_list)
            if self.cuda_graph_runner is None:
                from sglang.srt.multimodal.kimi_k3_vit_cuda_graph_runner import (
                    KimiK3ViTCudaGraphRunner,
                )

                self.cuda_graph_runner = KimiK3ViTCudaGraphRunner(
                    self,
                    capacity=(envs.SGLANG_KIMI_K3_VIT_CUDA_GRAPH_CACHE_CAPACITY.get()),
                    min_hits=envs.SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MIN_HITS.get(),
                    max_seqlen=(envs.SGLANG_KIMI_K3_VIT_CUDA_GRAPH_MAX_SEQLEN.get()),
                )
            return self.cuda_graph_runner.run(pixel_values, grid_thws, grid_thw_list)
        return self._forward_eager(
            pixel_values,
            grid_thws,
            grid_thw_list=grid_thw_list,
            forward_metadata=forward_metadata,
        )

    def _forward_eager(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
        *,
        grid_thw_list: Optional[Sequence[Sequence[int]]] = None,
        forward_metadata: Optional[KimiK3VisionForwardMetadata] = None,
    ) -> List[torch.Tensor]:
        if forward_metadata is not None:
            grid_thw_list = forward_metadata.grid_thw_list
        hidden_states = self.patch_embed(
            pixel_values,
            grid_thws,
            grid_thw_list=grid_thw_list,
            position_embeddings=(
                None
                if forward_metadata is None
                else forward_metadata.position_embeddings
            ),
        )
        hidden_states = self.encoder(
            hidden_states,
            grid_thws,
            forward_metadata=forward_metadata,
            grid_thw_list=grid_thw_list,
        )
        return tpool_patch_merger(
            hidden_states,
            grid_thws,
            merge_kernel_size=self.merge_kernel_size,
            grid_thw_list=grid_thw_list,
        )


class KimiK3MultiModalProjector(nn.Module):
    """PatchMergerMLPV2: bias-free two-layer MLP over merged patches with a
    post RMSNorm; K3 has no pre-norm, unlike the K2.5 projector."""

    def __init__(self, vision_config):
        super().__init__()
        config = vision_config
        mm_hidden_size = (
            getattr(config, "mm_hidden_size", None) or config.vt_hidden_size
        )
        merge_h, merge_w = config.merge_kernel_size
        self.hidden_size = mm_hidden_size * merge_h * merge_w
        text_hidden_size = getattr(config, "text_hidden_size", None) or getattr(
            config, "hidden_size"
        )
        eps = config.projector_ln_eps

        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden_size, text_hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(text_hidden_size, eps=eps)

    def forward(
        self, image_features: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        if isinstance(image_features, (list, tuple)):
            x = concat_or_single(
                [item.reshape(item.shape[0], -1) for item in image_features]
            )
        else:
            x = image_features.reshape(image_features.shape[0], -1)
        return self.post_norm(self.proj(x))
