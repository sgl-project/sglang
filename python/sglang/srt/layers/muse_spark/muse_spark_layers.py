"""Small, torch-native layer primitives used by the Muse Spark model."""

from __future__ import annotations

import enum
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.distributed import divide
from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    LinearParallelGroup,
    resolve_linear_parallel_group,
)
from sglang.srt.utils import is_cpu

_is_cpu = is_cpu()


class GainCenterType(enum.Enum):
    ONE = "one"
    ZERO = "zero"
    INVERSE_RESIDUAL_INDEX = "inverse_residual_index"
    INVERSE_RESIDUAL_INDEX_V2 = "inverse_residual_index_v2"
    INVERSE_SQRT_RESIDUAL_INDEX = "inverse_sqrt_residual_index"
    INVERSE_SQRT_LAYERS = "inverse_sqrt_layers"


def calculate_gain_center(
    gain_center_type: str, residual_id: int, n_layers: Optional[int] = None
) -> float:
    gain_center = GainCenterType(gain_center_type)
    if gain_center == GainCenterType.ONE:
        return 1.0
    if gain_center == GainCenterType.ZERO:
        return 0.0
    if gain_center == GainCenterType.INVERSE_RESIDUAL_INDEX:
        return 1.0 / (2.0 * math.ceil(residual_id / 2.0))
    if gain_center == GainCenterType.INVERSE_RESIDUAL_INDEX_V2:
        return 1.0 / (residual_id + 1.0)
    if gain_center == GainCenterType.INVERSE_SQRT_RESIDUAL_INDEX:
        return 1.0 / math.sqrt(2.0 * math.ceil(residual_id / 2.0))
    if gain_center == GainCenterType.INVERSE_SQRT_LAYERS:
        if n_layers is None:
            raise ValueError("n_layers is required for inverse_sqrt_layers")
        return 1.0 / math.sqrt(n_layers)
    raise ValueError(f"Unknown gain_center_type: {gain_center_type}")


class MuseSpark1xRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        has_weight: bool = True,
        is_post_norm: bool = False,
        zero_centered_gamma: bool = False,
        post_norm_gain_center_type: str = "one",
        output_norm_gain_center_type: str = "zero",
        residual_id: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.has_weight = has_weight
        self.zero_centered_gamma = zero_centered_gamma
        self.weight = nn.Parameter(torch.empty(hidden_size)) if has_weight else None
        # object.__setattr__ so this inference-only cache never registers as a
        # second Parameter; filled by precompute_effective_weight() after load.
        object.__setattr__(self, "_frozen_weight", None)
        self.gain_center = 1.0
        if residual_id is not None and is_post_norm:
            self.gain_center = calculate_gain_center(
                post_norm_gain_center_type, residual_id
            )
        elif residual_id == -1:
            self.gain_center = calculate_gain_center(
                output_norm_gain_center_type, residual_id
            )

    def _get_effective_weight(self) -> Optional[torch.Tensor]:
        if self.weight is None or not self.zero_centered_gamma:
            return self.weight
        if self._frozen_weight is not None:
            return self._frozen_weight
        return self.weight + self.gain_center

    @torch.no_grad()
    def precompute_effective_weight(self) -> None:
        if self.weight is not None and self.zero_centered_gamma:
            object.__setattr__(
                self, "_frozen_weight", (self.weight + self.gain_center).detach()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        dtype = x.dtype
        weight = self._get_effective_weight()
        if (
            _is_cpu
            and x.dtype in (torch.bfloat16, torch.float16)
            and (weight is None or weight.dtype == x.dtype)
        ):
            x_2d = x.reshape(-1, self.hidden_size)
            output = (
                torch.ops.sgl_kernel.l2norm_cpu(x_2d, self.eps)
                if weight is None
                else torch.ops.sgl_kernel.rmsnorm_cpu(x_2d, weight, self.eps)
            )
            return output.reshape(x.shape)
        return F.rms_norm(x, (self.hidden_size,), weight, self.eps).to(dtype)


class MuseSpark1xRMSNormWithInputScale(MuseSpark1xRMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        dtype = x.dtype
        weight = self._get_effective_weight()
        if weight is not None:
            x = weight * x
        if _is_cpu and x.dtype in (torch.bfloat16, torch.float16):
            return torch.ops.sgl_kernel.l2norm_cpu(
                x.reshape(-1, self.hidden_size), self.eps
            ).reshape(x.shape)
        return F.rms_norm(x, (self.hidden_size,), None, self.eps).to(dtype)


class MuseSpark1xResidualConnection(nn.Module):
    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        return stream + branch.to(stream.dtype)


class MuseSpark1xNormPreservingResidualConnection(nn.Module):
    """Training-compatible gated residual implemented only with torch ops."""

    def __init__(
        self,
        dim: int,
        temperature: float = 0.3,
        channel_wise: bool = True,
        use_stream_gate: bool = True,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_stream_gate = use_stream_gate
        self.eps = eps
        self.gate = nn.Parameter(torch.empty(dim if channel_wise else 1))
        self.register_buffer("_alpha", None, persistent=False)
        self.register_buffer("_beta", None, persistent=False)

    def _coefficients(
        self, w: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, float], torch.Tensor]:
        beta = torch.sigmoid(w)
        if not self.use_stream_gate:
            return 1.0, beta
        alpha_sq = torch.sigmoid(-w) * (1.0 + beta)
        return torch.sqrt(torch.clamp(alpha_sq, min=self.eps)), beta

    def precompute_coefficients(self) -> None:
        with torch.no_grad():
            alpha, beta = self._coefficients(
                self.gate.view(1, -1).float() / self.temperature
            )
            self._alpha = (
                alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha)
            )
            self._beta = beta

    def forward(self, stream: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        if stream.numel() == 0:
            return stream
        if self._alpha is None:
            alpha, beta = self._coefficients(
                self.gate.view(1, -1).float() / self.temperature
            )
        else:
            alpha, beta = self._alpha, self._beta
        return (alpha * stream + beta * branch.to(stream.dtype)).to(stream.dtype)


class QKVOGateParallelLinear(ColumnParallelLinear):
    """Tensor-parallel Q/K/V projection with one additional query-sized gate."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = False,
        prefix: str = "",
        parallel_group: LinearParallelGroup = "attn_tp",
    ):
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        _, tp_size = resolve_linear_parallel_group(parallel_group)
        self.num_heads = divide(total_num_heads, tp_size)
        if tp_size >= total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, total_num_kv_heads)
        else:
            self.num_kv_heads = divide(total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
        self.q_proj_shard_size = self.num_heads * head_size
        self.kv_proj_shard_size = self.num_kv_heads * head_size
        self.g_proj_shard_size = self.q_proj_shard_size
        local = (
            self.q_proj_shard_size
            + 2 * self.kv_proj_shard_size
            + self.g_proj_shard_size
        )
        super().__init__(
            input_size=hidden_size,
            output_size=local * tp_size,
            bias=bias,
            gather_output=False,
            quant_config=None,
            prefix=prefix,
            parallel_group=parallel_group,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):
        if loaded_shard_id not in ("q", "k", "v", "g"):
            raise ValueError(f"Unsupported QKV gate shard: {loaded_shard_id}")
        sizes = {
            "q": self.q_proj_shard_size,
            "k": self.kv_proj_shard_size,
            "v": self.kv_proj_shard_size,
            "g": self.g_proj_shard_size,
        }
        offsets = {
            "q": 0,
            "k": self.q_proj_shard_size,
            "v": self.q_proj_shard_size + self.kv_proj_shard_size,
            "g": self.q_proj_shard_size + 2 * self.kv_proj_shard_size,
        }
        output_dim = getattr(param, "output_dim", 0)
        tp_rank, _ = get_group_rank_size(self.tp_group)
        shard_size = sizes[loaded_shard_id]
        param_data = param.data.narrow(output_dim, offsets[loaded_shard_id], shard_size)
        if loaded_shard_id in ("q", "g"):
            start = tp_rank * shard_size
        else:
            start = (tp_rank // self.num_kv_head_replicas) * shard_size
        if _is_cpu:
            # Unaligned CPU TP pads the head count, so the parameter is wider
            # than the checkpoint tensor and a plain narrow runs off its end.
            from sglang.srt.model_loader.weight_utils import (
                narrow_padded_param_and_loaded_weight,
            )

            param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                param_data, loaded_weight, 0, start, output_dim, shard_size
            )
        else:
            loaded_weight = loaded_weight.narrow(output_dim, start, shard_size)
        param_data.copy_(loaded_weight)
