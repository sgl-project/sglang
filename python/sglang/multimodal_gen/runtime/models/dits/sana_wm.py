# SPDX-License-Identifier: Apache-2.0

import functools
import math
from typing import Any, Callable, List, NoReturn, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from sglang.multimodal_gen.configs.models.dits.sana_wm import SanaWMConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.layernorm import tensor_parallel_rms_norm
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
)
from sglang.multimodal_gen.runtime.realtime.causal_state import RealtimeCausalDiTState
from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SANA_WM_CROSS_ATTN_KV_CACHE_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_SANA_WM_REQUEST_RUNTIME_CACHE_NAMESPACE = "sana_wm"


def _sana_wm_deterministic_inference_enabled() -> bool:
    return get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_INFERENCE")


def _sana_wm_get_forward_context_or_none():
    try:
        return get_forward_context()
    except AssertionError:
        return None


@functools.lru_cache(maxsize=1)
def _get_sana_wm_triton_main_gdn():
    from sglang.jit_kernel.diffusion.sana_wm.main_gdn import (
        can_use_sana_wm_fused_bigdn_bidi,
        can_use_sana_wm_fused_bigdn_bidi_with_inv_rms,
        sana_wm_fused_bigdn_bidi,
        sana_wm_fused_bigdn_bidi_with_inv_rms,
    )

    return (
        sana_wm_fused_bigdn_bidi,
        can_use_sana_wm_fused_bigdn_bidi,
        sana_wm_fused_bigdn_bidi_with_inv_rms,
        can_use_sana_wm_fused_bigdn_bidi_with_inv_rms,
    )


@functools.lru_cache(maxsize=1)
def _get_sana_wm_triton_cam_preprocess():
    from sglang.jit_kernel.diffusion.sana_wm.camera_preprocess import (
        can_use_sana_wm_cam_gdn_preprocess,
        can_use_sana_wm_cam_gdn_preprocess_with_inv_rms,
        can_use_sana_wm_cam_output_apply_o,
        can_use_sana_wm_cam_softmax_preprocess_with_inv_rms,
        sana_wm_cam_gdn_preprocess,
        sana_wm_cam_gdn_preprocess_with_inv_rms,
        sana_wm_cam_output_apply_o,
        sana_wm_cam_softmax_preprocess_with_inv_rms,
    )

    return (
        sana_wm_cam_gdn_preprocess,
        can_use_sana_wm_cam_gdn_preprocess,
        sana_wm_cam_output_apply_o,
        can_use_sana_wm_cam_output_apply_o,
        sana_wm_cam_gdn_preprocess_with_inv_rms,
        can_use_sana_wm_cam_gdn_preprocess_with_inv_rms,
        sana_wm_cam_softmax_preprocess_with_inv_rms,
        can_use_sana_wm_cam_softmax_preprocess_with_inv_rms,
    )


@functools.lru_cache(maxsize=1)
def _get_sana_wm_triton_cam_scan():
    from sglang.jit_kernel.diffusion.sana_wm.camera_scan import (
        can_use_sana_wm_cam_scan_bidi_chunkwise,
        sana_wm_cam_scan_bidi_chunkwise,
    )

    return (
        sana_wm_cam_scan_bidi_chunkwise,
        can_use_sana_wm_cam_scan_bidi_chunkwise,
    )


def _tensor_cache_key(tensor: torch.Tensor) -> Tuple:
    """Compute a stable cache key for *tensor*.

    Uses ``data_ptr()`` (the memory address of element 0) as a content proxy.
    During inference weights are frozen and request tensors are always fresh
    allocations, so the same ``data_ptr`` implies identical content.  We
    intentionally omit ``tensor._version`` (a private PyTorch C++ field) to
    avoid relying on an undocumented API that can raise ``RuntimeError`` inside
    ``torch.compile`` or CUDA-graph capture contexts.
    """
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.device),
        tensor.dtype,
        tensor.data_ptr(),
    )


def _sana_wm_torch_compile_disable(fn: Callable) -> Callable:
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    if callable(disable):
        return disable(fn)

    dynamo = getattr(torch, "_dynamo", None)
    disable = getattr(dynamo, "disable", None)
    if callable(disable):
        return disable(fn)

    return fn


def _sana_wm_request_runtime_cache_enabled(enabled: bool = True) -> bool:
    if torch.is_grad_enabled():
        return False
    if not enabled:
        return False

    compiler = getattr(torch, "compiler", None)
    compiler_is_compiling = getattr(compiler, "is_compiling", None)
    if callable(compiler_is_compiling) and compiler_is_compiling():
        return False

    dynamo = getattr(torch, "_dynamo", None)
    dynamo_is_compiling = getattr(dynamo, "is_compiling", None)
    if callable(dynamo_is_compiling) and dynamo_is_compiling():
        return False

    return True


def _sana_wm_tp_world_size() -> int:
    if not model_parallel_is_initialized():
        return 1
    return get_tp_world_size()


def _sana_wm_tp_rank() -> int:
    if not model_parallel_is_initialized():
        return 0
    return get_tp_rank()


def _sana_wm_sp_world_size() -> int:
    if not model_parallel_is_initialized():
        return 1
    return get_sp_world_size()


def _sana_wm_sequence_shard_enabled(sp_size: int) -> bool:
    if sp_size <= 1:
        return False
    raise NotImplementedError(
        f"SANA-WM does not support sequence parallelism (sp_size={sp_size} > 1). "
        "Its stage-1 DiT contains layout-dependent cross-frame operators "
        "(bidirectional GDN scan, GLUMBConvTemp temporal convolution, camera "
        "UCPE, and Plucker conditioning) that require operator-aware "
        "state/reduction/halo exchange before SP can be enabled. "
        "Action: remove --sp-size (or set --sp-size 1) and use tensor "
        "parallelism instead, e.g. --tp-size 2 or --tp-size 4."
    )


def _sana_wm_linear(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = module(x)
    return out[0] if isinstance(out, tuple) else out


class _SanaWMLinearSequential(nn.Sequential):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for module in self:
            input = _sana_wm_linear(module, input)
        return input


def _sana_wm_column_parallel_or_linear(
    input_size: int,
    output_size: int,
    *,
    bias: bool = True,
    gather_output: bool = False,
) -> nn.Module:
    tp_size = _sana_wm_tp_world_size()
    if tp_size <= 1:
        return nn.Linear(input_size, output_size, bias=bias)
    if output_size % tp_size != 0:
        raise ValueError(
            "SANA-WM tensor parallelism requires column-parallel output_size "
            f"to be divisible by tp_size, got output_size={output_size}, "
            f"tp_size={tp_size}."
        )
    return ColumnParallelLinear(
        input_size,
        output_size,
        bias=bias,
        gather_output=gather_output,
    )


def _sana_wm_row_parallel_or_linear(
    input_size: int,
    output_size: int,
    *,
    bias: bool = True,
    input_is_parallel: bool = True,
) -> nn.Module:
    tp_size = _sana_wm_tp_world_size()
    if tp_size <= 1:
        return nn.Linear(input_size, output_size, bias=bias)
    if input_size % tp_size != 0:
        raise ValueError(
            "SANA-WM tensor parallelism requires row-parallel input_size "
            f"to be divisible by tp_size, got input_size={input_size}, "
            f"tp_size={tp_size}."
        )
    return RowParallelLinear(
        input_size,
        output_size,
        bias=bias,
        input_is_parallel=input_is_parallel,
    )


def _sana_wm_all_gather_hidden(x: torch.Tensor, tp_size: int) -> torch.Tensor:
    if tp_size == 1:
        return x
    return tensor_model_parallel_all_gather(x.contiguous(), dim=-1)


def _sana_wm_tp_rms_norm(
    x: torch.Tensor,
    norm: nn.Module,
    *,
    tp_size: int,
) -> torch.Tensor:
    if tp_size == 1 or not isinstance(norm, _RMSNorm):
        return norm(x)
    if norm.weight.shape[0] == x.shape[-1]:
        return norm(x)
    return tensor_parallel_rms_norm(x, norm)


def _sana_wm_tp_qk_rms_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.Module,
    k_norm: nn.Module,
    *,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SANA-WM TP RMSNorm to Q/K, fusing their variance all-reduce."""
    if (
        tp_size == 1
        or not isinstance(q_norm, _RMSNorm)
        or not isinstance(k_norm, _RMSNorm)
    ):
        return q_norm(q), k_norm(k)
    if q_norm.weight.shape[0] == q.shape[-1] and k_norm.weight.shape[0] == k.shape[-1]:
        return q_norm(q), k_norm(k)

    if (
        q_norm.weight.shape[0] == q.shape[-1]
        or k_norm.weight.shape[0] == k.shape[-1]
        or q.shape != k.shape
    ):
        return (
            _sana_wm_tp_rms_norm(q, q_norm, tp_size=tp_size),
            _sana_wm_tp_rms_norm(k, k_norm, tp_size=tp_size),
        )

    tp_rank = get_tp_rank()
    q_weight = q_norm.weight.tensor_split(tp_size)[tp_rank].float()
    k_weight = k_norm.weight.tensor_split(tp_size)[tp_rank].float()
    q_fp32 = q.float()
    k_fp32 = k.float()
    q_var = q_fp32.pow(2).mean(dim=-1, keepdim=True)
    k_var = k_fp32.pow(2).mean(dim=-1, keepdim=True)
    q_tokens = q_var.shape[1]
    both_var = torch.cat([q_var, k_var], dim=1)
    both_var = get_tp_group().all_reduce(
        both_var,
        op=torch._C._distributed_c10d.ReduceOp.AVG,
    )
    q_var = both_var[:, :q_tokens]
    k_var = both_var[:, q_tokens:]
    q_out = q_fp32 * torch.rsqrt(q_var + q_norm.variance_epsilon) * q_weight
    k_out = k_fp32 * torch.rsqrt(k_var + k_norm.variance_epsilon) * k_weight
    return q_out.to(q.dtype), k_out.to(k.dtype)


def _sana_wm_local_rms_norm_weight(
    norm: nn.Module,
    *,
    local_hidden_size: int,
    tp_rank: int,
    tp_size: int,
) -> Optional[torch.Tensor]:
    if not isinstance(norm, _RMSNorm):
        return None
    weight = norm.weight
    if weight.shape[0] == local_hidden_size:
        return weight
    if weight.shape[0] != local_hidden_size * tp_size:
        return None
    return weight.narrow(0, tp_rank * local_hidden_size, local_hidden_size)


def _sana_wm_tp_qk_inv_rms(
    qkv: torch.Tensor,
    *,
    norm_eps: float,
    tp_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _sana_wm_tp_pair_inv_rms(
        qkv[:, :, 0],
        qkv[:, :, 1],
        norm_eps=norm_eps,
        tp_size=tp_size,
    )


def _sana_wm_tp_pair_inv_rms(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    norm_eps: float,
    tp_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.shape != k.shape:
        raise ValueError(f"q/k shape mismatch: {q.shape}, {k.shape}.")
    if q.dim() < 3:
        raise ValueError(f"Expected q/k to have hidden dims, got {q.shape}.")

    local_hidden_size = math.prod(q.shape[2:])
    reduce_dims = tuple(range(2, q.dim()))
    q_local_sumsq = q.float().square().sum(dim=reduce_dims)
    k_local_sumsq = k.float().square().sum(dim=reduce_dims)
    local_sumsq = torch.stack((q_local_sumsq, k_local_sumsq), dim=-1).contiguous()
    if tp_size > 1:
        local_sumsq = tensor_model_parallel_all_reduce(local_sumsq)

    full_hidden_size = float(local_hidden_size * tp_size)
    inv_rms = torch.rsqrt(local_sumsq / full_hidden_size + norm_eps)
    return inv_rms[..., 0].contiguous(), inv_rms[..., 1].contiguous()


class _RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        scale_factor: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not _sana_wm_deterministic_inference_enabled()
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and self.weight.dtype == x.dtype
            and x.shape[-1] == self.weight.shape[0]
        ):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)

        x_in = x
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms * self.weight.to(dtype=x32.dtype)).type_as(x_in)


class _ShortConvolution(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, 1, kernel_size))
        set_weight_attrs(
            self.weight,
            {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            },
        )
        with torch.no_grad():
            self.weight[:, 0, -1] = 1.0

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        output_dim = getattr(param, "output_dim", None)
        if output_dim is not None and tuple(param.shape) != tuple(loaded_weight.shape):
            shard_size = param.shape[output_dim]
            start_idx = _sana_wm_tp_rank() * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # x: (B, T, C) -> (B, C, T) for conv1d
        x_bct = x.transpose(1, 2)
        x_pad = F.pad(x_bct, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, self.weight, bias=None, groups=self.hidden_size)
        return y.transpose(1, 2), None


def _bidirectional_short_conv(
    x: torch.Tensor,  # (B*S, T, C)
    conv: _ShortConvolution,
) -> torch.Tensor:
    y_fwd, _ = conv(x)
    y_bwd, _ = conv(x.flip(1))
    y_bwd = y_bwd.flip(1)
    w_center = conv.weight[:, 0, -1]  # (C,)
    center = x * w_center.view(1, 1, -1)
    return (y_fwd + y_bwd - center).to(x.dtype)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._init_freqs_buffer()

    def _init_freqs_buffer(self) -> None:
        # Extracted so SanaWMTransformer3DModel.post_load_weights can re-run
        # it: this is a persistent=False buffer, so it is not in the upstream
        # checkpoint and stays on meta after FSDP weight load.
        h_dim = w_dim = 2 * (self.attention_head_dim // 6)
        t_dim = self.attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim,
                self.max_seq_len,
                self.theta,
                use_real=False,
                repeat_interleave_real=False,
                freqs_dtype=torch.float64,
            )
            freqs.append(freq)
        self.register_buffer("_freqs", torch.cat(freqs, dim=1), persistent=False)

    def forward(self, fhw: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
        ppf, pph, ppw = fhw
        freqs = self._freqs.to(device)
        d = self.attention_head_dim
        t_size = d // 2 - 2 * (d // 6)
        h_size = d // 6
        w_size = d // 6
        ft, fh, fw = freqs.split_with_sizes([t_size, h_size, w_size], dim=1)

        freqs_t = ft[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = fh[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = fw[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        out = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)
        return out.reshape(1, 1, ppf * pph * ppw, -1)


def _apply_rotary_emb_dn(
    hidden_states: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Apply complex RoPE to a tensor of shape ``(B, H, D, N)`` (GDN layout).

    ``freqs`` is complex with shape ``(1, 1, N, D/2)``.
    """
    # (B, H, D, N) -> (B, H, N, D)
    x = hidden_states.permute(0, 1, 3, 2).to(torch.float64).contiguous()
    x_c = torch.view_as_complex(x.unflatten(-1, (-1, 2)))
    y = torch.view_as_real(x_c * freqs).flatten(-2, -1)
    return y.permute(0, 1, 3, 2).type_as(hidden_states)


def _apply_rotary_emb_bhnd(
    hidden_states: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Apply complex RoPE to ``(B, H, N, D)`` (softmax attention layout)."""
    x = hidden_states.to(torch.float64).contiguous()
    x_c = torch.view_as_complex(x.unflatten(-1, (-1, 2)))
    y = torch.view_as_real(x_c * freqs).flatten(-2, -1)
    return y.type_as(hidden_states)


def _sana_wm_repeated_batch_factor(
    source_batch: int,
    target_batch: int,
    *,
    name: str,
) -> int:
    if source_batch <= 0 or target_batch <= 0:
        raise ValueError(
            f"SANA-WM {name} batch dimensions must be positive, got "
            f"source={source_batch}, target={target_batch}."
        )
    if source_batch == target_batch:
        return 1
    if target_batch % source_batch != 0:
        raise ValueError(
            f"SANA-WM {name} batch mismatch: source batch {source_batch} cannot "
            f"be repeated to runtime batch {target_batch}."
        )
    return target_batch // source_batch


def _sana_wm_materialize_repeated_batch(
    tensor: Optional[torch.Tensor],
    target_batch: int,
    *,
    name: str,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    repeats = _sana_wm_repeated_batch_factor(
        tensor.shape[0],
        target_batch,
        name=name,
    )
    if repeats == 1:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(target_batch, *tensor.shape[1:]).contiguous()
    return torch.cat([tensor] * repeats, dim=0).contiguous()


def _sana_wm_optional_tensor_cache_key(tensor: Optional[torch.Tensor]) -> Tuple | None:
    return None if tensor is None else _tensor_cache_key(tensor)


def _sana_wm_materialize_repeated_raymats(
    raymats_flat: Optional[torch.Tensor],
    raymats_t: Optional[torch.Tensor],
    raymats_inv: Optional[torch.Tensor],
    target_batch: int,
    *,
    name: str,
    cache: Optional[dict] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    source_tensors = tuple(
        tensor
        for tensor in (raymats_flat, raymats_t, raymats_inv)
        if tensor is not None
    )
    if not source_tensors:
        return raymats_flat, raymats_t, raymats_inv
    if all(tensor.shape[0] == target_batch for tensor in source_tensors):
        return raymats_flat, raymats_t, raymats_inv

    cache_key = None
    if cache is not None and not torch.is_grad_enabled():
        cache_key = (
            "raymats",
            target_batch,
            _sana_wm_optional_tensor_cache_key(raymats_flat),
            _sana_wm_optional_tensor_cache_key(raymats_t),
            _sana_wm_optional_tensor_cache_key(raymats_inv),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    result = (
        _sana_wm_materialize_repeated_batch(
            raymats_flat,
            target_batch,
            name=f"{name} ray matrix",
        ),
        _sana_wm_materialize_repeated_batch(
            raymats_t,
            target_batch,
            name=f"{name} ray matrix transpose",
        ),
        _sana_wm_materialize_repeated_batch(
            raymats_inv,
            target_batch,
            name=f"{name} ray matrix inverse",
        ),
    )
    if cache_key is not None:
        cache[cache_key] = result
    return result


def _sana_wm_add_repeated_batch(
    x: torch.Tensor,
    addend: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if addend.shape[0] == x.shape[0]:
        return x + addend

    repeats = _sana_wm_repeated_batch_factor(
        addend.shape[0],
        x.shape[0],
        name=name,
    )
    if torch.is_grad_enabled():
        return x + torch.cat([addend] * repeats, dim=0)

    base_batch = addend.shape[0]
    for idx in range(repeats):
        start = idx * base_batch
        x[start : start + base_batch].add_(addend)
    return x


def _sana_wm_chunk_index_from_chunk_size(
    T: int,
    chunk_size: int,
    strategy: str = "uniform",
) -> list[int]:
    """Return upstream-style temporal chunk start indices."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}.")

    strategy = "uniform" if strategy is None else str(strategy).lower()

    if strategy in ("uniform", "default"):
        indices = list(range(0, T, chunk_size))
        if len(indices) > 1 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    if strategy in ("first_frame", "first_frame_alone", "first_frame_only"):
        if T <= 1:
            return [0]
        indices = [0] + list(range(1, T, chunk_size))
        if len(indices) > 2 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    if strategy in ("first_plus_one", "first_chunk_plus_one"):
        if T <= chunk_size + 1:
            return [0]
        indices = [0] + list(range(chunk_size + 1, T, chunk_size))
        if len(indices) > 1 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    raise ValueError(
        f"Unknown chunk_split_strategy '{strategy}'. Supported: "
        "uniform, first_frame, first_plus_one."
    )


def _sana_wm_normalize_chunk_index(
    chunk_index: Optional[List[int]],
    T: int,
    chunk_size: Optional[int] = None,
    chunk_split_strategy: str = "uniform",
) -> list[int]:
    if chunk_index is not None:
        normalized = [int(idx) for idx in chunk_index]
        if not normalized or normalized[0] != 0:
            normalized = [0] + [idx for idx in normalized if idx > 0]
        normalized = [idx for idx in normalized if idx < T]
        if not normalized:
            normalized = [0]
    else:
        if chunk_size is None:
            raise ValueError("Either chunk_index or chunk_size must be provided.")
        normalized = _sana_wm_chunk_index_from_chunk_size(
            T,
            int(chunk_size),
            strategy=chunk_split_strategy,
        )

    if normalized[-1] != T:
        normalized.append(T)
    if any(end <= start for start, end in zip(normalized[:-1], normalized[1:])):
        raise ValueError(f"chunk_index must be strictly increasing, got {normalized}.")
    return normalized


def _sana_wm_padded_attention_head_size(head_size: int) -> int:
    try:
        from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
            FlashAttentionBackend,
        )

        supported_head_sizes = FlashAttentionBackend.get_supported_head_sizes()
    except ImportError:
        supported_head_sizes = (32, 64, 96, 128, 160, 192, 224, 256)

    if head_size in supported_head_sizes:
        return head_size
    for supported_head_size in supported_head_sizes:
        if head_size < supported_head_size:
            return supported_head_size
    return head_size


class _SanaWMPaddedLocalAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.padded_head_size = _sana_wm_padded_attention_head_size(head_size)
        self.pad_size = self.padded_head_size - self.head_size
        self.softmax_scale = head_size**-0.5 if softmax_scale is None else softmax_scale

        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.padded_head_size,
            num_kv_heads=num_kv_heads,
            softmax_scale=self.softmax_scale,
            causal=causal,
            **extra_impl_args,
        )
        self.num_kv_heads = self.attn.num_kv_heads
        self.backend = self.attn.backend
        self.dtype = self.attn.dtype

    def _pad_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.pad_size <= 0:
            return tensor
        return F.pad(tensor, (0, self.pad_size))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pad_size <= 0 or attn_mask is not None:
            return self.attn(q, k, v, attn_mask=attn_mask)

        out = self.attn(
            self._pad_head_dim(q),
            self._pad_head_dim(k),
            self._pad_head_dim(v),
            attn_mask=attn_mask,
        )
        return out[..., : self.head_size]


def _make_sana_wm_local_attention(
    *,
    num_heads: int,
    head_size: int,
    pad_head_dim_to_flash: bool = False,
    **kwargs,
) -> nn.Module:
    attn_cls = _SanaWMPaddedLocalAttention if pad_head_dim_to_flash else LocalAttention
    return attn_cls(num_heads=num_heads, head_size=head_size, **kwargs)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _slice_rope_for_cam(
    rotary_emb: Optional[torch.Tensor],
    head_dim: int,
    rope_dim: int,
) -> Optional[torch.Tensor]:
    """Re-slice WanRotaryPosEmbed output to a smaller rope_dim."""
    if rotary_emb is None:
        return None
    orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
    orig_h_size = head_dim // 6
    new_t_size = rope_dim // 2 - 2 * (rope_dim // 6)
    new_h_size = rope_dim // 6
    new_w_size = rope_dim // 6
    t_part = rotary_emb[..., :new_t_size]
    h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
    w_part = rotary_emb[
        ..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size
    ]
    return torch.cat([t_part, h_part, w_part], dim=-1)


def _build_ucpe_raymat_bundle(
    raymats: torch.Tensor,  # (B, N, 4, 4) -- ray<-world
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    P = raymats.contiguous()
    P_T = P.transpose(-1, -2).contiguous()
    P_inv = _invert_SE3(P).contiguous()
    return P, P_T, P_inv, {}


def _compute_fov_from_focal(focal: torch.Tensor, image_size: int) -> torch.Tensor:
    """fov = 2 * atan(image_size / (2 * focal))"""
    return 2.0 * torch.atan(image_size / (2.0 * focal.clamp(min=1e-6)))


def _unproject_grid(
    x_fov: torch.Tensor,
    y_fov: torch.Tensor,
    H: int,
    W: int,
    cx: torch.Tensor,
    cy: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    B, F_dim = x_fov.shape
    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    u = u.view(1, 1, 1, W).expand(B, F_dim, H, W)
    v = v.view(1, 1, H, 1).expand(B, F_dim, H, W)
    cx_e = cx.view(B, F_dim, 1, 1)
    cy_e = cy.view(B, F_dim, 1, 1)
    tan_x = torch.tan(x_fov / 2.0).view(B, F_dim, 1, 1)
    tan_y = torch.tan(y_fov / 2.0).view(B, F_dim, 1, 1)
    dx = (u - cx_e) / max(W, 1) * 2.0 * tan_x
    dy = (v - cy_e) / max(H, 1) * 2.0 * tan_y
    dz = torch.ones_like(dx)
    d = torch.stack([dx, dy, dz], dim=-1)
    return F.normalize(d, dim=-1)


def process_camera_conditions_ucpe(
    camera_conditions: torch.Tensor,  # (B, F, 20)
    HW: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    B, F_dim, _ = camera_conditions.shape
    _, H, W = HW
    device = camera_conditions.device
    dtype = camera_conditions.dtype

    c2w_flat = camera_conditions[..., :16]
    C_to_W = c2w_flat.view(B, F_dim, 4, 4)

    fx = camera_conditions[..., 16]
    fy = camera_conditions[..., 17]
    cx = camera_conditions[..., 18]
    cy = camera_conditions[..., 19]

    image_h = H * patch_size[1]
    image_w = W * patch_size[2]
    x_fov = _compute_fov_from_focal(fx, image_w)
    y_fov = _compute_fov_from_focal(fy, image_h)

    cx_lat = cx / float(patch_size[2])
    cy_lat = cy / float(patch_size[1])

    d_cam = _unproject_grid(x_fov, y_fov, H, W, cx_lat, cy_lat, device, dtype)

    R_c2w = C_to_W[..., :3, :3]
    t_c2w = C_to_W[..., :3, 3]
    d_world = torch.einsum("bfij,bfhwj->bfhwi", R_c2w, d_cam)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    cam_y = R_c2w[..., :, 1].view(B, F_dim, 1, 1, 3).expand(B, F_dim, H, W, 3)
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-6)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-6)
    R_ray_to_world = torch.stack([x_ray, y_ray, z_ray], dim=-1)

    R_w_to_ray = R_ray_to_world.transpose(-1, -2)
    t_w_to_ray = -torch.einsum(
        "bfhwij,bfj->bfhwi",
        R_w_to_ray,
        t_c2w,
    )
    raymats = torch.zeros(B, F_dim, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w_to_ray
    raymats[..., :3, 3] = t_w_to_ray
    raymats[..., 3, 3] = 1.0
    invalid = torch.isnan(d_world).any(dim=-1)
    eye = torch.eye(4, device=device, dtype=dtype).view(1, 1, 1, 1, 4, 4)
    return torch.where(invalid[..., None, None], eye, raymats)


class PatchEmbedMS3D(nn.Module):
    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        in_chans: int,
        embed_dim: int,
        kernel_size: Optional[Tuple[int, int, int]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = kernel_size or patch_size
        assert patch_size[0] == 1, "Temporal patch must be 1 for SANA-WM."
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, T, H, W)
        return x.flatten(2).transpose(1, 2)  # (B, T*H*W, D)


class _UpstreamMlp(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int
    ) -> None:
        super().__init__()
        self.fc1 = _sana_wm_column_parallel_or_linear(
            in_features,
            hidden_features,
            bias=True,
            gather_output=False,
        )
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = _sana_wm_row_parallel_or_linear(
            hidden_features,
            out_features,
            bias=True,
            input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _sana_wm_linear(self.fc1, x)
        x = self.act(x)
        return _sana_wm_linear(self.fc2, x)


class CaptionEmbedder(nn.Module):
    def __init__(
        self, in_channels: int, hidden_size: int, token_num: int = 300
    ) -> None:
        super().__init__()
        self.y_proj = _UpstreamMlp(in_channels, hidden_size, hidden_size)
        # buffer in upstream -- registered with nn.Parameter wrapper but as buffer.
        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels**0.5,
            persistent=True,
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 3:
            y = y.unsqueeze(1)
        return self.y_proj(y)


class T2IFinalLayer(nn.Module):
    def __init__(
        self, hidden_size: int, patch_size: Tuple[int, int, int], out_channels: int
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = _sana_wm_column_parallel_or_linear(
            hidden_size,
            math.prod(patch_size) * out_channels,
            bias=True,
            gather_output=True,
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
            x = self.norm_final(x) * (1 + scale) + shift
        else:
            B, N, D = x.shape
            t = t.reshape(B, -1, D)
            num_frames = t.shape[1]
            tokens_per_frame = N // num_frames
            shift, scale = (
                self.scale_shift_table[None, None, :, :] + t[:, :, None, :]
            ).chunk(2, dim=2)
            x = self.norm_final(x).reshape(B, num_frames, tokens_per_frame, D)
            x = x * (1 + scale) + shift
            x = x.reshape(B, N, D)
        return _sana_wm_linear(self.linear, x)


def _sinusoidal_timestep_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = _SanaWMLinearSequential(
            _sana_wm_column_parallel_or_linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
                gather_output=False,
            ),
            nn.SiLU(),
            _sana_wm_row_parallel_or_linear(
                hidden_size,
                hidden_size,
                bias=True,
                input_is_parallel=True,
            ),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = _sinusoidal_timestep_embedding(t, self.frequency_embedding_size)
        # use the dtype of the first linear's weight to match upstream
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


_INT32_SAFE_CONV_ELEMENTS = 1 << 30


class _ConvLayer(nn.Module):
    def __init__(self, conv: nn.Module, act: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv = conv
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GLUMBConvTemp(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int, t_kernel_size: int = 3
    ) -> None:
        super().__init__()
        inverted_conv = nn.Conv2d(in_features, hidden_features * 2, 1, 1, 0, bias=True)
        depth_conv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            3,
            1,
            1,
            groups=hidden_features * 2,
            bias=True,
        )
        point_conv = nn.Conv2d(hidden_features, in_features, 1, 1, 0, bias=False)
        t_conv = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=(t_kernel_size, 1),
            stride=1,
            padding=(t_kernel_size // 2, 0),
            bias=False,
        )
        self.inverted_conv = _ConvLayer(
            inverted_conv,
            act=nn.SiLU(inplace=False),
        )
        self.depth_conv = _ConvLayer(
            depth_conv,
            act=None,
        )
        self.point_conv = _ConvLayer(
            point_conv,
            act=None,
        )
        self.glu_act = nn.SiLU(inplace=False)

        self.t_conv = t_conv
        nn.init.zeros_(self.t_conv.weight)

    def _apply_spatial(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        a, g = x.chunk(2, dim=1)
        return self.point_conv(a * self.glu_act(g))

    def _apply_spatial_autochunked(self, x: torch.Tensor) -> torch.Tensor:
        """Avoid oversized Conv2d calls on long videos while keeping short path fused."""
        BT, _, H, W = x.shape
        elements_per_bt = self.inverted_conv.conv.out_channels * H * W
        max_bt = max(1, _INT32_SAFE_CONV_ELEMENTS // elements_per_bt)
        if BT <= max_bt:
            return self._apply_spatial(x)
        return torch.cat(
            [
                self._apply_spatial(x[start : start + max_bt])
                for start in range(0, BT, max_bt)
            ],
            dim=0,
        )

    def forward(self, x: torch.Tensor, HW: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = HW
        assert N == T * H * W, f"GLUMBConvTemp: N={N} != T*H*W={T * H * W}"

        # Spatial path -- (B*T, C, H, W)
        memory_format = (
            torch.channels_last
            if x.device.type == "cuda" and x.dtype in (torch.float16, torch.bfloat16)
            else torch.contiguous_format
        )
        x_sp = (
            x.reshape(B * T, H, W, C)
            .permute(0, 3, 1, 2)
            .contiguous(memory_format=memory_format)
        )
        x_sp = self._apply_spatial_autochunked(x_sp)  # (B*T, C, H, W)

        # Temporal additive path -- (B, C, T, S=H*W)
        x_sp = x_sp.contiguous()
        x_t = x_sp.view(B, T, C, H * W).permute(0, 2, 1, 3).contiguous()
        x_out = x_t + self.t_conv(x_t)

        # back to (B, N, C)
        return x_out.permute(0, 2, 3, 1).reshape(B, N, C)


def _compute_frame_gates(
    x: torch.Tensor,  # (B, N, C)
    HW: Tuple[int, int, int],
    heads: int,
    beta_proj: nn.Module,
    gate_proj: nn.Module,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, C = x.shape
    T, H, W = HW
    S = H * W
    beta = (
        _sana_wm_linear(beta_proj, x)
        .sigmoid()
        .reshape(B, T, S, heads)
        .permute(0, 3, 1, 2)
    )
    x_frame = x.reshape(B, T, S, C).mean(dim=2)
    a_out = _sana_wm_linear(gate_proj, x_frame).float()
    dt = dt_bias.float().view(1, 1, -1)
    A_val = A_log.float().exp().view(1, 1, -1)
    decay = (-A_val * F.softplus(a_out + dt)).exp().transpose(1, 2)  # (B, H, T)
    return beta, decay


class BidirectionalGDNUCPESinglePathLiteLA(nn.Module):
    def __init__(
        self,
        in_dim: int,
        heads: int,
        head_dim: int,
        qk_norm: bool = True,
        conv_kernel_size: int = 4,
        k_conv_only: bool = True,
        eps: float = 1e-8,
        softmax_main: bool = False,
        pad_attention_head_dim_to_flash: bool = False,
    ) -> None:
        super().__init__()
        out_dim = heads * head_dim
        assert (
            out_dim == in_dim
        ), f"in_dim ({in_dim}) must equal heads*head_dim ({out_dim})"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = head_dim
        self.tp_size = _sana_wm_tp_world_size()
        if self.tp_size < 1:
            raise ValueError(f"Invalid SANA-WM tp_size={self.tp_size}.")
        if heads % self.tp_size != 0:
            raise ValueError(
                "SANA-WM tensor parallelism requires num_attention_heads "
                f"({heads}) to be divisible by tp_size ({self.tp_size})."
            )
        self.tp_rank = _sana_wm_tp_rank()
        self.local_heads = heads // self.tp_size
        self.local_out_dim = self.local_heads * head_dim
        self.use_tp = self.tp_size > 1
        self.eps = eps
        self.softmax_main = softmax_main

        # Main branch: fused QKV + output proj (shared with cam branch)
        if self.use_tp:
            self.qkv = MergedColumnParallelLinear(
                in_dim,
                [out_dim, out_dim, out_dim],
                bias=False,
                gather_output=False,
            )
        else:
            self.qkv = nn.Linear(in_dim, 3 * out_dim, bias=False)
        if self.use_tp:
            self.proj = RowParallelLinear(
                out_dim, out_dim, bias=True, input_is_parallel=True
            )
        else:
            self.proj = nn.Linear(out_dim, out_dim, bias=True)

        if qk_norm:
            self.q_norm = _RMSNorm(in_dim, eps=1e-5)
            self.k_norm = _RMSNorm(in_dim, eps=1e-5)
            self.q_norm_cam = _RMSNorm(in_dim, eps=1e-5)
            self.k_norm_cam = _RMSNorm(in_dim, eps=1e-5)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_cam = nn.Identity()
            self.k_norm_cam = nn.Identity()

        # GDN-specific (also held by softmax variant for state_dict compat)
        if self.use_tp:
            self.beta_proj = ColumnParallelLinear(
                in_dim, heads, bias=True, gather_output=False
            )
            self.gate_proj = ColumnParallelLinear(
                in_dim, heads, bias=True, gather_output=False
            )
        else:
            self.beta_proj = nn.Linear(in_dim, heads, bias=True)
            self.gate_proj = nn.Linear(in_dim, heads, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(0, 16)))
        self.dt_bias = nn.Parameter(torch.full((heads,), -5.0))
        self.register_buffer("recall_gate", torch.zeros(1))
        if self.use_tp:
            self.output_gate = ColumnParallelLinear(
                in_dim, out_dim, bias=True, gather_output=False
            )
        else:
            self.output_gate = nn.Linear(in_dim, out_dim, bias=True)

        conv_hidden_size = self.local_out_dim if self.use_tp else out_dim
        if conv_kernel_size > 0 and not softmax_main:
            self.conv_k = _ShortConvolution(conv_hidden_size, conv_kernel_size)
            if not k_conv_only:
                self.conv_q = _ShortConvolution(conv_hidden_size, conv_kernel_size)
                self.conv_v = _ShortConvolution(conv_hidden_size, conv_kernel_size)
            else:
                self.conv_q = None
                self.conv_v = None
        else:
            self.conv_k = self.conv_q = self.conv_v = None
        self.conv_kernel_size = conv_kernel_size
        self.k_conv_only = k_conv_only

        if self.use_tp:
            self.q_proj_cam = ColumnParallelLinear(
                in_dim, out_dim, bias=True, gather_output=False
            )
            self.k_proj_cam = ColumnParallelLinear(
                in_dim, out_dim, bias=True, gather_output=False
            )
            self.v_proj_cam = ColumnParallelLinear(
                in_dim, out_dim, bias=True, gather_output=False
            )
        else:
            self.q_proj_cam = nn.Linear(in_dim, out_dim, bias=True)
            self.k_proj_cam = nn.Linear(in_dim, out_dim, bias=True)
            self.v_proj_cam = nn.Linear(in_dim, out_dim, bias=True)
        if self.use_tp:
            self.out_proj_cam = ColumnParallelLinear(
                out_dim, out_dim, bias=True, gather_output=False
            )
        else:
            self.out_proj_cam = nn.Linear(out_dim, out_dim, bias=True)
        nn.init.zeros_(self.out_proj_cam.weight)
        nn.init.zeros_(self.out_proj_cam.bias)
        if conv_kernel_size > 0 and not softmax_main:
            self.conv_k_cam = _ShortConvolution(conv_hidden_size, conv_kernel_size)
            if not k_conv_only:
                self.conv_q_cam = _ShortConvolution(conv_hidden_size, conv_kernel_size)
                self.conv_v_cam = _ShortConvolution(conv_hidden_size, conv_kernel_size)
            else:
                self.conv_q_cam = None
                self.conv_v_cam = None
        else:
            self.conv_k_cam = self.conv_q_cam = self.conv_v_cam = None

        if softmax_main:
            self.softmax_attn = _make_sana_wm_local_attention(
                num_heads=self.local_heads,
                head_size=head_dim,
                pad_head_dim_to_flash=pad_attention_head_dim_to_flash,
            )

    @staticmethod
    def _temporal_short_conv(
        x: torch.Tensor,  # (B, N, C)
        conv: _ShortConvolution,
        HW: Tuple[int, int, int],
        bidirectional: bool = True,
    ) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        # Move T into the time axis: (B, T, S, C) -> (B*S, T, C)
        y = x.view(B, T, S, C).permute(0, 2, 1, 3).contiguous().reshape(B * S, T, C)
        if bidirectional:
            y = _bidirectional_short_conv(y, conv)
        else:
            y, _ = conv(y)
        # back to (B, T*S, C)
        return y.reshape(B, S, T, C).permute(0, 2, 1, 3).reshape(B, N, C)

    def _cam_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            _sana_wm_linear(self.q_proj_cam, x),
            _sana_wm_linear(self.k_proj_cam, x),
            _sana_wm_linear(self.v_proj_cam, x),
        )

    def _raise_missing_triton_fast_path(
        self,
        kernel_name: str,
    ) -> NoReturn:
        raise RuntimeError(
            f"SANA-WM {kernel_name} did not run. "
            "SANA-WM native runtime requires this Triton fast path; "
            "fix the kernel support/guards instead of falling back to torch."
        )

    def _try_triton_main_gdn(
        self,
        qkv: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        *,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        k_scale: float,
    ) -> Optional[torch.Tensor]:
        if (
            torch.is_grad_enabled()
            or _sana_wm_deterministic_inference_enabled()
            or not isinstance(self.q_norm, _RMSNorm)
            or not isinstance(self.k_norm, _RMSNorm)
        ):
            return None
        if not qkv.is_cuda:
            return None

        _, N, _, Hh, D = qkv.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        if Hh != self.local_heads or D != self.dim or N != T * S:
            return None

        beta_f = beta.float().contiguous()
        decay_f = decay.float().contiguous()
        kernels = _get_sana_wm_triton_main_gdn()

        fused_bigdn, can_use, fused_bigdn_with_inv_rms, can_use_with_inv_rms = kernels

        q_weight = _sana_wm_local_rms_norm_weight(
            self.q_norm,
            local_hidden_size=Hh * D,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        k_weight = _sana_wm_local_rms_norm_weight(
            self.k_norm,
            local_hidden_size=Hh * D,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        if q_weight is None or k_weight is None:
            return None

        if self.use_tp:
            q_inv_rms, k_inv_rms = _sana_wm_tp_qk_inv_rms(
                qkv,
                norm_eps=self.q_norm.eps,
                tp_size=self.tp_size,
            )
            if not can_use_with_inv_rms(
                qkv,
                q_inv_rms,
                k_inv_rms,
                q_weight,
                k_weight,
                rotary_emb,
                beta_f,
                decay_f,
                F=T,
                S=S,
            ):
                return None
            return fused_bigdn_with_inv_rms(
                qkv,
                q_inv_rms,
                k_inv_rms,
                q_weight,
                k_weight,
                rotary_emb,
                beta_f,
                decay_f,
                F=T,
                S=S,
                k_scale=k_scale,
                eps=self.eps,
                norm_eps=self.q_norm.eps,
            )

        if not can_use(
            qkv,
            q_weight,
            k_weight,
            rotary_emb,
            beta_f,
            decay_f,
            F=T,
            S=S,
        ):
            return None
        return fused_bigdn(
            qkv,
            q_weight,
            k_weight,
            rotary_emb,
            beta_f,
            decay_f,
            F=T,
            S=S,
            k_scale=k_scale,
            eps=self.eps,
            norm_eps=self.q_norm.eps,
        )

    def _try_triton_cam_gdn_preprocess(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        raymats_flat: Optional[torch.Tensor],
        raymats_t: Optional[torch.Tensor],
        raymats_inv: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if (
            torch.is_grad_enabled()
            or _sana_wm_deterministic_inference_enabled()
            or raymats_flat is None
            or not isinstance(self.q_norm_cam, _RMSNorm)
            or not isinstance(self.k_norm_cam, _RMSNorm)
        ):
            return None
        if not q.is_cuda:
            return None

        B, N, Hh, D = q.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        if Hh != self.local_heads or D != self.dim or N != T * S:
            return None
        if tuple(raymats_flat.shape) != (B, N, 4, 4):
            return None

        kernels = _get_sana_wm_triton_cam_preprocess()

        (
            preprocess,
            can_use,
            _,
            _,
            preprocess_with_inv_rms,
            can_use_with_inv_rms,
            *_,
        ) = kernels
        P = raymats_flat if raymats_flat.is_contiguous() else raymats_flat.contiguous()
        P_T = (
            raymats_t
            if raymats_t is not None and raymats_t.is_contiguous()
            else P.transpose(-1, -2).contiguous()
        )
        P_inv = (
            raymats_inv
            if raymats_inv is not None and raymats_inv.is_contiguous()
            else _invert_SE3(P).contiguous()
        )
        rotary_emb_cam = _slice_rope_for_cam(rotary_emb, self.dim, self.dim // 2)

        q_weight = _sana_wm_local_rms_norm_weight(
            self.q_norm_cam,
            local_hidden_size=Hh * D,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        k_weight = _sana_wm_local_rms_norm_weight(
            self.k_norm_cam,
            local_hidden_size=Hh * D,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        if q_weight is None or k_weight is None:
            return None

        if self.use_tp:
            q_inv_rms, k_inv_rms = _sana_wm_tp_pair_inv_rms(
                q,
                k,
                norm_eps=self.q_norm_cam.eps,
                tp_size=self.tp_size,
            )
            if not can_use_with_inv_rms(
                q,
                k,
                v,
                q_inv_rms,
                k_inv_rms,
                q_weight,
                k_weight,
                P_T,
                P_inv,
                rotary_emb_cam,
            ):
                return None
            return preprocess_with_inv_rms(
                q,
                k,
                v,
                q_inv_rms,
                k_inv_rms,
                q_weight,
                k_weight,
                P_T,
                P_inv,
                rotary_emb_cam,
                k_scale=(self.dim**-0.5) * (S**-0.5),
                eps=self.q_norm_cam.eps,
            )

        if not can_use(
            q,
            k,
            v,
            q_weight,
            k_weight,
            P_T,
            P_inv,
            rotary_emb_cam,
        ):
            return None

        return preprocess(
            q,
            k,
            v,
            q_weight,
            k_weight,
            P_T,
            P_inv,
            rotary_emb_cam,
            k_scale=(self.dim**-0.5) * (S**-0.5),
            eps=self.q_norm_cam.eps,
        )

    def _try_triton_cam_output_apply_o(
        self,
        x: torch.Tensor,
        *,
        rotary_emb: Optional[torch.Tensor],
        raymats_flat: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            torch.is_grad_enabled()
            or _sana_wm_deterministic_inference_enabled()
            or raymats_flat is None
        ):
            return None
        if not x.is_cuda:
            return None

        B, Hh, N, D = x.shape
        if Hh != self.local_heads or D != self.dim:
            return None
        if tuple(raymats_flat.shape) != (B, N, 4, 4):
            return None

        kernels = _get_sana_wm_triton_cam_preprocess()

        _, _, output_apply_o, can_use_output_apply_o, *_ = kernels
        P = raymats_flat if raymats_flat.is_contiguous() else raymats_flat.contiguous()
        rotary_emb_cam = _slice_rope_for_cam(rotary_emb, self.dim, self.dim // 2)
        if not can_use_output_apply_o(x, P, rotary_emb_cam):
            return None

        return output_apply_o(x, P, rotary_emb_cam)

    def _try_triton_cam_softmax_preprocess(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        raymats_flat: Optional[torch.Tensor],
        raymats_t: Optional[torch.Tensor],
        raymats_inv: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if (
            torch.is_grad_enabled()
            or _sana_wm_deterministic_inference_enabled()
            or raymats_flat is None
            or not isinstance(self.q_norm_cam, _RMSNorm)
            or not isinstance(self.k_norm_cam, _RMSNorm)
        ):
            return None
        if not q.is_cuda:
            return None

        B, N, Hh, D = q.shape
        T, H_sp, W_sp = HW
        if Hh != self.local_heads or D != self.dim or N != T * H_sp * W_sp:
            return None
        if tuple(raymats_flat.shape) != (B, N, 4, 4):
            return None

        kernels = _get_sana_wm_triton_cam_preprocess()

        softmax_preprocess_with_inv_rms = kernels[6]
        can_use_softmax_with_inv_rms = kernels[7]
        P = raymats_flat if raymats_flat.is_contiguous() else raymats_flat.contiguous()
        P_T = (
            raymats_t
            if raymats_t is not None and raymats_t.is_contiguous()
            else P.transpose(-1, -2).contiguous()
        )
        P_inv = (
            raymats_inv
            if raymats_inv is not None and raymats_inv.is_contiguous()
            else _invert_SE3(P).contiguous()
        )
        rotary_emb_cam = _slice_rope_for_cam(rotary_emb, self.dim, self.dim // 2)

        q_weight = _sana_wm_local_rms_norm_weight(
            self.q_norm_cam,
            local_hidden_size=Hh * D,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        k_weight = _sana_wm_local_rms_norm_weight(
            self.k_norm_cam,
            local_hidden_size=Hh * D,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        )
        if q_weight is None or k_weight is None:
            return None

        q_inv_rms, k_inv_rms = _sana_wm_tp_pair_inv_rms(
            q,
            k,
            norm_eps=self.q_norm_cam.eps,
            tp_size=self.tp_size,
        )
        if not can_use_softmax_with_inv_rms(
            q,
            k,
            v,
            q_inv_rms,
            k_inv_rms,
            q_weight,
            k_weight,
            P_T,
            P_inv,
            rotary_emb_cam,
        ):
            return None
        return softmax_preprocess_with_inv_rms(
            q,
            k,
            v,
            q_inv_rms,
            k_inv_rms,
            q_weight,
            k_weight,
            P_T,
            P_inv,
            rotary_emb_cam,
            downscale_eps=1e-6,
        )

    def _try_triton_cam_scan_bidirectional(
        self,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        *,
        HW: Tuple[int, int, int],
    ) -> Optional[torch.Tensor]:
        if torch.is_grad_enabled() or _sana_wm_deterministic_inference_enabled():
            return None
        if not q_rot.is_cuda:
            return None

        B, Hh, D, N = q_rot.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        if Hh != self.local_heads or D != self.dim or N != T * S:
            return None

        kernels = _get_sana_wm_triton_cam_scan()

        scan_bidi, can_use = kernels
        if not can_use(q_rot, k_rot, v, beta, decay):
            return None

        return scan_bidi(
            q_rot.contiguous(),
            k_rot.contiguous(),
            v.contiguous(),
            beta.contiguous(),
            decay.contiguous(),
        )

    def _main_branch_gdn(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        local_C = self.local_out_dim

        qkv = (
            _sana_wm_linear(self.qkv, x)
            .reshape(B, N, 3, self.local_heads, self.dim)
            .contiguous()
        )
        q, k, v = qkv.unbind(2)

        # Short conv on K only.
        if self.conv_k is not None:
            k = self._temporal_short_conv(
                k.reshape(B, N, local_C), self.conv_k, HW, bidirectional=True
            )
            k = k.reshape(B, N, self.local_heads, self.dim)
            qkv = torch.stack((q, k, v), dim=2).contiguous()

        gate_heads = self.local_heads if self.use_tp else self.heads
        dt_bias = self.dt_bias
        A_log = self.A_log
        if self.use_tp:
            head_start = self.tp_rank * self.local_heads
            dt_bias = dt_bias.narrow(0, head_start, self.local_heads)
            A_log = A_log.narrow(0, head_start, self.local_heads)

        beta, decay = _compute_frame_gates(
            x,
            HW,
            gate_heads,
            self.beta_proj,
            self.gate_proj,
            dt_bias,
            A_log,
        )

        k_scale = (self.dim**-0.5) * (S**-0.5)
        triton_out = self._try_triton_main_gdn(
            qkv,
            beta,
            decay,
            HW=HW,
            rotary_emb=rotary_emb,
            k_scale=k_scale,
        )
        if triton_out is not None:
            out = triton_out.reshape(B, N, local_C)
            return out, beta, decay
        self._raise_missing_triton_fast_path("main GDN Triton kernel")

    def _main_branch_softmax(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        chunk_size: Optional[int] = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: Optional[List[int]] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        local_C = self.local_out_dim
        qkv = _sana_wm_linear(self.qkv, x).reshape(B, N, 3, self.local_heads, self.dim)
        q, k, v = qkv.unbind(2)
        if self.conv_k is not None:
            k = self._temporal_short_conv(
                k.reshape(B, N, local_C), self.conv_k, HW, bidirectional=True
            )
            k = k.reshape(B, N, self.local_heads, self.dim)
        q_flat, k_flat = _sana_wm_tp_qk_rms_norm(
            q.reshape(B, N, local_C),
            k.reshape(B, N, local_C),
            self.q_norm,
            self.k_norm,
            tp_size=self.tp_size,
        )
        q = q_flat.reshape(B, N, self.local_heads, self.dim)
        k = k_flat.reshape(B, N, self.local_heads, self.dim)
        # RoPE primitives are written for (B, H, N, D); LocalAttention takes
        # (B, N, H, D). Permute for RoPE, then transpose back at the call site.
        q = q.permute(0, 2, 1, 3)  # (B, H, N, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if rotary_emb is not None:
            q = _apply_rotary_emb_bhnd(q, rotary_emb)
            k = _apply_rotary_emb_bhnd(k, rotary_emb)
        q_in = q.transpose(1, 2).contiguous()
        k_in = k.transpose(1, 2).contiguous()
        v_in = v.transpose(1, 2).contiguous()
        out = self.softmax_attn(q_in, k_in, v_in)
        out = out.reshape(B, N, local_C)
        return out

    def _cam_branch(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        beta: torch.Tensor,
        decay: torch.Tensor,
        rotary_emb: Optional[torch.Tensor] = None,
        raymats_flat: Optional[torch.Tensor] = None,
        raymats_t: Optional[torch.Tensor] = None,
        raymats_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        local_C = self.local_out_dim

        q, k, v = self._cam_qkv(x)
        q = q.reshape(B, N, self.local_heads, self.dim)
        k = k.reshape(B, N, self.local_heads, self.dim)
        v = v.reshape(B, N, self.local_heads, self.dim)

        # Short conv on K only (k_conv_only)
        if self.conv_k_cam is not None:
            k = self._temporal_short_conv(
                k.reshape(B, N, local_C), self.conv_k_cam, HW, bidirectional=True
            )
            k = k.reshape(B, N, self.local_heads, self.dim)

        (
            raymats_flat_triton,
            raymats_t_triton,
            raymats_inv_triton,
        ) = _sana_wm_materialize_repeated_raymats(
            raymats_flat,
            raymats_t,
            raymats_inv,
            B,
            name="camera",
        )
        cam_preprocessed = self._try_triton_cam_gdn_preprocess(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            HW=HW,
            rotary_emb=rotary_emb,
            raymats_flat=raymats_flat_triton,
            raymats_t=raymats_t_triton,
            raymats_inv=raymats_inv_triton,
        )
        if cam_preprocessed is None:
            self._raise_missing_triton_fast_path("camera preprocess Triton kernel")

        q_dn, k_dn, v_dn, inflation_sq = cam_preprocessed
        frame_inflation_sq = inflation_sq.view(B, self.local_heads, T, S).mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        dtype = q_dn.dtype
        q_scan = q_dn.float()
        k_scan = k_dn.float()
        v_scan = v_dn.float()
        out = self._try_triton_cam_scan_bidirectional(
            q_scan,
            k_scan,
            v_scan,
            beta.float(),
            decay.float(),
            HW=HW,
        )
        if out is None:
            self._raise_missing_triton_fast_path("camera scan Triton kernel")
        out_bhnd = out.to(dtype).permute(0, 1, 3, 2)
        triton_out_bhnd = self._try_triton_cam_output_apply_o(
            out_bhnd,
            rotary_emb=rotary_emb,
            raymats_flat=raymats_flat_triton,
        )
        if triton_out_bhnd is None:
            self._raise_missing_triton_fast_path("camera output Triton kernel")
        out_bhnd = triton_out_bhnd
        out = out_bhnd.permute(0, 2, 1, 3).reshape(B, N, local_C)
        return out

    def _cam_branch_softmax(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        chunk_size: Optional[int] = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: Optional[List[int]] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        raymats_flat: Optional[torch.Tensor] = None,
        raymats_t: Optional[torch.Tensor] = None,
        raymats_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        local_C = self.local_out_dim

        q, k, v = self._cam_qkv(x)
        q = q.reshape(B, N, self.local_heads, self.dim)
        k = k.reshape(B, N, self.local_heads, self.dim)
        v = v.reshape(B, N, self.local_heads, self.dim)

        if self.conv_k_cam is not None:
            k = self._temporal_short_conv(
                k.reshape(B, N, local_C), self.conv_k_cam, HW, bidirectional=True
            )
            k = k.reshape(B, N, self.local_heads, self.dim)

        (
            raymats_flat_triton,
            raymats_t_triton,
            raymats_inv_triton,
        ) = _sana_wm_materialize_repeated_raymats(
            raymats_flat,
            raymats_t,
            raymats_inv,
            B,
            name="softmax camera",
        )
        cam_preprocessed = self._try_triton_cam_softmax_preprocess(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            HW=HW,
            rotary_emb=rotary_emb,
            raymats_flat=raymats_flat_triton,
            raymats_t=raymats_t_triton,
            raymats_inv=raymats_inv_triton,
        )
        if cam_preprocessed is None:
            self._raise_missing_triton_fast_path(
                "softmax camera preprocess Triton kernel"
            )
        q_dn, k_dn, v_dn = cam_preprocessed

        q_in = q_dn.permute(0, 3, 1, 2).contiguous()
        k_in = k_dn.permute(0, 3, 1, 2).contiguous()
        v_in = v_dn.permute(0, 3, 1, 2).contiguous()
        out = self.softmax_attn(q_in, k_in, v_in)  # (B, N, H, D)

        out_bhnd = out.transpose(1, 2).contiguous()
        triton_out_bhnd = self._try_triton_cam_output_apply_o(
            out_bhnd,
            rotary_emb=rotary_emb,
            raymats_flat=raymats_flat_triton,
        )
        if triton_out_bhnd is None:
            self._raise_missing_triton_fast_path("camera output Triton kernel")
        out_bhnd = triton_out_bhnd
        out = out_bhnd.transpose(1, 2).reshape(B, N, local_C)
        return out

    # Keep this Triton-heavy attention branch eager when regional torch.compile
    # captures the surrounding block. The GDN/UCPE kernels build launch grids from
    # concrete runtime shapes; symbolic Dynamo shapes can break those launches.
    @_sana_wm_torch_compile_disable
    def forward(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor] = None,
        ucpe_raymat_bundle: Optional[Tuple] = None,
        chunk_size: Optional[int] = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if self.softmax_main:
            main_raw = self._main_branch_softmax(
                x,
                HW,
                rotary_emb,
                chunk_size=chunk_size,
                chunk_split_strategy=chunk_split_strategy,
                chunk_index=chunk_index,
            )
            beta = decay = None
        else:
            main_raw, beta, decay = self._main_branch_gdn(x, HW, rotary_emb)

        if ucpe_raymat_bundle is not None:
            raymats_flat, raymats_t, raymats_inv = ucpe_raymat_bundle[:3]
            raymat_cache = (
                ucpe_raymat_bundle[3] if len(ucpe_raymat_bundle) > 3 else None
            )
            if not isinstance(raymat_cache, dict):
                raymat_cache = None
            raymats_flat, raymats_t, raymats_inv = (
                _sana_wm_materialize_repeated_raymats(
                    raymats_flat,
                    raymats_t,
                    raymats_inv,
                    x.shape[0],
                    name="camera",
                    cache=raymat_cache,
                )
            )
            if self.softmax_main:
                cam_raw = self._cam_branch_softmax(
                    x,
                    HW,
                    chunk_size=chunk_size,
                    chunk_split_strategy=chunk_split_strategy,
                    chunk_index=chunk_index,
                    rotary_emb=rotary_emb,
                    raymats_flat=raymats_flat,
                    raymats_t=raymats_t,
                    raymats_inv=raymats_inv,
                )
            else:
                assert beta is not None and decay is not None
                cam_raw = self._cam_branch(
                    x,
                    HW,
                    beta,
                    decay,
                    rotary_emb=rotary_emb,
                    raymats_flat=raymats_flat,
                    raymats_t=raymats_t,
                    raymats_inv=raymats_inv,
                )
            if self.use_tp:
                cam_raw = _sana_wm_all_gather_hidden(cam_raw, self.tp_size)
            cam_contrib = _sana_wm_linear(self.out_proj_cam, cam_raw)
            combined = main_raw + cam_contrib
        else:
            combined = main_raw

        # Shared output gate + shared output projection. Upstream evaluates
        # the SiLU gate in fp32 and multiplies before casting for proj.
        gate = F.silu(_sana_wm_linear(self.output_gate, x).to(torch.float32))
        combined = combined * gate
        out = _sana_wm_linear(self.proj, combined.to(self.proj.weight.dtype))
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        qk_norm: bool = True,
        pad_attention_head_dim_to_flash: bool = False,
        request_runtime_cache: bool = True,
        cross_attn_kv_cache_max_bytes: int = (
            _SANA_WM_CROSS_ATTN_KV_CACHE_DEFAULT_MAX_BYTES
        ),
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.tp_size = _sana_wm_tp_world_size()
        if self.tp_size < 1:
            raise ValueError(f"Invalid SANA-WM tp_size={self.tp_size}.")
        if num_heads % self.tp_size != 0:
            raise ValueError(
                "SANA-WM cross-attention tensor parallelism requires "
                f"num_heads ({num_heads}) to be divisible by tp_size ({self.tp_size})."
            )
        self.tp_rank = _sana_wm_tp_rank()
        self.local_num_heads = num_heads // self.tp_size
        self.local_d_model = self.local_num_heads * self.head_dim
        self.use_tp = self.tp_size > 1
        self.request_runtime_cache = bool(request_runtime_cache)
        self.cross_attn_kv_cache_max_bytes = int(cross_attn_kv_cache_max_bytes)

        if self.use_tp:
            self.q_linear = ColumnParallelLinear(
                d_model, d_model, bias=True, gather_output=False
            )
            self.kv_linear = MergedColumnParallelLinear(
                d_model, [d_model, d_model], bias=True, gather_output=False
            )
        else:
            self.q_linear = nn.Linear(d_model, d_model, bias=True)
            self.kv_linear = nn.Linear(d_model, d_model * 2, bias=True)
        if self.use_tp:
            self.proj = RowParallelLinear(
                d_model, d_model, bias=True, input_is_parallel=True
            )
        else:
            self.proj = nn.Linear(d_model, d_model, bias=True)
        if qk_norm:
            self.q_norm = _RMSNorm(d_model, eps=1e-6)
            self.k_norm = _RMSNorm(d_model, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        # Cross-attention dispatched through SGLang's pluggable backend.
        # The padding-mask path falls back to SDPA internally; the unmasked
        # path can pick FA3 / FlashInfer / etc.
        self.attn = _make_sana_wm_local_attention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            pad_head_dim_to_flash=pad_attention_head_dim_to_flash,
        )
        self.request_cache_name: Optional[str] = None

    def _get_cross_attention_kv(
        self,
        cond: torch.Tensor,
        *,
        batch_size: int,
        forward_batch: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_enabled = _sana_wm_request_runtime_cache_enabled(
            self.request_runtime_cache
        )
        key = None
        request_cache = None
        if cache_enabled:
            request_cache = SanaWMTransformer3DModel._get_request_cache(
                forward_batch,
                self.request_cache_name or f"sana_wm_cross_attn_kv_{id(self)}",
            )
            if request_cache is not None:
                key = (
                    "cross_attn_kv",
                    self.local_d_model,
                    self.local_num_heads,
                    self.head_dim,
                    self.tp_size,
                    _tensor_cache_key(cond),
                    id(self.kv_linear),
                    id(self.k_norm),
                )
                cached = request_cache.get("entry")
                if cached is not None and cached[0] == key:
                    return cached[2], cached[3]
                request_cache.clear()

        kv = _sana_wm_linear(self.kv_linear, cond).view(
            batch_size, -1, 2, self.local_d_model
        )
        k, v = kv.unbind(2)
        k = _sana_wm_tp_rms_norm(
            k,
            self.k_norm,
            tp_size=self.tp_size,
        ).view(batch_size, -1, self.local_num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.local_num_heads, self.head_dim)

        if cache_enabled:
            cache_bytes = k.numel() * k.element_size() + v.numel() * v.element_size()
            max_bytes = self.cross_attn_kv_cache_max_bytes
            cache_fits = max_bytes < 0 or cache_bytes <= max_bytes
            if request_cache is not None and cache_fits:
                k = k.detach()
                v = v.detach()
                request_cache["entry"] = (key, cond, k, v)
        return k, v

    def forward(
        self,
        x: torch.Tensor,  # (B, N, D)
        cond: torch.Tensor,  # (B, L, D)
        mask: Optional[torch.Tensor] = None,
        forward_batch: Any = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        q = _sana_wm_linear(self.q_linear, x)
        # LocalAttention takes (B, N, H, D); skip the legacy BHND transpose.
        q = _sana_wm_tp_rms_norm(
            q,
            self.q_norm,
            tp_size=self.tp_size,
        ).view(B, N, self.local_num_heads, self.head_dim)
        k, v = self._get_cross_attention_kv(
            cond, batch_size=B, forward_batch=forward_batch
        )

        attn_mask = mask.bool() if mask is not None else None
        out = self.attn(q, k, v, attn_mask=attn_mask)  # (B, N, H, D)
        out = out.reshape(B, N, self.local_d_model)
        return _sana_wm_linear(self.proj, out)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class SanaWMBlock(nn.Module):
    """One transformer block of SANA-WM."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        t_kernel_size: int,
        qk_norm: bool,
        cross_norm: bool,
        conv_kernel_size: int,
        k_conv_only: bool,
        softmax_main: bool,
        use_chunk_plucker_post_attn: bool,
        chunk_size: Optional[int] = None,
        chunk_split_strategy: str = "uniform",
        pad_attention_head_dim_to_flash: bool = False,
        request_runtime_cache: bool = True,
        cross_attn_kv_cache_max_bytes: int = (
            _SANA_WM_CROSS_ATTN_KV_CACHE_DEFAULT_MAX_BYTES
        ),
    ) -> None:
        super().__init__()
        self.softmax_main = softmax_main
        self.chunk_size = chunk_size
        self.chunk_split_strategy = chunk_split_strategy

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=hidden_size,
            heads=num_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            conv_kernel_size=conv_kernel_size,
            k_conv_only=k_conv_only,
            softmax_main=softmax_main,
            pad_attention_head_dim_to_flash=pad_attention_head_dim_to_flash,
        )

        self.cross_attn = MultiHeadCrossAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            qk_norm=cross_norm,
            pad_attention_head_dim_to_flash=pad_attention_head_dim_to_flash,
            request_runtime_cache=request_runtime_cache,
            cross_attn_kv_cache_max_bytes=cross_attn_kv_cache_max_bytes,
        )

        self.mlp = GLUMBConvTemp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            t_kernel_size=t_kernel_size,
        )

        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )

        if use_chunk_plucker_post_attn:
            self.plucker_proj = _sana_wm_column_parallel_or_linear(
                hidden_size,
                hidden_size,
                bias=True,
                gather_output=True,
            )
            nn.init.zeros_(self.plucker_proj.weight)
            nn.init.zeros_(self.plucker_proj.bias)
        else:
            self.plucker_proj = None

    @staticmethod
    def _modulate(
        x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        return x * (1 + scale) + shift

    @staticmethod
    def _reshape_framewise_modulation(
        x: torch.Tensor,
        num_frames: int,
    ) -> tuple[torch.Tensor, int]:
        B, N, C = x.shape
        tokens_per_frame = N // num_frames
        return x.reshape(B, num_frames, tokens_per_frame, C), tokens_per_frame

    def _add_plucker_post_attn(
        self,
        x: torch.Tensor,
        plucker_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.plucker_proj is None:
            return x
        if plucker_emb.shape[0] == x.shape[0]:
            return x + _sana_wm_linear(self.plucker_proj, plucker_emb)

        projected = _sana_wm_linear(self.plucker_proj, plucker_emb)
        return _sana_wm_add_repeated_batch(
            x,
            projected,
            name="Plucker embedding",
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, N, D)
        y: torch.Tensor,  # (B, L, D) text embeds
        t: torch.Tensor,  # (B, 6*D) AdaLN-single
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        ucpe_raymat_bundle: Optional[Tuple],
        plucker_emb: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        chunk_size: Optional[int] = None,
        chunk_split_strategy: Optional[str] = None,
        chunk_index: Optional[List[int]] = None,
        forward_batch: Any = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        if t.dim() == 2:
            num_frames = None
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
        else:
            num_frames = t.reshape(B, -1, 6, t.shape[-1] // 6).shape[1]
            t = t.reshape(B, num_frames, 6, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None, None, :, :] + t
            ).chunk(6, dim=2)

        # Self-attention with UCPE camera branch
        if num_frames is None:
            x_in = self._modulate(self.norm1(x), shift_msa, scale_msa)
        else:
            x_norm, tokens_per_frame = self._reshape_framewise_modulation(
                self.norm1(x), num_frames
            )
            x_in = self._modulate(x_norm, shift_msa, scale_msa).reshape_as(x)
        attn_out = self.attn(
            x_in,
            HW=HW,
            rotary_emb=rotary_emb,
            ucpe_raymat_bundle=ucpe_raymat_bundle,
            chunk_size=self.chunk_size if chunk_size is None else chunk_size,
            chunk_split_strategy=(
                self.chunk_split_strategy
                if chunk_split_strategy is None
                else chunk_split_strategy
            ),
            chunk_index=chunk_index,
        )
        if num_frames is None:
            x = x + gate_msa * attn_out
        else:
            attn_out = attn_out.reshape(B, num_frames, tokens_per_frame, -1)
            x = x + (gate_msa * attn_out).reshape_as(x)

        # Plucker post-attn injection (zero-init linear)
        if self.plucker_proj is not None and plucker_emb is not None:
            x = self._add_plucker_post_attn(x, plucker_emb)

        # Cross-attention
        x = x + self.cross_attn(x, y, mask=mask, forward_batch=forward_batch)

        # FFN
        if num_frames is None:
            x_in = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = x + gate_mlp * self.mlp(x_in, HW=HW)
        else:
            x_norm, tokens_per_frame = self._reshape_framewise_modulation(
                self.norm2(x), num_frames
            )
            x_in = self._modulate(x_norm, shift_mlp, scale_mlp).reshape_as(x)
            mlp_out = self.mlp(x_in, HW=HW).reshape(B, num_frames, tokens_per_frame, -1)
            x = x + (gate_mlp * mlp_out).reshape_as(x)
        return x


class SanaWMTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _repeated_blocks = ["SanaWMBlock"]
    _fsdp_shard_conditions = SanaWMConfig()._fsdp_shard_conditions
    _compile_conditions = SanaWMConfig()._compile_conditions
    _supported_attention_backends = SanaWMConfig()._supported_attention_backends
    # Extension point for future true SP. SANA-WM needs layout-aware sharding for
    # non-attention operators (GDN scan, GLUMBConvTemp, camera UCPE, Plucker).
    # Existing USPAttention only covers softmax attention, so sequence sharding
    # stays disabled until those operators get explicit state/reduction/halo
    # exchange.
    supports_true_sequence_parallel = False
    param_names_mapping = SanaWMConfig().param_names_mapping
    reverse_param_names_mapping = SanaWMConfig().reverse_param_names_mapping
    lora_param_names_mapping: dict = {}

    @staticmethod
    def _get_request_cache(forward_batch: Any, name: str) -> dict | None:
        if forward_batch is None:
            return None
        session = getattr(forward_batch, "session", None)
        if session is not None:
            state = session.get_or_create_state(RealtimeCausalDiTState)
            runtime_cache = state.runtime_cache
        else:
            runtime_cache = getattr(forward_batch, "extra", None)
            if runtime_cache is None:
                return None

        sana_wm_cache = runtime_cache.setdefault(
            _SANA_WM_REQUEST_RUNTIME_CACHE_NAMESPACE, {}
        )
        return sana_wm_cache.setdefault(name, {})

    @staticmethod
    def _validate_tp_config(arch, tp_size: int) -> None:
        if tp_size < 1:
            raise ValueError(f"Invalid SANA-WM tp_size={tp_size}.")
        if tp_size == 1:
            return

        if arch.num_attention_heads % tp_size != 0:
            raise ValueError(
                "SANA-WM tensor parallelism requires num_attention_heads to be "
                f"divisible by tp_size, got num_attention_heads="
                f"{arch.num_attention_heads}, tp_size={tp_size}."
            )
        hidden_size = arch.num_attention_heads * arch.attention_head_dim
        if hidden_size % tp_size != 0:
            raise ValueError(
                "SANA-WM tensor parallelism requires hidden_size to be divisible "
                f"by tp_size, got hidden_size={hidden_size}, tp_size={tp_size}."
            )

    def __init__(self, config: SanaWMConfig, hf_config=None, **kwargs) -> None:
        super().__init__(config, hf_config=hf_config or {}, **kwargs)
        if hasattr(config, "apply_user_flags_to_arch_config"):
            config.apply_user_flags_to_arch_config()
        arch = config.arch_config
        self.tp_size = _sana_wm_tp_world_size()
        self._validate_tp_config(arch, self.tp_size)

        self.patch_size = (arch.patch_size_t, arch.patch_size, arch.patch_size)
        self.inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.hidden_size = self.inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.attention_head_dim = arch.attention_head_dim
        self.out_channels = arch.out_channels
        self.num_channels_latents = arch.num_channels_latents
        self.vae_temporal_stride = arch.vae_temporal_stride
        self.timestep_norm_scale_factor = getattr(
            arch, "timestep_norm_scale_factor", 1.0
        )

        self.x_embedder = PatchEmbedMS3D(
            self.patch_size,
            arch.in_channels,
            self.inner_dim,
            bias=True,
        )

        self.t_embedder = TimestepEmbedder(self.inner_dim, frequency_embedding_size=256)
        self.t_block = _SanaWMLinearSequential(
            nn.SiLU(),
            _sana_wm_column_parallel_or_linear(
                self.inner_dim,
                6 * self.inner_dim,
                bias=True,
                gather_output=True,
            ),
        )

        self.y_embedder = CaptionEmbedder(
            in_channels=arch.caption_channels,
            hidden_size=self.inner_dim,
            token_num=arch.model_max_length,
        )
        self.y_norm = bool(getattr(arch, "y_norm", True))
        self.attention_y_norm = _RMSNorm(
            self.inner_dim,
            scale_factor=getattr(arch, "y_norm_scale_factor", 1.0),
            eps=getattr(arch, "y_norm_eps", 1e-5),
        )

        self.raymap_embedder = PatchEmbedMS3D(
            self.patch_size,
            3,
            self.inner_dim,
            bias=True,
        )

        if arch.use_chunk_plucker_post_attn or arch.use_chunk_plucker_input:
            self.plucker_embedder = PatchEmbedMS3D(
                self.patch_size,
                arch.chunk_plucker_channels,
                self.inner_dim,
                bias=True,
            )
            nn.init.zeros_(self.plucker_embedder.proj.weight)
            nn.init.zeros_(self.plucker_embedder.proj.bias)
        else:
            self.plucker_embedder = None
        self.use_chunk_plucker_post_attn = arch.use_chunk_plucker_post_attn
        self.use_chunk_plucker_input = arch.use_chunk_plucker_input
        self.chunk_size = getattr(arch, "chunk_size", None)
        self.chunk_split_strategy = getattr(arch, "chunk_split_strategy", "uniform")
        self.pad_attention_head_dim_to_flash = bool(
            getattr(arch, "pad_attention_head_dim_to_flash", False)
        )
        self.request_runtime_cache = bool(getattr(arch, "request_runtime_cache", True))
        self.cross_attn_kv_cache_max_bytes = int(
            getattr(
                arch,
                "cross_attn_kv_cache_max_bytes",
                _SANA_WM_CROSS_ATTN_KV_CACHE_DEFAULT_MAX_BYTES,
            )
        )
        effective_softmax_head_dim = (
            _sana_wm_padded_attention_head_size(arch.linear_head_dim)
            if self.pad_attention_head_dim_to_flash
            else arch.linear_head_dim
        )

        # --- RoPE ---
        self.rope = WanRotaryPosEmbed(
            attention_head_dim=arch.linear_head_dim,
            patch_size=self.patch_size,
            max_seq_len=1024,
        )

        # --- Transformer blocks ---
        depth = arch.num_layers
        self.softmax_every_n = arch.softmax_every_n
        softmax_idx = set(
            i
            for i in range(depth)
            if arch.softmax_every_n > 0 and (i + 1) % arch.softmax_every_n == 0
        )
        self.softmax_block_indices = tuple(sorted(softmax_idx))
        logger.info(
            "SANA-WM attention config: pad_attention_head_dim_to_flash=%s, "
            "attention_head_dim=%d, "
            "effective_softmax_head_dim=%d, tp_size=%d, local_attention_heads=%d, "
            "softmax_blocks=%s, chunk_size=%s, chunk_split_strategy=%s, "
            "true_sp_supported=%s",
            self.pad_attention_head_dim_to_flash,
            arch.linear_head_dim,
            effective_softmax_head_dim,
            self.tp_size,
            arch.num_attention_heads // self.tp_size,
            self.softmax_block_indices,
            self.chunk_size,
            self.chunk_split_strategy,
            False,
        )

        self.blocks = nn.ModuleList(
            [
                SanaWMBlock(
                    hidden_size=self.inner_dim,
                    num_heads=arch.num_attention_heads,
                    head_dim=arch.linear_head_dim,
                    mlp_ratio=arch.mlp_ratio,
                    t_kernel_size=arch.t_kernel_size,
                    qk_norm=arch.qk_norm,
                    cross_norm=arch.cross_norm,
                    conv_kernel_size=arch.conv_kernel_size,
                    k_conv_only=arch.k_conv_only,
                    softmax_main=(i in softmax_idx),
                    use_chunk_plucker_post_attn=(
                        arch.use_chunk_plucker_post_attn
                        and (
                            arch.chunk_plucker_post_attn_blocks < 0
                            or i < arch.chunk_plucker_post_attn_blocks
                        )
                    ),
                    chunk_size=self.chunk_size,
                    chunk_split_strategy=self.chunk_split_strategy,
                    pad_attention_head_dim_to_flash=self.pad_attention_head_dim_to_flash,
                    request_runtime_cache=self.request_runtime_cache,
                    cross_attn_kv_cache_max_bytes=self.cross_attn_kv_cache_max_bytes,
                )
                for i in range(depth)
            ]
        )
        for block_index, block in enumerate(self.blocks):
            block.cross_attn.request_cache_name = (
                f"sana_wm_cross_attn_kv_{block_index}"
            )
        self.final_layer = T2IFinalLayer(
            self.inner_dim, self.patch_size, self.out_channels
        )

        # FSDP shard targets
        self.layer_names = ["blocks"]

    @property
    def transformer_blocks(self) -> nn.ModuleList:
        """Compatibility alias for Cache-DiT's Sana block adapter."""
        return self.blocks

    def get_cache_dit_block_adapter(self):
        from cache_dit import BlockAdapter, ForwardPattern

        blocks = getattr(self, "blocks", None)
        if blocks is None:
            return None

        # SANA-WM blocks use the native signature:
        #   forward(x, *, y, t, HW, rotary_emb, ...)
        # Pattern_3 matches the tensor flow, while disabling signature checks lets
        # cache-dit pass through SANA-WM's native kwargs unchanged.
        return BlockAdapter(
            transformer=self,
            blocks=blocks,
            blocks_name="blocks",
            forward_pattern=ForwardPattern.Pattern_3,
            check_forward_pattern=False,
        )

    def post_load_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, WanRotaryPosEmbed):
                if module._freqs.is_meta:
                    module._init_freqs_buffer()

    def _get_freqs(
        self,
        T: int,
        H: int,
        W: int,
        device: torch.device,
        forward_batch: Any = None,
    ) -> torch.Tensor:
        key = (T, H, W, str(device))
        request_cache = (
            self._get_request_cache(forward_batch, "sana_wm_freqs")
            if _sana_wm_request_runtime_cache_enabled(
                getattr(self, "request_runtime_cache", True)
            )
            else None
        )
        if request_cache is None:
            return self.rope((T, H, W), device)
        if key not in request_cache:
            request_cache[key] = self.rope((T, H, W), device)
        return request_cache[key]

    def _get_ucpe_raymat_bundle(
        self,
        camera_conditions: torch.Tensor,
        *,
        HW: Tuple[int, int, int],
        forward_batch: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        cache_enabled = _sana_wm_request_runtime_cache_enabled(
            getattr(self, "request_runtime_cache", True)
        )
        request_cache = None
        if not cache_enabled:
            raymats = process_camera_conditions_ucpe(
                camera_conditions,
                HW=HW,
                patch_size=self.patch_size,
            )
            raymats_flat = raymats.reshape(camera_conditions.shape[0], -1, 4, 4)
            return _build_ucpe_raymat_bundle(raymats_flat)

        key = (
            "ucpe",
            HW,
            self.patch_size,
            _tensor_cache_key(camera_conditions),
        )
        request_cache = self._get_request_cache(
            forward_batch, "sana_wm_ucpe_raymat_bundle"
        )
        if request_cache is not None:
            cached = request_cache.get("entry")
            if cached is not None and cached[0] == key:
                return cached[2]
            request_cache.clear()

        raymats = process_camera_conditions_ucpe(
            camera_conditions,
            HW=HW,
            patch_size=self.patch_size,
        )
        raymats_flat = raymats.reshape(camera_conditions.shape[0], -1, 4, 4)
        ucpe_raymat_bundle = _build_ucpe_raymat_bundle(raymats_flat)
        if request_cache is not None:
            request_cache["entry"] = (key, camera_conditions, ucpe_raymat_bundle)
        return ucpe_raymat_bundle

    def _get_plucker_emb(
        self,
        chunk_plucker: torch.Tensor,
        *,
        latent_token_count: int,
        forward_batch: Any = None,
    ) -> torch.Tensor:
        if self.plucker_embedder is None:
            raise ValueError("SANA-WM plucker_embedder is not initialized.")

        weight = self.plucker_embedder.proj.weight
        bias = self.plucker_embedder.proj.bias
        key = (
            "plucker_emb",
            latent_token_count,
            self.patch_size,
            _tensor_cache_key(chunk_plucker),
            _tensor_cache_key(weight),
            None if bias is None else _tensor_cache_key(bias),
        )
        cache_enabled = _sana_wm_request_runtime_cache_enabled(
            getattr(self, "request_runtime_cache", True)
        )
        request_cache = None
        if cache_enabled:
            request_cache = self._get_request_cache(
                forward_batch, "sana_wm_plucker_emb"
            )
            if request_cache is not None:
                cached = request_cache.get("entry")
                if cached is not None and cached[0] == key:
                    return cached[2]
                request_cache.clear()

        plucker_emb = self.plucker_embedder(chunk_plucker.to(weight.dtype))
        if plucker_emb.shape[1] != latent_token_count:
            raise ValueError(
                f"plucker_emb token count {plucker_emb.shape[1]} != "
                f"latent token count {latent_token_count}; "
                "expected chunk_plucker shape (B, 48, T, H, W)."
            )

        if request_cache is not None:
            request_cache["entry"] = (key, chunk_plucker, plucker_emb)
        return plucker_emb

    def _get_projected_y(
        self,
        encoder_hidden_states: torch.Tensor,
        *,
        batch_size: int,
        forward_batch: Any = None,
    ) -> torch.Tensor:
        cache_enabled = _sana_wm_request_runtime_cache_enabled(
            getattr(self, "request_runtime_cache", True)
        )
        key = None
        request_cache = None
        if cache_enabled:
            request_cache = self._get_request_cache(
                forward_batch, "sana_wm_y_projection"
            )
            if request_cache is not None:
                key = (
                    "y_projection",
                    batch_size,
                    bool(self.y_norm),
                    _tensor_cache_key(encoder_hidden_states),
                    id(self.y_embedder),
                    id(self.attention_y_norm) if self.y_norm else None,
                )
                cached = request_cache.get("entry")
                if cached is not None and cached[0] == key:
                    return cached[2]
                request_cache.clear()

        y = encoder_hidden_states
        if y.dim() == 3:
            y = y.unsqueeze(1)
        y = self.y_embedder(y).squeeze(1)  # (B, L, D)
        if y.shape[0] != batch_size:
            y = y.expand(batch_size, -1, -1).contiguous()
        if self.y_norm:
            y = self.attention_y_norm(y)

        if request_cache is not None:
            y = y.detach()
            request_cache["entry"] = (key, encoder_hidden_states, y)
        return y

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        camera_conditions: Optional[torch.Tensor] = None,
        chunk_plucker: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            raise ValueError("SANA-WM forward requires encoder_hidden_states.")
        if timestep is None:
            raise ValueError("SANA-WM forward requires timestep.")

        B, C, T_raw, H_raw, W_raw = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        T = T_raw // p_t
        H = H_raw // p_h
        W = W_raw // p_w
        chunk_size = kwargs.get("chunk_size", self.chunk_size)
        chunk_split_strategy = kwargs.get(
            "chunk_split_strategy", self.chunk_split_strategy
        )
        chunk_index = kwargs.get("chunk_index", None)
        forward_batch = kwargs.get("forward_batch", None)
        if forward_batch is None and _sana_wm_request_runtime_cache_enabled(
            getattr(self, "request_runtime_cache", True)
        ):
            ctx = _sana_wm_get_forward_context_or_none()
            forward_batch = None if ctx is None else ctx.forward_batch

        # --- 1. Patch embed: (B, C, T, H, W) -> (B, T*H*W, D) ---
        x = self.x_embedder(hidden_states.to(dtype=self.x_embedder.proj.weight.dtype))

        # --- 2. Timestep AdaLN-single ---
        # SANA-WM's LTX sampler passes per-frame timesteps shaped (B, 1, T)
        # so the clean first-frame condition can stay at timestep 0 while the
        # remaining latent frames denoise. Keep the scalar path for generic
        # scheduler compatibility.
        if self.timestep_norm_scale_factor != 1.0:
            timestep_for_embed = (
                timestep.float() / self.timestep_norm_scale_factor
            ).to(torch.float32)
        else:
            timestep_for_embed = timestep.long().to(torch.float32)

        if timestep_for_embed.dim() == 1:
            t_emb = self.t_embedder(timestep_for_embed)  # (B, D)
            t6 = self.t_block(t_emb)  # (B, 6D)
        else:
            timestep_shape = tuple(timestep_for_embed.shape)
            t_flat = self.t_embedder(timestep_for_embed.flatten())
            t6_flat = self.t_block(t_flat)
            t_emb = t_flat.unflatten(0, timestep_shape)
            t6 = t6_flat.unflatten(0, timestep_shape)

        # --- 3. Caption projection + y_norm ---
        if isinstance(encoder_attention_mask, (list, tuple)):
            encoder_attention_mask = encoder_attention_mask[0]
        y = self._get_projected_y(
            encoder_hidden_states, batch_size=B, forward_batch=forward_batch
        )
        if encoder_attention_mask is not None and encoder_attention_mask.shape[0] != B:
            encoder_attention_mask = encoder_attention_mask.expand(B, -1).contiguous()

        # --- 4. RoPE ---
        freqs = self._get_freqs(T, H, W, x.device, forward_batch=forward_batch)

        # --- 5. Camera conditioning: compute UCPE raymat bundle + Plucker ---
        ucpe_raymat_bundle = None
        if camera_conditions is not None:
            if camera_conditions.shape[1] != T:
                raise ValueError(
                    "SANA-WM camera_conditions must be sampled at latent "
                    f"frames: got {camera_conditions.shape[1]} frames, "
                    f"expected T={T}."
                )
            ucpe_raymat_bundle = self._get_ucpe_raymat_bundle(
                camera_conditions,
                HW=(T, H, W),
                forward_batch=forward_batch,
            )

        # Plucker post-attn embedding (shared across all blocks)
        plucker_emb = None
        needs_plucker_emb = (
            chunk_plucker is not None
            and self.plucker_embedder is not None
            and (self.use_chunk_plucker_post_attn or self.use_chunk_plucker_input)
        )
        if needs_plucker_emb:
            plucker_emb = self._get_plucker_emb(
                chunk_plucker,
                latent_token_count=x.shape[1],
                forward_batch=forward_batch,
            )  # (B, T*H*W, D)

        if self.use_chunk_plucker_input and plucker_emb is not None:
            x = _sana_wm_add_repeated_batch(
                x,
                plucker_emb,
                name="Plucker input embedding",
            )

        if not self.use_chunk_plucker_post_attn:
            plucker_emb = None

        # SANA-WM currently supports TP/FSDP/CFG, but not true SP. Keep this
        # guard close to the token path so accidental SP runs fail clearly.
        _sana_wm_sequence_shard_enabled(_sana_wm_sp_world_size())

        # --- 6. Transformer blocks ---
        HW = (T, H, W)
        for block in self.blocks:
            x = block(
                x,
                y=y,
                t=t6,
                HW=HW,
                rotary_emb=freqs,
                ucpe_raymat_bundle=ucpe_raymat_bundle,
                plucker_emb=plucker_emb,
                mask=encoder_attention_mask,
                chunk_size=chunk_size,
                chunk_split_strategy=chunk_split_strategy,
                chunk_index=chunk_index,
                forward_batch=forward_batch,
            )

        # --- 7. Final layer ---
        x = self.final_layer(x, t_emb)  # (B, N, p_t*p_h*p_w*C_out)

        # --- 8. Un-patch ---
        x = x.reshape(B, T, H, W, p_t, p_h, p_w, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(B, self.out_channels, T * p_t, H * p_h, W * p_w)
        return x


EntryClass = SanaWMTransformer3DModel
