from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from functools import lru_cache, partial
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.jit_kernel.norm import can_use_fused_inplace_qknorm as can_use_jit_qk_norm
from sglang.srt.environ import envs
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_exec, get_mm, get_parallel
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_capability,
    is_blackwell_supported,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    print_info_once,
    use_intel_xpu_backend,
)
from sglang.srt.utils.multi_stream_utils import (
    maybe_execute_in_parallel,
    with_multi_stream,
)

_is_cpu = is_cpu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_npu = is_npu()
_is_hip = is_hip()
_is_cpu_amx_available = cpu_has_amx_support()
_is_xpu = is_xpu()

if _is_cuda:
    from flashinfer.prefill import cudnn_batch_prefill_with_kv_cache

    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

    def flash_attn_func(*args, ver: int = 3, **kwargs):
        if ver == 4:
            from sglang.kernels.ops.attention.flash_attention_v4 import (
                flash_attn_varlen_func as flash_attn_varlen_func_fa4,
            )

            return flash_attn_varlen_func_fa4(*args, **kwargs)
        return flash_attn_varlen_func(*args, **kwargs)


if _is_cpu and _is_cpu_amx_available:
    flash_attn_varlen_func = torch.ops.sgl_kernel.flash_attn_varlen_func

if _is_musa:
    from flash_attn_interface import flash_attn_varlen_func

if _is_npu:
    import torch_npu
if _is_xpu:
    from sgl_kernel.flash_attn import flash_attn_varlen_func

from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd
from sglang.srt.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
)
from sglang.srt.distributed import utils as dist_utils
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_pos_emb_native_eager
from sglang.srt.utils import add_prefix

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

ROTARY_EMBED_CLASSES = {
    "normal": apply_rotary_pos_emb,
}

# === Vision Encoder === #
FLASHINFER_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024

# Batch buckets for cuDNN graph caching - graphs are cached per bucket size
# This avoids creating a new graph for each unique batch size at runtime
BATCH_BUCKETS = [8, 16, 32, 64]

# Bucketized max seqlens to reduce cuDNN recompilation frequency while
# preserving a tighter upper bound than a single fixed max seqlen.
FLASHINFER_MAX_SEQLEN_BUCKETS = [
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
]


@dataclasses.dataclass
class SingletonCache:
    data: Any = None
    _max_seqlen: Optional[int] = None

    def set_data(self, value: Any) -> None:
        self.data = value
        self._max_seqlen = None

    def get_data(self) -> Optional[Any]:
        return self.data

    def empty(self) -> bool:
        return self.get_data() is None


@dataclasses.dataclass
class VisionAttentionMetadata:

    cu_seqlens: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen: int

    # flashinfer_cudnn specific (optional)
    packed_indptrs: Optional[torch.Tensor] = None
    sequence_lengths: Optional[torch.Tensor] = None
    flashinfer_max_seqlen: Optional[int] = None


def prepare_vision_attention_metadata(
    cu_seqlens: torch.Tensor,
    device: torch.device,
    *,
    packed_indptrs: Optional[torch.Tensor] = None,
    sequence_lengths: Optional[torch.Tensor] = None,
    flashinfer_max_seqlen: Optional[int] = None,
) -> VisionAttentionMetadata:
    # Compute all attention metadata once before the encoder layer loop.

    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32, non_blocking=True)
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(seq_lens.max().item())
    return VisionAttentionMetadata(
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        max_seqlen=max_seqlen,
        packed_indptrs=packed_indptrs,
        sequence_lengths=sequence_lengths,
        flashinfer_max_seqlen=flashinfer_max_seqlen,
    )


# TODO: requires real seqlens from images
@functools.lru_cache(maxsize=128)
def _get_cu_seqlens_for_shape(batch_size: int, seqlen: int, device) -> torch.Tensor:
    """
    Generates cumulative sequence lengths (cu_seqlens) for a given batch_size, seqlen, and device.
    Caches the result based on these parameters.
    """
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * seqlen,
        step=seqlen,
        dtype=torch.int32,
        device=device,
    )
    return cu_seqlens


def resolve_seqlens(
    cu_seqlens: torch.Tensor | SingletonCache | None,
    bsz: int,
    seq_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    if cu_seqlens is None:
        resolved_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=device)
    elif isinstance(cu_seqlens, SingletonCache):
        if cu_seqlens.empty():
            cu_seqlens.set_data(_get_cu_seqlens_for_shape(bsz, seq_len, device=device))
        resolved_seqlens = cu_seqlens.get_data()
    else:
        resolved_seqlens = cu_seqlens
    assert isinstance(
        resolved_seqlens, torch.Tensor
    ), "cu_seqlens must be a torch.Tensor"
    return resolved_seqlens


def resolve_max_seqlen(
    source: torch.Tensor | SingletonCache | None, cu_seqlens: torch.Tensor
) -> int:
    """Return the maximum segment length, caching only on ``SingletonCache``.

    Raw tensors have no mutable instance dictionary, so caching on them would
    raise ``AttributeError``. They use the same calculation without a cache.
    """
    if isinstance(source, SingletonCache):
        cached = getattr(source, "_max_seqlen", None)
        if cached is None:
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            cached = int(seq_lens.max().item())
            source._max_seqlen = cached
        return cached
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    return int(seq_lens.max().item())


def resolve_precomputed_max_seqlen(
    cu_seqlens: torch.Tensor, max_seqlen: int | torch.Tensor | None
) -> int:
    """Use an encoder-provided max sequence length when one is available.

    Packed vision encoders execute many attention blocks for one image batch.
    Deriving the max from GPU ``cu_seqlens`` in every block synchronizes the
    launch stream, whereas the encoder can materialize this host scalar once.
    """
    if max_seqlen is None:
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        return int(seq_lens.max().item())
    if isinstance(max_seqlen, torch.Tensor):
        return int(max_seqlen.item())
    return int(max_seqlen)


class VisionSdpaAttention(nn.Module):
    r"""
    Scaled Dot Product Attention inner product

    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        flatten_batch: bool = False,
        softmax_in_single_precision: bool = False,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = flatten_batch
        self.softmax_in_single_precision = softmax_in_single_precision
        self.dropout = dropout
        self.scale = (
            softmax_scale
            if softmax_scale is not None
            else 1.0 / math.sqrt(self.head_size)
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _generate_mask_cache(
        s: int, flatten_batch: bool, cu_seqlens: tuple
    ) -> torch.BoolTensor:
        """
        Generate a boolean attention mask with caching mechanism.
        Args:
            s: sequence length
            flatten_batch: whether to flatten batch dimension
            cu_seqlens: tuple of cumulative sequence lengths
        Returns:
            attention mask tensor of shape [b, 1, s, s] or [1, s, s]
        """
        if flatten_batch:
            mask = torch.zeros([1, s, s], dtype=torch.bool)
            for i in range(1, len(cu_seqlens)):
                start = cu_seqlens[i - 1]
                end = cu_seqlens[i]
                mask[..., start:end, start:end] = True
        else:
            # [1, 1, 1, s]
            row_indices = torch.arange(s).view(1, 1, 1, s)
            # [1, 1, s, 1]
            col_indices = torch.arange(s).view(1, 1, s, 1)
            # [b, 1, 1, 1]
            seq_lens = torch.tensor(
                [end - start for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])],
            ).view(-1, 1, 1, 1)

            mask = (row_indices < seq_lens) & (col_indices < seq_lens)

        return mask

    def generate_patch_attention_mask(
        self,
        s: int,
        cu_seqlens: Optional[torch.Tensor],
        flatten_batch: bool = False,
    ) -> Optional[torch.Tensor]:
        r"""
        Creates a non-causal 4D mask of shape `(b, 1, s, s)` or `(1, 1, s, s)`.
        Args:
            s: sequence length
            cu_seqlens: cumulative sequence lengths tensor. If not, returns an empty mask
            flatten_batch: whether to flatten batch dimension
        Returns:
            attention mask tensor or None
        """
        if cu_seqlens is None:
            return None

        cu_seqlens_tuple = tuple(cu_seqlens.cpu().tolist())

        return self._generate_mask_cache(s, flatten_batch, cu_seqlens_tuple)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bsz: int,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if self.flatten_batch:
            assert bsz == 1, "flatten_batch is True, bsz must be 1"

        assert q.dim() == 3, q.shape

        s = q.shape[0] // bsz

        # [b, 1, s, s]
        if attention_mask is None:
            attention_mask = self.generate_patch_attention_mask(
                s, cu_seqlens, flatten_batch=self.flatten_batch
            )

        if attention_mask is None:
            if self.softmax_in_single_precision:
                raise RuntimeError("Empty attention mask")
        else:
            attention_mask = attention_mask.to(device=q.device)

        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]

        if self.softmax_in_single_precision:
            k = rearrange(k, "b h s d -> b h d s")
            attn_weights = torch.matmul(q, k) * self.scale
            del k
            # masking
            attention_mask = (~attention_mask) * torch.finfo(q.dtype).min
            attn_weights = attn_weights + attention_mask
            del attention_mask
            # full-precision
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=False
            )
            output = torch.matmul(attn_weights, v)
            del attn_weights, v
        else:
            # SDPA
            # [b, h, s, head_size]
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout,
                is_causal=False,
                scale=self.scale,
            )

        # [b, h, s, head_size] --> [b * s, h, head_size]
        output = rearrange(output, "b h s d -> (b s) h d")

        return output


class VisionTritonAttention(nn.Module):
    """
    Triton-implemented attention without a causal mask
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        use_data_parallel = (
            kwargs["use_data_parallel"] if "use_data_parallel" in kwargs else False
        )
        self.tp_size = 1 if use_data_parallel else get_parallel().attn_tp_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | list | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        forward_metadata: Optional[VisionAttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if forward_metadata is not None:
            cu_seqlens_gpu = forward_metadata.cu_seqlens
            seq_lens = forward_metadata.seq_lens
            max_seqlen = forward_metadata.max_seqlen
            output = torch.empty_like(q)
        elif envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            if "output_ws" not in kwargs:
                raise RuntimeError("output_ws should be prepared for cuda-graph mode")

            if not isinstance(cu_seqlens, list):
                raise RuntimeError("cuda-graph mode cu_seqlens should be a list")

            output = kwargs["output_ws"]
            cu_seqlens_gpu = cu_seqlens[0]
            seq_lens = cu_seqlens[1]
            max_seqlen = cu_seqlens[2]
        else:
            cu_seqlens_gpu = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
            seq_lens = kwargs.get("sequence_lengths")
            if seq_lens is None:
                seq_lens = cu_seqlens_gpu[1:] - cu_seqlens_gpu[:-1]
            else:
                seq_lens = seq_lens.to(device=q.device, dtype=torch.int32)
            max_seqlen = resolve_precomputed_max_seqlen(
                cu_seqlens_gpu, kwargs.get("max_seqlen")
            )
            # [b * s, head, head_size]
            output = torch.empty_like(q)

        context_attention_fwd(
            q,
            k,
            v,
            output,
            cu_seqlens_gpu,
            seq_lens,
            max_seqlen,
            is_causal=False,
            sm_scale=softmax_scale,
        )

        return output


class VisionFlash3Attention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not (_is_cuda or _is_musa):
            raise Exception("VisionFlash3Attention is only available for cuda or musa")
        super().__init__()
        use_data_parallel = (
            kwargs["use_data_parallel"] if "use_data_parallel" in kwargs else False
        )
        self.tp_size = 1 if use_data_parallel else get_parallel().attn_tp_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | list | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        forward_metadata: Optional[VisionAttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        window_size = kwargs.get("window_size", (-1, -1))
        s_aux = kwargs.get("s_aux", None)

        if forward_metadata is not None:
            cu_seqlens_gpu = forward_metadata.cu_seqlens
            max_seqlen = forward_metadata.max_seqlen
        elif envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            if not isinstance(cu_seqlens, list):
                raise RuntimeError("cuda-graph mode cu_seqlens should be a list")
            cu_seqlens_gpu = cu_seqlens[0]
            max_seqlen = cu_seqlens[1]
        else:
            cu_seqlens_gpu = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
            cu_seqlens_gpu = cu_seqlens_gpu.to(dtype=torch.int32).to(q.device)
            max_seqlen = resolve_precomputed_max_seqlen(
                cu_seqlens_gpu, kwargs.get("max_seqlen")
            )

        fa_kwargs = dict(
            cu_seqlens_q=cu_seqlens_gpu,
            cu_seqlens_k=cu_seqlens_gpu,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            window_size=window_size,
        )
        if s_aux is not None:
            fa_kwargs["sinks"] = s_aux
        output = flash_attn_func(q, k, v, **fa_kwargs)

        return output


class VisionFlash4Attention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlash4Attention is only available for cuda")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        forward_metadata: Optional[VisionAttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if forward_metadata is not None:
            cu_seqlens_gpu = forward_metadata.cu_seqlens
            max_seqlen = forward_metadata.max_seqlen
        else:
            cu_seqlens_gpu = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
            cu_seqlens_gpu = cu_seqlens_gpu.to(dtype=torch.int32).to(q.device)
            max_seqlen = resolve_precomputed_max_seqlen(
                cu_seqlens_gpu, kwargs.get("max_seqlen")
            )

        output = flash_attn_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_gpu,
            cu_seqlens_k=cu_seqlens_gpu,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            ver=4,
        )

        return output


class VisionFlashInferAttention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cuda:
            raise Exception("VisionFlashInferAttention is only available for cuda")
        super().__init__()
        self.workspace_buffer = (
            kwargs["workspace_buffer"] if "workspace_buffer" in kwargs else None
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        forward_metadata: Optional[VisionAttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        # ---- resolve sequence_lengths, packed indptrs, max_seqlen ----
        if forward_metadata is not None and forward_metadata.packed_indptrs is not None:
            sequence_lengths = forward_metadata.sequence_lengths
            packed_cu_seqlens = forward_metadata.packed_indptrs
            max_seqlen = forward_metadata.flashinfer_max_seqlen
        else:
            if "sequence_lengths" not in kwargs:
                raise RuntimeError(
                    "sequence_lengths should be prepared for vision flashinfer_cudnn attention backend"
                )
            if "max_seqlen" not in kwargs:
                raise RuntimeError(
                    "max_seqlen should be prepared for vision flashinfer_cudnn attention backend"
                )
            sequence_lengths = kwargs["sequence_lengths"]
            packed_cu_seqlens = cu_seqlens
            max_seqlen = kwargs["max_seqlen"]

        # max_seqlen must be python int
        if isinstance(max_seqlen, torch.Tensor):
            if max_seqlen.is_cuda:
                max_seqlen = int(max_seqlen.detach().cpu().item())
            else:
                max_seqlen = int(max_seqlen.item())
        else:
            max_seqlen = int(max_seqlen)

        # flatten if caller gives (b, s, h, d)
        is_reshaped = q.dim() == 4
        if is_reshaped:
            reshape_batch_size = q.shape[0]
            q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

        if not isinstance(packed_cu_seqlens, torch.Tensor):
            raise RuntimeError(
                "flashinfer_cudnn expects packed indptrs as a torch.Tensor"
            )

        # sequence_lengths -> (B,)
        if not isinstance(sequence_lengths, torch.Tensor):
            raise RuntimeError("sequence_lengths must be a torch.Tensor")
        seq_lens_1d = sequence_lengths.view(-1).to(device=q.device, dtype=torch.int32)
        B = int(seq_lens_1d.numel())

        # cu_seqlens contains packed *element indptrs*:
        # [qk_indptr(B+1), v_indptr(B+1), o_indptr(B+1)] => total 3*(B+1)
        cu_seqlens_1d = packed_cu_seqlens.view(-1).to(
            device=q.device, dtype=torch.int32
        )
        expected = 3 * (B + 1)
        if int(cu_seqlens_1d.numel()) != expected:
            raise RuntimeError(
                f"packed indptr numel mismatch: got {cu_seqlens_1d.numel()}, expected {expected} (= 3*(B+1))"
            )

        split = B + 1
        indptr_qk = cu_seqlens_1d[:split].view(split, 1, 1, 1)
        indptr_v = cu_seqlens_1d[split : 2 * split].view(split, 1, 1, 1)
        indptr_o = cu_seqlens_1d[2 * split :].view(split, 1, 1, 1)

        # cuDNN style: (B,1,1,1)
        seq_lens_4d = seq_lens_1d.view(B, 1, 1, 1)

        # indptr are in ELEMENT offsets (not token offsets)
        token_width_q = int(q.shape[1] * q.shape[2])  # heads * head_dim on this rank
        total_elems_q = int(q.numel())

        # check each real sequence fits
        # (skip padded tail where seq_len==0)
        start_elems = indptr_qk.view(-1)[:-1]  # (B,)
        end_elems = start_elems + seq_lens_1d * token_width_q
        if (end_elems > total_elems_q).any():
            raise RuntimeError("offset + len out of bounds; packed indptr is wrong")

        _, _, head_size = q.shape
        scale = softmax_scale if softmax_scale is not None else head_size**-0.5

        output, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            scale,
            self.workspace_buffer,
            max_token_per_sequence=max_seqlen,
            max_sequence_kv=max_seqlen,
            actual_seq_lens_q=seq_lens_4d,
            actual_seq_lens_kv=seq_lens_4d,
            causal=False,
            return_lse=True,
            batch_offsets_q=indptr_qk,
            batch_offsets_k=indptr_qk,
            batch_offsets_v=indptr_v,
            batch_offsets_o=indptr_o,
            is_cuda_graph_compatible=True,
        )

        if is_reshaped:
            output = rearrange(output, "(b s) h d -> b s h d", b=reshape_batch_size)

        return output


class VisionAiterAttention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_hip:
            raise Exception("aiter_attn is only available for AMD")
        try:
            from aiter import flash_attn_varlen_func as aiter_flash_attn_varlen_func
        except ImportError as e:
            raise ImportError(
                "aiter is AMD specific kernel library. Please make sure aiter is installed on your AMD device."
            ) from e

        self.flash_attn_varlen_func = aiter_flash_attn_varlen_func
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        forward_metadata: Optional[VisionAttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if forward_metadata is not None:
            cu_seqlens_gpu = forward_metadata.cu_seqlens
            max_seqlen = forward_metadata.max_seqlen
        else:
            cu_seqlens_gpu = resolve_seqlens(cu_seqlens, bsz, seq_len, device=q.device)
            cu_seqlens_gpu = cu_seqlens_gpu.to(dtype=torch.int32).to(q.device)
            seq_lens = cu_seqlens_gpu[1:] - cu_seqlens_gpu[:-1]
            max_seqlen = seq_lens.max().item()

        return self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_gpu,
            cu_seqlens_k=cu_seqlens_gpu,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
        )


class VisionAscendAttention(nn.Module):

    def __init__(
        self,
        **kwargs,
    ):
        if not _is_npu:
            raise Exception("VisionAscendAttention is only available for ascend npu")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        forward_metadata: Optional[VisionAttentionMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if forward_metadata is not None:
            # TND fused attention expects cumulative seqlens (cu_seqlens[1:]),
            # not per-sequence lengths in forward_metadata.seq_lens.
            cu = forward_metadata.cu_seqlens.to("cpu")
            output = torch.empty_like(q)
            seq_len_arg = cu[1:].to(torch.int32)
        elif envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            if "output_ws" not in kwargs:
                raise RuntimeError("output_ws should be prepared for npu-graph mode")
            output = kwargs["output_ws"]
            seq_len_arg = cu_seqlens
        else:
            cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device="cpu")
            seq_len_arg = cu_seqlens[1:].to(torch.int32)

        _, num_heads, head_size = q.shape
        num_kv_heads = k.shape[1]

        scale_value = softmax_scale if softmax_scale is not None else head_size**-0.5

        seq_len_arg = seq_len_arg.tolist()
        output = torch_npu.npu_fused_infer_attention_score(
            query=q,
            key=k,
            value=v,
            actual_seq_lengths=seq_len_arg,
            actual_seq_lengths_kv=seq_len_arg,
            scale=scale_value,
            num_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            sparse_mode=0,
            input_layout="TND",
        )[0]
        return output


class VisionAMXAttention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not _is_cpu or not _is_cpu_amx_available:
            raise Exception(
                "VisionAMXAttention is only available for cpu with amx support"
            )
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        if cu_seqlens is None:
            cu_seqlens = _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
        elif isinstance(cu_seqlens, SingletonCache):
            if cu_seqlens.empty():
                cu_seqlens.set_data(
                    _get_cu_seqlens_for_shape(bsz, seq_len, device=q.device)
                )
            cu_seqlens = cu_seqlens.get_data()

        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seq_lens.max().item()

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
        )

        return output


class VisionIntelXPUAttention(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        if not (_is_xpu):
            raise Exception("VisionIntelXPUAttention is only available for Intel XPU")
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor | SingletonCache | None,
        bsz: int,
        seq_len: int,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            cu_seqlens: [b]
        Returns:
             [b * s, h, head_size]
        """
        window_size = kwargs.get("window_size", (-1, -1))
        s_aux = kwargs.get("s_aux", None)

        cu_seqlens_source = cu_seqlens
        cu_seqlens = resolve_seqlens(cu_seqlens_source, bsz, seq_len, device=q.device)
        cu_seqlens = cu_seqlens.to(dtype=torch.int32).to(q.device)
        max_seqlen = resolve_max_seqlen(cu_seqlens_source, cu_seqlens)

        fa_kwargs = dict(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            window_size=window_size,
        )
        if s_aux is not None:
            fa_kwargs["sinks"] = s_aux
        output = flash_attn_varlen_func(q, k, v, **fa_kwargs)

        return output


QKV_BACKEND_IMPL = {
    "triton_attn": VisionTritonAttention,
    "sdpa": VisionSdpaAttention,
    "fa3": VisionFlash3Attention,
    "fa4": VisionFlash4Attention,
    "flashinfer_cudnn": VisionFlashInferAttention,
    "ascend_attn": VisionAscendAttention,
    "aiter_attn": VisionAiterAttention,
    "amx_attn": VisionAMXAttention,
    "xpu_attn": VisionIntelXPUAttention,
}


class VisionAttention(nn.Module):
    r"""
        Multi-headed attention without any cache, mostly used for multimodal transformers.


    Args:
        use_qkv_parallel (bool, optional): If True, use QKV-parallel attention.
        softmax_in_single_precision (bool, default to False):
            if ``True``, the softmax will be performed in single-precision
            Otherwise, it will be performed in half-precision

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        use_qkv_parallel: bool,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        qkv_backend: Optional[str] = None,
        quant_config: Optional[QuantizationConfig] = None,
        dropout: float = 0.0,
        softmax_in_single_precision: bool = False,
        softmax_scale: Optional[float] = None,
        flatten_batch: bool = False,
        prefix: str = "",
        proj_bias: bool = True,
        num_dummy_heads: int = 0,
        qkv_bias: bool = True,
        qk_normalization: bool = False,
        qk_normalization_by_head_size: bool = False,
        layer_norm_eps: float = 1e-06,
        customized_position_embedding_applier: Callable[
            [torch.Tensor, torch.Tensor, Any, Any], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        use_data_parallel: bool = False,
        use_dp_attention_reduce: bool = False,
        aux_stream: Optional[torch.cuda.Stream] = None,
        workspace_buffer: Optional[torch.Tensor] = None,
        use_sink: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        **kwargs,
    ):
        super().__init__()
        if head_dim is None and "head_size" in kwargs:
            head_dim = kwargs.pop("head_size")
            warnings.warn(
                "VisionAttention(head_size=...) is deprecated; use head_dim=...",
                DeprecationWarning,
                stacklevel=2,
            )
        self.tp_size = 1 if use_data_parallel else get_parallel().attn_tp_size
        self.tp_rank = 0 if use_data_parallel else get_parallel().attn_tp_rank
        self.dropout = dropout
        num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_dim if head_dim is not None else embed_dim // num_heads
        self.softmax_scale = softmax_scale
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_heads, self.tp_size
        )
        self.num_attention_kv_heads_per_partition = dist_utils.divide(
            num_dummy_heads + num_kv_heads, self.tp_size
        )

        self.q_size = self.num_attention_heads_per_partition * self.head_size
        self.kv_size = self.num_attention_kv_heads_per_partition * self.head_size

        self.qk_normalization = qk_normalization
        self.qk_normalization_by_head_size = qk_normalization_by_head_size

        # Additional dummy heads are used to enable TP for common GPU counts.
        self.dummy_dim = (num_dummy_heads + num_heads) * self.head_size

        if self.qk_normalization:
            self.q_norm, self.k_norm = self._init_qk_norm(
                self.dummy_dim, layer_norm_eps, embed_dim
            )

        elif self.qk_normalization_by_head_size:
            self.q_norm, self.k_norm = self._init_qk_norm(
                self.head_size, layer_norm_eps
            )

        # Select attention backend via a unified method
        _passed_backend = qkv_backend
        qkv_backend = self._determine_attention_backend(_passed_backend)
        if get_mm().mm_attention_backend is None and _passed_backend is None:
            print_info_once(f"Multimodal attention backend not set. Use {qkv_backend}.")
        print_info_once(f"Using {qkv_backend} as multimodal attention backend.")

        # Keep the resolved name alongside the implementation. Graph runners need
        # the effective backend after applying CLI, model, and platform defaults;
        # reading the raw server argument again loses that information when it is
        # left unset.
        self.qkv_backend_name: str = qkv_backend

        self.customized_position_embedding_applier = (
            customized_position_embedding_applier
        )
        self.qkv_backend = QKV_BACKEND_IMPL[qkv_backend](
            head_dim=self.head_size,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_kv_heads_per_partition,
            dropout=dropout,
            flatten_batch=flatten_batch,
            softmax_in_single_precision=softmax_in_single_precision,
            softmax_scale=softmax_scale,
            use_data_parallel=use_data_parallel,
            workspace_buffer=workspace_buffer,
        )

        self.use_qkv_parallel = use_qkv_parallel
        if use_qkv_parallel:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=embed_dim,
                head_size=self.head_size,
                total_num_heads=num_dummy_heads + num_heads,
                total_num_kv_heads=num_dummy_heads + num_kv_heads,
                bias=qkv_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
        else:
            self.qkv_proj = ColumnParallelLinear(
                input_size=embed_dim,
                output_size=3 * self.dummy_dim,
                bias=qkv_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                prefix=add_prefix("qkv_proj", prefix),
            )
        self.proj = RowParallelLinear(
            input_size=self.dummy_dim,
            output_size=embed_dim,
            bias=proj_bias,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            prefix=add_prefix("proj", prefix),
            use_dp_attention_reduce=use_dp_attention_reduce,
        )

        self.workspace_buffer = workspace_buffer
        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()] if aux_stream else []

        self.window_size = window_size
        if use_sink:
            # Allocate the full (unsharded) sink tensor for weight loading;
            # only the local TP slice is used in forward.
            self.sinks = nn.Parameter(
                torch.empty(
                    self.num_attention_heads_per_partition * self.tp_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
        else:
            self.sinks = None

    def _init_qk_norm(
        self, norm_dim: int, eps: float, var_hidden_size: Optional[int] = None
    ):
        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_exec().deterministic.rl_on_policy_target is not None
            else {}
        )
        q_norm = RMSNorm(
            norm_dim,
            eps=eps,
            var_hidden_size=var_hidden_size,
            **norm_kwargs,
        )
        k_norm = RMSNorm(
            norm_dim,
            eps=eps,
            var_hidden_size=var_hidden_size,
            **norm_kwargs,
        )
        return q_norm, k_norm

    def _determine_attention_backend(self, passed_backend: Optional[str]) -> str:
        """Decide the multimodal attention backend string.

        Priority: server args override > constructor arg > platform default.

        Platform defaults:
        - CUDA (Hopper SM90): "fa3"
        - CUDA (Blackwell SM100): "fa4"
        - CUDA (other): "triton_attn"
        - Ascend NPU: "ascend_attn"
        - Other platforms: device-specific optimized backend or "sdpa"
        """
        override_backend = get_mm().mm_attention_backend
        if override_backend is not None:
            backend = override_backend
        elif passed_backend is not None:
            backend = passed_backend
        elif is_cuda():
            major, minor = get_device_capability()
            if major == 9:
                backend = "fa3"
            elif major == 10 and minor != 3:
                backend = "fa4"
            else:
                backend = "triton_attn"
        elif _is_npu:
            backend = "ascend_attn"
        elif _is_musa:
            if get_device_capability() >= (3, 1):
                backend = "fa3"
            else:
                backend = "triton_attn"
        elif _is_hip:
            if get_device_capability() >= (9, 4) and _use_aiter:
                backend = "aiter_attn"
            else:
                backend = "triton_attn"
        elif _is_cpu and _is_cpu_amx_available:
            backend = "amx_attn"
        elif _is_xpu:
            backend = "triton_attn" if not use_intel_xpu_backend() else "xpu_attn"
        else:
            backend = "sdpa"
        if backend == "fa3" and is_blackwell_supported():
            raise ValueError("The 'fa3' backend is not supported on Blackwell GPUs")

        return backend

    def _apply_qk_norm_head_size(self, q: torch.Tensor, k: torch.Tensor):
        """apply qk norm for GLM-OCR vit attn"""
        q_by_head = q.reshape(-1, self.head_size)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = k.reshape(-1, self.head_size)
        k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor):
        """apply qk norm for internvl vit attn"""

        def q_l2norm():
            q_ = q.flatten(1, 2)
            if self.tp_size > 1:
                q_ = tensor_model_parallel_all_gather(q_.contiguous())
            q_ = self.q_norm(q_)
            if self.tp_size > 1:
                splitter = partial(
                    split_tensor_along_last_dim, num_partitions=self.tp_size
                )
                q_ = splitter(q_)[self.tp_rank]
            q_ = q_.unflatten(-1, (-1, self.head_size))
            return q_

        def k_l2norm():
            k_ = k.flatten(1, 2)
            if self.tp_size > 1:
                k_ = tensor_model_parallel_all_gather(k_.contiguous())
            k_ = self.k_norm(k_)
            if self.tp_size > 1:
                splitter = partial(
                    split_tensor_along_last_dim, num_partitions=self.tp_size
                )
                k_ = splitter(k_)[self.tp_rank]
            k_ = k_.unflatten(-1, (-1, self.head_size))
            return k_

        with with_multi_stream(True):
            q, k = maybe_execute_in_parallel(
                q_l2norm,
                k_l2norm,
                self.ln_events,
                self.aux_stream,
            )
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_metadata: Optional[VisionAttentionMetadata] = None,
        full_attn: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            x: [b, s, embed_dim]
            cu_seqlens: [b]
        Returns:
             [s, b, head * head_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        assert x.dim() == 3, x.shape
        if (
            get_exec().deterministic.rl_on_policy_target is not None
            and position_embeddings is not None
        ):
            assert isinstance(position_embeddings, tuple), (
                "expected position_embeddings to be a tuple of two tensors,\n"
                f"but got {type(position_embeddings)}, change if needed"
            )
            position_embeddings = tuple(p.to(x.dtype) for p in position_embeddings)
        x_shape = x.shape
        bsz, s, _ = x_shape
        head = self.num_attention_heads_per_partition
        kv_head = self.num_attention_kv_heads_per_partition

        attn_output_ws = kwargs["output_ws"] if "output_ws" in kwargs else None
        max_seqlen = kwargs["max_seqlen"] if "max_seqlen" in kwargs else None
        sequence_lengths = (
            kwargs["sequence_lengths"] if "sequence_lengths" in kwargs else None
        )
        if self.use_qkv_parallel:
            # [b, s, embed_dim] --> [b, s, embed_dim]
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [b, s, embed_dim] --> [b * s, head, head_size]
            q = q.reshape(bsz * s, head, -1)
            k = k.reshape(bsz * s, kv_head, -1)
            v = v.reshape(bsz * s, kv_head, -1)
        else:
            # [b, s, embed_dim] --> [s, b, embed_dim]
            x = rearrange(x, "b s ... -> s b ...")
            # [s, b, embed_dim] --> [s, b, head * 3 * head_size]
            qkv, _ = self.qkv_proj(x)

            # [s, b, head, head_dim_sum]
            new_x_shape = qkv.size()[:-1] + (
                head,
                self.q_size + 2 * self.kv_size,
            )
            qkv = qkv.view(*new_x_shape)

            # [s, b, head, 3 * head_size] --> 3 [s, b, head, head_size]
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # [s, b, head, head_size] --> [b, s, head, head_size]
            q, k, v = [rearrange(x, "s b ... -> b s ...") for x in (q, k, v)]

        if not (_is_cpu and _is_cpu_amx_available):
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        if self.qk_normalization_by_head_size:
            q, k = self._apply_qk_norm_head_size(q, k)

        cos = None
        sin = None

        if position_embeddings is not None:
            if self.customized_position_embedding_applier is not None:
                q, k = self.customized_position_embedding_applier(
                    q, k, position_embeddings, x_shape
                )
            else:
                cos, sin = position_embeddings
        elif rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            cos = rotary_pos_emb_cos
            sin = rotary_pos_emb_sin

        if cos is not None and sin is not None:
            original_q_shape = q.shape
            original_k_shape = k.shape

            # [total_tokens, head, head_size] for q / [total_tokens, kv_head, head_size] for k
            q = q.view(-1, head, self.head_size)
            k = k.view(-1, kv_head, self.head_size)

            if cos.size(-1) * 2 == self.head_size:
                cos = torch.cat([cos, cos], dim=-1)
                sin = torch.cat([sin, sin], dim=-1)

            # `apply_rotary_pos_emb` is torch.compile-decorated. Its first
            # specialization may otherwise be compiled while a ViT CUDA graph
            # is being captured, which makes Inductor attempt an illegal
            # CPU-to-CUDA copy. The eager version is captured as part of the
            # graph, so its pointwise work is still replayed without launch
            # overhead.
            rotary_fn = (
                apply_rotary_pos_emb_native_eager
                if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get()
                else apply_rotary_pos_emb
            )
            q, k = rotary_fn(q, k, cos, sin)
            q = q.view(original_q_shape)
            k = k.view(original_k_shape)

        if q.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            q = rearrange(q, "b s ... -> (b s) ...")
        if k.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            k = rearrange(k, "b s ... -> (b s) ...")
        if v.dim() == 4:
            # [b, s, head, head_size] --> [b * s, head, head_size]
            v = rearrange(v, "b s ... -> (b s) ...")

        assert q.dim() == 3, q.dim()
        assert k.dim() == 3, k.dim()
        assert v.dim() == 3, v.dim()

        # internvl
        if self.qk_normalization and not self.qk_normalization_by_head_size:
            # jit kernel
            if can_use_jit_qk_norm(self.head_size, q.dtype):

                # q: [tokens, head, head_size]  ->  [tokens, embed_dim]
                head_dim_for_norm = head * self.head_size

                q, k = apply_qk_norm(
                    q=q,
                    k=k,
                    q_norm=self.q_norm,
                    k_norm=self.k_norm,
                    head_dim=head_dim_for_norm,
                    alt_stream=self.aux_stream,
                )

            else:
                q, k = self._apply_qk_norm(q, k)

        if full_attn or self.sinks is None:
            effective_window_size = (-1, -1)
            s_aux = None
        else:
            effective_window_size = self.window_size
            q_head_start = self.tp_rank * self.num_attention_heads_per_partition
            q_head_end = (self.tp_rank + 1) * self.num_attention_heads_per_partition
            s_aux = self.sinks[q_head_start:q_head_end]

        output = self.qkv_backend.forward(
            q=q,
            k=k,
            v=v,
            bsz=bsz,
            seq_len=s,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            forward_metadata=forward_metadata,
            sequence_lengths=sequence_lengths,
            max_seqlen=max_seqlen,
            output_ws=attn_output_ws,
            softmax_scale=self.softmax_scale,
            window_size=effective_window_size,
            s_aux=s_aux,
        )

        assert output.dim() == 3, output.shape

        if self.use_qkv_parallel:
            # [b * s, h, head_size] --> [b, s, h * head_size]
            output = rearrange(output, "(b s) ... h d -> b s ... (h d)", b=bsz)

            # [b, s, h * head_size] --> [b, s, h * head_size]
            output, _ = self.proj(output)
        else:
            # [b * s, h, head_size] --> [s, b, h * head_size]
            context_layer = rearrange(
                output, "(b s) h d -> s b (h d)", b=bsz, s=s
            ).contiguous()

            # [s, b, h * head_size] --> [s, b, h * head_size]
            output, _ = self.proj(context_layer)

            # [s, b, h * head_size] --> [b, s, h * head_size]
            output = output.view(bsz, s, -1)

        return output
